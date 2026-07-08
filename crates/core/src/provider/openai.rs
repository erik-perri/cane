use crate::message::{ContentBlock, Message, Role, StopReason};
use crate::provider::ProviderError;
use crate::tool::ToolDefinition;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Eq, PartialEq, Serialize)]
struct OpenAiRequest {
    max_tokens: u32,
    messages: Vec<OpenAiRequestMessage>,
    model: String,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiRequestTool>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
struct OpenAiRequestTool {
    function: OpenAiRequestFunction,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
struct OpenAiRequestFunction {
    description: String,
    name: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
enum OpenAiRequestMessage {
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<OpenAiRequestToolCall>>,
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
}

#[derive(Debug, Eq, PartialEq, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiResponseChoice>,
}

#[derive(Debug, Eq, PartialEq, Deserialize)]
struct OpenAiResponseChoice {
    finish_reason: Option<String>,
    message: OpenAiResponseMessage,
}

/// Response-side wire types are Deserialize-only and deliberately tolerant:
/// every field a server might omit is Option, and unknown fields are ignored
/// (serde's default). Role is always "assistant" so we don't bother parsing it.
#[derive(Debug, Eq, PartialEq, Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiResponseToolCall>>,
}

#[derive(Debug, Eq, PartialEq, Deserialize)]
struct OpenAiResponseToolCall {
    function: OpenAiResponseFunctionCall,
    id: String,
}

#[derive(Debug, Eq, PartialEq, Deserialize)]
struct OpenAiResponseFunctionCall {
    arguments: String,
    name: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
struct OpenAiRequestToolCall {
    function: OpenAiRequestFunctionCall,
    id: String,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
struct OpenAiRequestFunctionCall {
    arguments: String,
    name: String,
}

pub(crate) struct OpenAiClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
    max_tokens: u32,
    model: String,
}

impl OpenAiClient {
    pub(crate) fn new(
        base_url: String,
        api_key: String,
        model: String,
        max_tokens: u32,
    ) -> Result<Self, ProviderError> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(600))
            .build()?;

        Ok(Self {
            api_key,
            base_url,
            http,
            max_tokens,
            model,
        })
    }

    pub(crate) async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason), ProviderError> {
        let openai_messages = to_wire(messages);
        let mut openai_tools = Vec::new();

        for tool in tools {
            openai_tools.push(OpenAiRequestTool {
                function: OpenAiRequestFunction {
                    description: tool.description.clone(),
                    name: tool.name.clone(),
                    parameters: tool.input_schema.clone(),
                },
                kind: "function".to_string(),
            });
        }

        let body = OpenAiRequest {
            max_tokens: self.max_tokens,
            messages: openai_messages,
            model: self.model.clone(),
            stream: false,
            tools: openai_tools,
        };

        let response = self.post(&body).await?.json::<OpenAiResponse>().await?;

        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or(ProviderError::Api {
                body: "No choices returned".to_string(),
                status: 400,
            })?;

        let message = from_wire(choice.message);
        let has_tool_use = message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. }));

        let end_reason = if has_tool_use {
            StopReason::ToolUse
        } else {
            map_stop_reason(choice.finish_reason.as_deref())
        };

        Ok((message, end_reason))
    }

    async fn post(&self, body: &OpenAiRequest) -> Result<reqwest::Response, ProviderError> {
        let response = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await?;

        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();

            return Err(ProviderError::Api {
                body,
                status: status.as_u16(),
            });
        }

        Ok(response)
    }
}

fn to_wire(messages: &[Message]) -> Vec<OpenAiRequestMessage> {
    let mut wire_messages: Vec<OpenAiRequestMessage> = Vec::new();

    for message in messages {
        match message.role {
            Role::User => {
                for block in &message.content {
                    match block {
                        ContentBlock::Text { text } => {
                            wire_messages.push(OpenAiRequestMessage::User {
                                content: text.clone(),
                            });
                        }
                        ContentBlock::ToolResult {
                            content,
                            is_error,
                            tool_use_id,
                        } => {
                            let content = if *is_error {
                                format!("Error: {}", content)
                            } else {
                                content.clone()
                            };
                            wire_messages.push(OpenAiRequestMessage::Tool {
                                content,
                                tool_call_id: tool_use_id.clone(),
                            });
                        }
                        ContentBlock::ToolUse { .. } => {
                            tracing::warn!("unexpected tool use content block")
                        }
                    }
                }
            }
            Role::Assistant => {
                let mut combined_text = String::new();
                let mut tool_calls = Vec::new();

                for block in &message.content {
                    match block {
                        ContentBlock::Text { text } => {
                            combined_text.push_str(text);
                        }
                        ContentBlock::ToolUse { id, input, name } => {
                            let arguments = serde_json::to_string(&input)
                                .expect("failed to serialize tool call arguments");

                            tool_calls.push(OpenAiRequestToolCall {
                                function: OpenAiRequestFunctionCall {
                                    arguments,
                                    name: name.clone(),
                                },
                                id: id.clone(),
                                kind: "function".to_string(),
                            });
                        }
                        ContentBlock::ToolResult { .. } => {
                            tracing::warn!("unexpected tool result content block")
                        },
                    }
                }

                let content = if combined_text.is_empty() {
                    None
                } else {
                    Some(combined_text)
                };

                let tool_calls = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };

                wire_messages.push(OpenAiRequestMessage::Assistant {
                    content,
                    tool_calls,
                })
            }
        }
    }

    wire_messages
}

fn from_wire(msg: OpenAiResponseMessage) -> Message {
    let mut content = Vec::new();

    if let Some(text) = msg.content
        && !text.is_empty()
    {
        content.push(ContentBlock::Text { text });
    }

    for tool_call in msg.tool_calls.unwrap_or_default() {
        let raw = tool_call.function.arguments;
        let input = match serde_json::from_str(&raw) {
            Ok(value) => value,
            Err(_) => serde_json::Value::String(raw),
        };

        content.push(ContentBlock::ToolUse {
            id: tool_call.id,
            input,
            name: tool_call.function.name,
        })
    }

    Message {
        content,
        role: Role::Assistant,
    }
}

fn map_stop_reason(finish_reason: Option<&str>) -> StopReason {
    match finish_reason {
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls") => StopReason::ToolUse,
        Some("stop") => StopReason::EndTurn,
        Some(other) => StopReason::Other(other.to_string()),
        None => StopReason::Other("finish_reason missing".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ContentBlock::{Text, ToolResult, ToolUse};
    use serde_json::json;

    #[test]
    fn to_wire_maps_a_full_tool_loop_history() {
        // Arrange: user prompt → assistant tool call → tool results → assistant answer
        let history = vec![
            Message {
                role: Role::User,
                content: vec![Text {
                    text: "What's in Cargo.toml?".to_string(),
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ToolUse {
                    id: "call_abc".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path": "Cargo.toml"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![
                    ToolResult {
                        tool_use_id: "call_abc".to_string(),
                        content: "[package]\nname = \"cane\"".to_string(),
                        is_error: false,
                    },
                    ToolResult {
                        tool_use_id: "call_def".to_string(),
                        content: "file not found: Cargo.lock".to_string(),
                        is_error: true,
                    },
                ],
            },
            Message {
                role: Role::Assistant,
                content: vec![Text {
                    text: "It declares the cane package.".to_string(),
                }],
            },
        ];

        // Act
        let wire = serde_json::to_value(to_wire(&history)).unwrap();

        // Assert: one internal message can fan out (tool results) or fold (assistant);
        // text-less content is null, empty tool_calls is ABSENT (not []), errors get a prefix
        assert_eq!(
            wire,
            json!([
                { "role": "user", "content": "What's in Cargo.toml?" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"Cargo.toml\"}"
                        }
                    }]
                },
                { "role": "tool", "tool_call_id": "call_abc", "content": "[package]\nname = \"cane\"" },
                { "role": "tool", "tool_call_id": "call_def", "content": "Error: file not found: Cargo.lock" },
                { "role": "assistant", "content": "It declares the cane package." }
            ])
        );
    }

    // ---- from_wire ------------------------------------------------------
    // These build the wire message by deserializing JSON fixtures rather than
    // constructing structs, so they also pin the response *parsing* — including
    // tolerance of fields we don't model (role, index, refusal, ...).

    #[test]
    fn from_wire_maps_a_text_response() {
        // Arrange
        let wire: OpenAiResponseMessage = serde_json::from_value(json!({
            "role": "assistant",
            "content": "It declares the cane package."
        }))
        .unwrap();

        // Act / Assert
        assert_eq!(
            from_wire(wire),
            Message {
                role: Role::Assistant,
                content: vec![Text {
                    text: "It declares the cane package.".to_string(),
                }],
            }
        );
    }

    #[test]
    fn from_wire_maps_a_tool_call_response() {
        // Arrange: content is null, arguments is a STRING of JSON — both per the wire format
        let wire: OpenAiResponseMessage = serde_json::from_value(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"Cargo.toml\"}"
                }
            }]
        }))
        .unwrap();

        // Act / Assert: arguments parsed into a Value; no empty Text block for null content
        assert_eq!(
            from_wire(wire),
            Message {
                role: Role::Assistant,
                content: vec![ToolUse {
                    id: "call_abc".to_string(),
                    name: "read_file".to_string(),
                    input: json!({ "path": "Cargo.toml" }),
                }],
            }
        );
    }

    #[test]
    fn from_wire_keeps_malformed_tool_arguments_as_a_raw_string() {
        // This pins the "model mistakes are model feedback" choice (DESIGN §11):
        // unparseable arguments become Value::String(raw) so tool dispatch can
        // reject them with an error tool result instead of aborting the turn.
        // If you go the ProviderError route instead, rewrite this test.
        // Also: empty-string content should not produce a Text block.

        // Arrange
        let wire: OpenAiResponseMessage = serde_json::from_value(json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_bad",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\": unclosed"
                }
            }]
        }))
        .unwrap();

        // Act / Assert
        assert_eq!(
            from_wire(wire),
            Message {
                role: Role::Assistant,
                content: vec![ToolUse {
                    id: "call_bad".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::Value::String("{\"path\": unclosed".to_string()),
                }],
            }
        );
    }

    // ---- map_stop_reason -------------------------------------------------

    #[test]
    fn map_stop_reason_maps_known_finish_reasons() {
        assert_eq!(map_stop_reason(Some("stop")), StopReason::EndTurn);
        assert_eq!(map_stop_reason(Some("tool_calls")), StopReason::ToolUse);
        assert_eq!(map_stop_reason(Some("length")), StopReason::MaxTokens);
    }

    #[test]
    fn map_stop_reason_preserves_unknown_finish_reasons() {
        // Compat servers emit values we don't model ("content_filter",
        // "function_call", ...) — carry them through rather than guessing.
        assert_eq!(
            map_stop_reason(Some("content_filter")),
            StopReason::Other("content_filter".to_string())
        );
        assert_eq!(
            map_stop_reason(None),
            StopReason::Other("finish_reason missing".to_string())
        );
    }

    // ---- complete ---------------------------------------------------------
    // End-to-end through a mock HTTP server: request assembly on the way out,
    // envelope parsing + stop-reason translation on the way back. Response
    // fixtures carry the fields real servers send but we don't model (id,
    // object, created, usage, index) so they double as tolerance tests.

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_client(server: &MockServer) -> OpenAiClient {
        OpenAiClient::new(
            server.uri(),
            "test-key".to_string(),
            "test-model".to_string(),
            1234,
        )
        .unwrap()
    }

    fn user_history() -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: vec![Text {
                text: "What's in Cargo.toml?".to_string(),
            }],
        }]
    }

    fn envelope(message: serde_json::Value, finish_reason: &str) -> serde_json::Value {
        json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1751980000,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": { "prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19 }
        })
    }

    fn tool_call_message() -> serde_json::Value {
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"Cargo.toml\"}"
                }
            }]
        })
    }

    async fn mount_response(server: &MockServer, response: ResponseTemplate) {
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(response)
            .expect(1)
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn complete_sends_a_well_formed_request() {
        // Arrange
        let server = MockServer::start().await;
        mount_response(
            &server,
            ResponseTemplate::new(200).set_body_json(envelope(
                json!({ "role": "assistant", "content": "hi" }),
                "stop",
            )),
        )
        .await;
        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file from disk".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        }];

        // Act
        test_client(&server)
            .complete(&user_history(), &tools)
            .await
            .unwrap();

        // Assert
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(
            request.headers.get("authorization").unwrap(),
            "Bearer test-key"
        );

        let body: serde_json::Value = request.body_json().unwrap();
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["stream"], false);
        assert!(body["max_tokens"].is_u64(), "max_tokens missing: {body}");
        assert_eq!(
            body["messages"],
            json!([{ "role": "user", "content": "What's in Cargo.toml?" }])
        );
        // ToolDefinition → wire: tagged "function", schema under "parameters"
        assert_eq!(
            body["tools"],
            json!([{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from disk",
                    "parameters": {
                        "type": "object",
                        "properties": { "path": { "type": "string" } },
                        "required": ["path"]
                    }
                }
            }])
        );
    }

    #[tokio::test]
    async fn complete_omits_the_tools_key_when_there_are_no_tools() {
        // Some compat servers 400 on "tools": [] — the key must be absent.

        // Arrange
        let server = MockServer::start().await;
        mount_response(
            &server,
            ResponseTemplate::new(200).set_body_json(envelope(
                json!({ "role": "assistant", "content": "hi" }),
                "stop",
            )),
        )
        .await;

        // Act
        test_client(&server)
            .complete(&user_history(), &[])
            .await
            .unwrap();

        // Assert
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = requests[0].body_json().unwrap();
        assert!(
            body.get("tools").is_none(),
            "tools should be absent: {body}"
        );
    }

    #[tokio::test]
    async fn complete_maps_a_text_response() {
        // Arrange
        let server = MockServer::start().await;
        mount_response(
            &server,
            ResponseTemplate::new(200).set_body_json(envelope(
                json!({ "role": "assistant", "content": "It declares the cane package." }),
                "stop",
            )),
        )
        .await;

        // Act
        let (message, stop_reason) = test_client(&server)
            .complete(&user_history(), &[])
            .await
            .unwrap();

        // Assert
        assert_eq!(
            message,
            Message {
                role: Role::Assistant,
                content: vec![Text {
                    text: "It declares the cane package.".to_string(),
                }],
            }
        );
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn complete_maps_a_tool_call_response() {
        // Arrange
        let server = MockServer::start().await;
        mount_response(
            &server,
            ResponseTemplate::new(200).set_body_json(envelope(tool_call_message(), "tool_calls")),
        )
        .await;

        // Act
        let (message, stop_reason) = test_client(&server)
            .complete(&user_history(), &[])
            .await
            .unwrap();

        // Assert
        assert_eq!(
            message,
            Message {
                role: Role::Assistant,
                content: vec![ToolUse {
                    id: "call_abc".to_string(),
                    name: "read_file".to_string(),
                    input: json!({ "path": "Cargo.toml" }),
                }],
            }
        );
        assert_eq!(stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn complete_overrides_the_finish_reason_when_the_message_has_tool_calls() {
        // Some compat backends ship finish_reason "stop" alongside tool calls.
        // Trusting the label ends the turn with unanswered tool calls in
        // history, and the *next* request 400s far from the cause — so the
        // stop reason must be decided from the message content, not the label.

        // Arrange
        let server = MockServer::start().await;
        mount_response(
            &server,
            ResponseTemplate::new(200).set_body_json(envelope(tool_call_message(), "stop")),
        )
        .await;

        // Act
        let (message, stop_reason) = test_client(&server)
            .complete(&user_history(), &[])
            .await
            .unwrap();

        // Assert
        assert_eq!(stop_reason, StopReason::ToolUse);
        assert!(
            message
                .content
                .iter()
                .any(|block| matches!(block, ToolUse { .. }))
        );
    }

    #[tokio::test]
    async fn complete_errors_instead_of_panicking_on_empty_choices() {
        // A wobbly server sending "choices": [] is its bug, not ours — but it
        // must surface as an Err, never an index panic.

        // Arrange
        let server = MockServer::start().await;
        mount_response(
            &server,
            ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1751980000,
                "model": "test-model",
                "choices": []
            })),
        )
        .await;

        // Act
        let result = test_client(&server).complete(&user_history(), &[]).await;

        // Assert (tighten to your parse-flavored variant once it has a name)
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn complete_errors_on_a_non_json_response_body() {
        // Proxies love returning HTML error pages with a 200.

        // Arrange
        let server = MockServer::start().await;
        mount_response(
            &server,
            ResponseTemplate::new(200).set_body_string("<html>Bad Gateway</html>"),
        )
        .await;

        // Act
        let result = test_client(&server).complete(&user_history(), &[]).await;

        // Assert (tighten to your parse-flavored variant once it has a name)
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn complete_surfaces_api_errors_with_status_and_body() {
        // Arrange
        let server = MockServer::start().await;
        mount_response(
            &server,
            ResponseTemplate::new(429).set_body_string("rate limited"),
        )
        .await;

        // Act
        let error = test_client(&server)
            .complete(&user_history(), &[])
            .await
            .unwrap_err();

        // Assert
        match error {
            ProviderError::Api { status, body } => {
                assert_eq!(status, 429);
                assert_eq!(body, "rate limited");
            }
            other => panic!("expected ProviderError::Api, got {other:?}"),
        }
    }

    #[tokio::test]
    #[ignore = "requires a live server; run with -- --ignored"]
    async fn smoke_complete_against_a_live_server() {
        let base_url = std::env::var("CANE_BASE_URL").expect("set CANE_BASE_URL");
        let api_key = std::env::var("CANE_API_KEY").unwrap_or("none".to_string());
        let model = std::env::var("CANE_MODEL").expect("set CANE_MODEL");

        let client = OpenAiClient::new(base_url, api_key, model, 1000).unwrap();
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Say hi".to_string(),
            }],
        }];
        let tools = Vec::new();

        let (message, stop_reason) = client.complete(&messages, &tools).await.unwrap();

        dbg!(&message);

        assert!(!message.content.is_empty());
        assert_eq!(stop_reason, StopReason::EndTurn);
    }
}
