use crate::message::{AgentEvent, ContentBlock, Message, Role, StopReason};
use crate::provider::ProviderError;
use crate::provider::sse::SseParser;
use crate::tool::ToolDefinition;
use futures_util::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const MAX_TOOL_CALLS_PER_TURN: usize = 64;

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

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    args_buf: String,
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

        // Read the body ourselves rather than `.json()` so a body that isn't
        // the shape we expect surfaces as ProviderError::Protocol, not
        // ProviderError::Network.
        let text = self.post(&body).await?.text().await?;
        let response: OpenAiResponse =
            serde_json::from_str(&text).map_err(|err| ProviderError::Protocol {
                detail: format!("could not decode response body: {err}"),
            })?;

        let choice =
            response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| ProviderError::Protocol {
                    detail: "response contained no choices".to_string(),
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

    /// Streams the response: sends `TextDelta` events out as they arrive and
    /// returns the complete assistant message once the turn's stream ends.
    pub(crate) async fn stream_message(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        events: &mpsc::Sender<AgentEvent>,
        cancel: &CancellationToken,
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
            stream: true,
            tools: openai_tools,
        };

        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
            result = self.post(&body) => result?,
        };
        let mut stream = response.bytes_stream();
        let mut parser = SseParser::default();

        let mut text = String::new();
        let mut tool_calls: Vec<PartialToolCall> = Vec::new();
        let mut finish = None;
        let mut saw_done = false;

        loop {
            let chunk = tokio::select! {
                _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                next = stream.next() => next,
            };

            let bytes = match chunk {
                Some(Ok(bytes)) => bytes,
                Some(Err(error)) => return Err(error.into()),
                None => break,
            };

            for event in parser.feed(&bytes)? {
                if event.data == "[DONE]" {
                    saw_done = true;
                    break;
                }

                let parsed_data = serde_json::from_str::<OpenAiStreamChunk>(&event.data);
                match parsed_data {
                    Ok(data) => {
                        for choice in data.choices {
                            if let Some(delta) = choice.delta.content {
                                text.push_str(&delta);

                                let _ = events.send(AgentEvent::TextDelta(delta)).await;
                            }

                            if let Some(delta_tool_calls) = choice.delta.tool_calls {
                                for delta in delta_tool_calls {
                                    if delta.index >= MAX_TOOL_CALLS_PER_TURN {
                                        return Err(ProviderError::Protocol {
                                            detail: format!("tool call index {} exceeds the per-turn cap", delta.index),
                                        });
                                    }

                                    if delta.index >= tool_calls.len() {
                                        tool_calls
                                            .resize_with(delta.index + 1, PartialToolCall::default);
                                    }

                                    let slot = &mut tool_calls[delta.index];

                                    if let Some(id) = delta.id {
                                        slot.id = id;
                                    }

                                    if let Some(function) = delta.function {
                                        if let Some(name) = function.name {
                                            slot.name = name;
                                        }
                                        if let Some(arguments) = function.arguments {
                                            slot.args_buf.push_str(&arguments);
                                        }
                                    }
                                }
                            }

                            if let Some(finish_reason) = choice.finish_reason {
                                finish = Some(finish_reason);
                            }
                        }
                    }
                    Err(error) => {
                        return Err(ProviderError::Protocol {
                            detail: format!("unable to parse response: {}", error),
                        });
                    }
                }
            }

            if saw_done {
                break;
            }
        }

        if let Some(finish) = finish {
            let mut content = Vec::new();

            if !text.is_empty() {
                content.push(ContentBlock::Text { text })
            }

            for tool_call in tool_calls {
                if tool_call.id.is_empty() || tool_call.name.is_empty() {
                    return Err(ProviderError::Protocol {
                        detail: "tool call id and name cannot be empty".to_string(),
                    });
                }

                let input = match serde_json::from_str(&tool_call.args_buf) {
                    Ok(value) => value,
                    Err(_) => serde_json::Value::String(tool_call.args_buf),
                };

                content.push(ContentBlock::ToolUse {
                    id: tool_call.id,
                    input,
                    name: tool_call.name,
                })
            }

            let has_tool_use = content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. }));

            let end_reason = if has_tool_use {
                StopReason::ToolUse
            } else {
                map_stop_reason(Some(&finish))
            };

            return Ok((
                Message {
                    role: Role::Assistant,
                    content,
                },
                end_reason,
            ));
        }

        Err(ProviderError::Protocol {
            detail: if saw_done {
                "stream completed ([DONE]) but no chunk carried a finish_reason".to_string()
            } else {
                "stream ended before a finish_reason arrived (truncated?)".to_string()
            },
        })
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
                // OpenAI-compat servers require each `tool` message to immediately
                // follow the assistant message carrying the matching `tool_calls`, so
                // emit tool results before any text within a user message.
                let (tool_results, others): (Vec<_>, Vec<_>) = message
                    .content
                    .iter()
                    .partition(|block| matches!(block, ContentBlock::ToolResult { .. }));

                for block in tool_results.into_iter().chain(others) {
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
                        }
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
        // Arrange
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

        // Assert
        // One internal message can fan out (tool results) or fold (assistant);
        // text-less content is null, empty tool_calls is ABSENT (not []), errors
        // get a prefix
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

    #[test]
    fn to_wire_emits_tool_results_before_text_in_a_mixed_user_message() {
        // OpenAI-compat servers reject a `user` message appearing between an
        // assistant `tool_calls` and its `tool` reply, so tool results must lead
        // regardless of block order within the source message.

        // Arrange
        let history = vec![Message {
            role: Role::User,
            content: vec![
                Text {
                    text: "and also, what about this?".to_string(),
                },
                ToolResult {
                    tool_use_id: "call_abc".to_string(),
                    content: "ok".to_string(),
                    is_error: false,
                },
            ],
        }];

        // Act
        let wire = serde_json::to_value(to_wire(&history)).unwrap();

        // Assert
        assert_eq!(
            wire,
            json!([
                { "role": "tool", "tool_call_id": "call_abc", "content": "ok" },
                { "role": "user", "content": "and also, what about this?" }
            ])
        );
    }

    #[test]
    fn from_wire_maps_a_text_response() {
        // Arrange
        let wire: OpenAiResponseMessage = serde_json::from_value(json!({
            "role": "assistant",
            "content": "It declares the cane package."
        }))
        .unwrap();

        // Act
        let message = from_wire(wire);

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
    }

    #[test]
    fn from_wire_maps_a_tool_call_response() {
        // Arrange
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

        // Act
        let message = from_wire(wire);

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
    }

    #[test]
    fn from_wire_keeps_malformed_tool_arguments_as_a_raw_string() {
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

        // Act
        let message = from_wire(wire);

        // Assert
        assert_eq!(
            message,
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
        // ToolDefinition -> wire: tagged "function", schema under "parameters"
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
        // Some compat servers 400 on "tools": [] so we don't include the key.

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
        // history, and the *next* request 400s far from the cause. Due to that
        // the stop reason must be decided from the message content, not the label.

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
        // A wobbly server sending "choices": [] is it's bug, not ours.

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
        let error = test_client(&server)
            .complete(&user_history(), &[])
            .await
            .unwrap_err();

        // Assert
        assert!(
            matches!(error, ProviderError::Protocol { .. }),
            "expected ProviderError::Protocol, got {error:?}"
        );
    }

    #[tokio::test]
    async fn complete_errors_on_a_non_json_response_body() {
        // Proxies occasionally return HTML error pages with a 200.

        // Arrange
        let server = MockServer::start().await;
        mount_response(
            &server,
            ResponseTemplate::new(200).set_body_string("<html>Bad Gateway</html>"),
        )
        .await;

        // Act
        let error = test_client(&server)
            .complete(&user_history(), &[])
            .await
            .unwrap_err();

        // Assert
        assert!(
            matches!(error, ProviderError::Protocol { .. }),
            "expected ProviderError::Protocol, got {error:?}"
        );
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

    // --- Step 5: streaming adapter (chat.completion.chunk -> our events) ---

    /// Wrap one `choices[0]` delta in a `chat.completion.chunk` envelope. A null
    /// `finish_reason` (the common case mid-stream) is passed as `None`.
    fn stream_chunk(delta: serde_json::Value, finish_reason: Option<&str>) -> serde_json::Value {
        json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1751980000,
            "model": "test-model",
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }]
        })
    }

    /// Serialize chunks as a data-only SSE stream terminated by `[DONE]` — the
    /// exact shape an OpenAI-compat server sends over the wire.
    fn sse_stream(chunks: &[serde_json::Value]) -> String {
        let mut body = String::new();
        for chunk in chunks {
            body.push_str(&format!("data: {chunk}\n\n"));
        }
        body.push_str("data: [DONE]\n\n");
        body
    }

    async fn mount_stream(server: &MockServer, body: String) {
        mount_response(
            server,
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .await;
    }

    /// Drain every event the adapter emitted. Called after the stream ends, so
    /// all sends have completed and `try_recv` sees the full sequence.
    fn drain_events(rx: &mut mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        events
    }

    /// Concatenate the payloads of every `TextDelta`, ignoring other events.
    fn joined_text_deltas(events: &[AgentEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn stream_message_emits_text_deltas_and_accumulates_the_message() {
        // Content fragments arrive across chunks; the adapter emits each as a
        // TextDelta *and* accumulates them into the final assistant message.

        // Arrange
        let server = MockServer::start().await;
        mount_stream(
            &server,
            sse_stream(&[
                stream_chunk(json!({ "role": "assistant", "content": "Hello" }), None),
                stream_chunk(json!({ "content": " world" }), None),
                stream_chunk(json!({}), Some("stop")),
            ]),
        )
        .await;
        let (tx, mut rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();

        // Act
        let (message, stop_reason) = test_client(&server)
            .stream_message(&user_history(), &[], &tx, &cancel)
            .await
            .unwrap();

        // Assert
        let events = drain_events(&mut rx);
        assert_eq!(joined_text_deltas(&events), "Hello world");
        assert_eq!(
            message,
            Message {
                role: Role::Assistant,
                content: vec![Text {
                    text: "Hello world".to_string(),
                }],
            }
        );
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn stream_message_accumulates_tool_call_argument_fragments() {
        // `function.arguments` streams as string fragments keyed by `index`; the
        // id and name land only on the first fragment. The adapter buffers per
        // index and parses the joined string once the stream ends. Tool
        // arguments are NOT surfaced as TextDelta.

        // Arrange
        let server = MockServer::start().await;
        mount_stream(
            &server,
            sse_stream(&[
                stream_chunk(
                    json!({
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": { "name": "read_file", "arguments": "" }
                        }]
                    }),
                    None,
                ),
                stream_chunk(
                    json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "{\"path\":" } }] }),
                    None,
                ),
                stream_chunk(
                    json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "\"Cargo.toml\"}" } }] }),
                    None,
                ),
                stream_chunk(json!({}), Some("tool_calls")),
            ]),
        )
            .await;
        let (tx, mut rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();

        // Act
        let (message, stop_reason) = test_client(&server)
            .stream_message(&user_history(), &[], &tx, &cancel)
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
        assert!(
            joined_text_deltas(&drain_events(&mut rx)).is_empty(),
            "tool-call arguments must not leak out as text deltas"
        );
    }

    #[tokio::test]
    async fn stream_message_accumulates_multiple_tool_calls_in_one_turn() {
        // A single turn can carry several tool calls, distinguished by `index`;
        // fragments for different indices interleave. Accumulate all of them.

        // Arrange
        let server = MockServer::start().await;
        mount_stream(
            &server,
            sse_stream(&[
                stream_chunk(
                    json!({
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": { "name": "read_file", "arguments": "{\"path\":\"a.txt\"}" }
                        }]
                    }),
                    None,
                ),
                stream_chunk(
                    json!({
                        "tool_calls": [{
                            "index": 1,
                            "id": "call_def",
                            "type": "function",
                            "function": { "name": "read_file", "arguments": "{\"path\":\"b.txt\"}" }
                        }]
                    }),
                    None,
                ),
                stream_chunk(json!({}), Some("tool_calls")),
            ]),
        )
        .await;
        let (tx, _rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();

        // Act
        let (message, stop_reason) = test_client(&server)
            .stream_message(&user_history(), &[], &tx, &cancel)
            .await
            .unwrap();

        // Assert
        assert_eq!(
            message,
            Message {
                role: Role::Assistant,
                content: vec![
                    ToolUse {
                        id: "call_abc".to_string(),
                        name: "read_file".to_string(),
                        input: json!({ "path": "a.txt" }),
                    },
                    ToolUse {
                        id: "call_def".to_string(),
                        name: "read_file".to_string(),
                        input: json!({ "path": "b.txt" }),
                    },
                ],
            }
        );
        assert_eq!(stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn stream_message_accumulates_text_and_a_tool_call_together() {
        // A model may emit prose and then call a tool in the same turn. Both the
        // text and the tool use land in the message, and the tool call wins the
        // stop reason.

        // Arrange
        let server = MockServer::start().await;
        mount_stream(
            &server,
            sse_stream(&[
                stream_chunk(
                    json!({ "role": "assistant", "content": "Let me check that file." }),
                    None,
                ),
                stream_chunk(
                    json!({
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": { "name": "read_file", "arguments": "{\"path\":\"Cargo.toml\"}" }
                        }]
                    }),
                    None,
                ),
                stream_chunk(json!({}), Some("tool_calls")),
            ]),
        )
            .await;
        let (tx, mut rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();

        // Act
        let (message, stop_reason) = test_client(&server)
            .stream_message(&user_history(), &[], &tx, &cancel)
            .await
            .unwrap();

        // Assert
        assert_eq!(
            joined_text_deltas(&drain_events(&mut rx)),
            "Let me check that file."
        );
        assert_eq!(
            message,
            Message {
                role: Role::Assistant,
                content: vec![
                    Text {
                        text: "Let me check that file.".to_string(),
                    },
                    ToolUse {
                        id: "call_abc".to_string(),
                        name: "read_file".to_string(),
                        input: json!({ "path": "Cargo.toml" }),
                    },
                ],
            }
        );
        assert_eq!(stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn stream_message_errors_when_the_stream_ends_without_done_or_finish_reason() {
        // DESIGN §11 invariant: only complete messages count. If the connection
        // closes after a content delta but before `[DONE]`/a finish_reason, the
        // partial must surface as an error, never be handed to the agent loop.

        // Arrange
        let server = MockServer::start().await;
        mount_stream(
            &server,
            // No `[DONE]`, no finish_reason — a truncated stream.
            format!(
                "data: {}\n\n",
                stream_chunk(json!({ "role": "assistant", "content": "Hel" }), None)
            ),
        )
        .await;
        let (tx, _rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();

        // Act
        let error = test_client(&server)
            .stream_message(&user_history(), &[], &tx, &cancel)
            .await
            .unwrap_err();

        // Assert
        assert!(
            matches!(error, ProviderError::Protocol { .. }),
            "expected ProviderError::Protocol, got {error:?}"
        );
    }

    #[tokio::test]
    async fn stream_message_errors_on_a_chunk_that_is_not_valid_json() {
        // Compat servers occasionally interleave a non-JSON data line. A chunk we
        // can't deserialize breaks the protocol contract — it is not model
        // feedback, so it aborts the turn as a ProviderError.

        // Arrange
        let server = MockServer::start().await;
        mount_stream(
            &server,
            "data: this is not json\n\ndata: [DONE]\n\n".to_string(),
        )
        .await;
        let (tx, _rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();

        // Act
        let error = test_client(&server)
            .stream_message(&user_history(), &[], &tx, &cancel)
            .await
            .unwrap_err();

        // Assert
        assert!(
            matches!(error, ProviderError::Protocol { .. }),
            "expected ProviderError::Protocol, got {error:?}"
        );
    }

    #[tokio::test]
    async fn stream_message_aborts_promptly_when_cancelled() {
        // DESIGN §3, non-negotiable: a tripped CancellationToken aborts the HTTP
        // stream. With the token already cancelled and the server stalling, the
        // call must return `Cancelled` at once rather than waiting on the wire.

        // Arrange
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_delay(Duration::from_secs(30))
                    .set_body_string(sse_stream(&[stream_chunk(
                        json!({ "role": "assistant", "content": "too late" }),
                        Some("stop"),
                    )])),
            )
            .mount(&server)
            .await;
        let (tx, _rx) = mpsc::channel(16);
        let cancel = CancellationToken::new();
        cancel.cancel();

        // Act
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            test_client(&server).stream_message(&user_history(), &[], &tx, &cancel),
        )
        .await
        .expect("stream_message should return promptly on cancellation, not hang");

        // Assert
        assert!(
            matches!(result, Err(ProviderError::Cancelled)),
            "expected ProviderError::Cancelled, got {result:?}"
        );
    }

    #[tokio::test]
    #[ignore = "requires a live server; run with -- --ignored"]
    async fn smoke_stream_message_against_a_live_server() {
        let base_url = std::env::var("CANE_BASE_URL").expect("set CANE_BASE_URL");
        let api_key = std::env::var("CANE_API_KEY").unwrap_or("none".to_string());
        let model = std::env::var("CANE_MODEL").expect("set CANE_MODEL");

        // Generous budget: thinking models (qwen3, deepseek-r1, ...) spend
        // tokens on reasoning before any content, and can burn 1000+ on it.
        let client = OpenAiClient::new(base_url, api_key, model, 8192).unwrap();
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Count to five".to_string(),
            }],
        }];

        // Drain concurrently: a live model can emit more deltas than the
        // channel holds, and stream_message blocks on a full channel.
        let (tx, mut rx) = mpsc::channel(16);
        let collector = tokio::spawn(async move {
            let mut events = Vec::new();
            while let Some(event) = rx.recv().await {
                events.push(event);
            }
            events
        });
        let cancel = CancellationToken::new();

        let (message, stop_reason) = client
            .stream_message(&messages, &[], &tx, &cancel)
            .await
            .unwrap();

        drop(tx); // close the channel so the collector finishes
        let events = collector.await.unwrap();

        dbg!(&message, &stop_reason, events.len());

        let streamed = joined_text_deltas(&events);
        assert!(!streamed.is_empty(), "expected TextDelta events");
        assert!(
            events.len() > 1,
            "expected the text to arrive as multiple deltas, not one blob"
        );
        assert_eq!(stop_reason, StopReason::EndTurn);

        // The deltas and the accumulated message must tell the same story.
        match &message.content[..] {
            [ContentBlock::Text { text }] => assert_eq!(text, &streamed),
            other => panic!("expected a single text block, got {other:?}"),
        }
    }
}
