use crate::message::{AgentEvent, ContentBlock, Message, Role, StopReason};
use crate::provider::{OpenAiClient, ProviderConfig, ProviderError};
use crate::{FileTool, Tool, ToolDefinition, dispatch};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct AgentHandle {
    pub events: mpsc::Receiver<AgentEvent>,
    pub cancel: CancellationToken,
}

pub fn spawn_agent(prompt: String, provider: ProviderConfig) -> AgentHandle {
    let (events_tx, events_rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let task_cancel = cancel.clone();
    tokio::spawn(async move {
        if let Err(e) = agent_loop(prompt, provider, events_tx.clone(), task_cancel).await {
            let _ = events_tx.send(AgentEvent::Error(e.to_string())).await;
        }
    });

    AgentHandle {
        events: events_rx,
        cancel,
    }
}

async fn agent_loop(
    prompt: String,
    provider: ProviderConfig,
    events: mpsc::Sender<AgentEvent>,
    cancel: CancellationToken,
) -> Result<(), ProviderError> {
    let client = OpenAiClient::new(
        provider.base_url,
        provider.api_key,
        provider.model,
        provider.max_tokens,
    )?;

    let mut history = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text { text: prompt }],
    }];

    let tools: Vec<Box<dyn Tool>> = vec![Box::new(FileTool {})];
    let tool_definitions = tools
        .iter()
        .map(|tool| tool.definition())
        .collect::<Vec<ToolDefinition>>();

    loop {
        let (assistant_msg, stop_reason) = client
            .stream_message(&history, &tool_definitions, &events, &cancel)
            .await?;

        tracing::debug!(history_len = history.len(), ?stop_reason);

        history.push(assistant_msg.clone());

        if stop_reason != StopReason::ToolUse {
            let _ = events.send(AgentEvent::TurnComplete { stop_reason }).await;
            return Ok(());
        }

        let mut results = Vec::new();

        for block in assistant_msg.content {
            match block {
                ContentBlock::ToolUse { id, input, name } => {
                    let _ = events
                        .send(AgentEvent::ToolStarted {
                            input: input.clone(),
                            name: name.clone(),
                        })
                        .await;

                    let (content, is_error) = match dispatch(&tools, &name, input.clone()).await {
                        Ok(output) => (output, false),
                        Err(err) => (err, true),
                    };

                    let _ = events
                        .send(AgentEvent::ToolFinished {
                            name: name.clone(),
                            output: content.clone(),
                        })
                        .await;

                    results.push(ContentBlock::ToolResult {
                        content,
                        is_error,
                        tool_use_id: id,
                    });
                }
                ContentBlock::Text { .. } => {}
                ContentBlock::ToolResult { .. } => {
                    tracing::warn!("unexpected tool result content block")
                }
            }
        }

        if results.is_empty() {
            return Err(ProviderError::Protocol {
                detail: "no tool results were generated".to_string(),
            });
        }

        history.push(Message {
            role: Role::User,
            content: results,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use std::time::Duration;
    use tempfile::NamedTempFile;
    use tokio::time::timeout;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn stream_chunk(delta: serde_json::Value, finish_reason: Option<&str>) -> serde_json::Value {
        json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1751980000,
            "model": "test-model",
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }]
        })
    }

    fn sse_response(chunks: &[serde_json::Value]) -> ResponseTemplate {
        let mut body = String::new();
        for chunk in chunks {
            body.push_str(&format!("data: {chunk}\n\n"));
        }
        body.push_str("data: [DONE]\n\n");
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(body)
    }

    /// An assistant turn that streams `text` and stops.
    fn text_turn(text: &str) -> ResponseTemplate {
        sse_response(&[
            stream_chunk(json!({ "role": "assistant", "content": text }), None),
            stream_chunk(json!({}), Some("stop")),
        ])
    }

    /// An assistant turn of tool calls, one per `(id, name, arguments)`.
    fn tool_call_turn(calls: &[(&str, &str, serde_json::Value)]) -> ResponseTemplate {
        let chunks: Vec<_> = calls
            .iter()
            .enumerate()
            .map(|(index, (id, name, args))| {
                stream_chunk(
                    json!({
                        "tool_calls": [{
                            "index": index,
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": args.to_string() }
                        }]
                    }),
                    None,
                )
            })
            .chain([stream_chunk(json!({}), Some("tool_calls"))])
            .collect();
        sse_response(&chunks)
    }

    /// Script the conversation: the nth request receives the nth response.
    /// Each turn must be consumed exactly once or the mock server panics when
    /// dropped. A run that loops forever dies on `expect(1)`, not silently.
    async fn mount_turns(server: &MockServer, turns: Vec<ResponseTemplate>) {
        for (i, turn) in turns.into_iter().enumerate() {
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(turn)
                .up_to_n_times(1)
                .with_priority((i + 1) as u8)
                .expect(1)
                .mount(server)
                .await;
        }
    }

    fn test_provider(server: &MockServer) -> ProviderConfig {
        ProviderConfig {
            base_url: server.uri(),
            api_key: "test-key".to_string(),
            model: "test-model".to_string(),
            max_tokens: 1234,
        }
    }

    /// Spawn the agent and collect every event until the channel closes.
    /// The timeout turns a non-terminating loop into a test failure
    /// instead of a hung suite.
    async fn run_agent(prompt: &str, server: &MockServer) -> Vec<AgentEvent> {
        let mut handle = spawn_agent(prompt.to_string(), test_provider(server));
        let mut events = Vec::new();
        timeout(Duration::from_secs(5), async {
            while let Some(event) = handle.events.recv().await {
                events.push(event);
            }
        })
        .await
        .expect("event channel never closed; is the agent loop terminating?");
        events
    }

    fn temp_file_with(contents: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(contents).unwrap();
        file
    }

    async fn nth_request_messages(server: &MockServer, n: usize) -> serde_json::Value {
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = requests[n].body_json().unwrap();
        body["messages"].clone()
    }

    #[tokio::test]
    async fn a_text_only_turn_streams_text_and_completes_cleanly() {
        // Arrange
        let server = MockServer::start().await;
        mount_turns(&server, vec![text_turn("Hello world")]).await;

        // Act
        let events = run_agent("Say hi", &server).await;

        // Assert
        assert_eq!(
            events,
            vec![
                AgentEvent::TextDelta("Hello world".to_string()),
                AgentEvent::TurnComplete {
                    stop_reason: StopReason::EndTurn
                },
            ]
        );
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value = requests[0].body_json().unwrap();
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(
            body["messages"],
            json!([{ "role": "user", "content": "Say hi" }])
        );
    }

    #[tokio::test]
    async fn a_tool_turn_executes_the_tool_and_round_trips_the_result() {
        // Success criteria 2 & 3: the model calls read_file, the harness
        // executes it, and the follow-up request carries the assistant echo
        // (tool_calls intact) plus a role:"tool" result with the file content.

        // Arrange
        let file = temp_file_with(b"[workspace]\nmembers = [\"crates/core\"]");
        let file_path = file.path().to_str().unwrap();
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                tool_call_turn(&[("call_abc", "read_file", json!({ "path": file_path }))]),
                text_turn("It has one member."),
            ],
        )
        .await;

        // Act
        let events = run_agent("What's in Cargo.toml?", &server).await;

        // Assert
        assert_eq!(
            events,
            vec![
                AgentEvent::ToolStarted {
                    name: "read_file".to_string(),
                    input: json!({ "path": file_path }),
                },
                AgentEvent::ToolFinished {
                    name: "read_file".to_string(),
                    output: "[workspace]\nmembers = [\"crates/core\"]".to_string(),
                },
                AgentEvent::TextDelta("It has one member.".to_string()),
                AgentEvent::TurnComplete {
                    stop_reason: StopReason::EndTurn
                },
            ]
        );

        // Assert
        let messages = nth_request_messages(&server, 1).await;
        assert_eq!(
            messages[1],
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json!({ "path": file_path }).to_string()
                    }
                }]
            }),
            "assistant echo must keep its tool_calls intact"
        );
        assert_eq!(
            messages[2],
            json!({
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "[workspace]\nmembers = [\"crates/core\"]"
            })
        );
    }

    #[tokio::test]
    async fn a_tool_error_is_fed_back_to_the_model_not_raised() {
        // Tool errors are model feedback, not failures. A missing file
        // must become an error tool result the model can see; the turn
        // continues and no Error event is emitted.

        // Arrange
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                tool_call_turn(&[(
                    "call_abc",
                    "read_file",
                    json!({ "path": "/definitely/not/a/real/file" }),
                )]),
                text_turn("That file doesn't exist."),
            ],
        )
        .await;

        // Act
        let events = run_agent("What's in nope.txt?", &server).await;

        // Assert
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::Error(_))),
            "a tool error must not surface as an agent Error: {events:?}"
        );
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnComplete {
                stop_reason: StopReason::EndTurn
            })
        ));

        let messages = nth_request_messages(&server, 1).await;
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_abc");
        let content = messages[2]["content"].as_str().unwrap();
        assert!(
            content.starts_with("Error:"),
            "error results state the error in content on the OpenAI wire: {content}"
        );
        assert!(
            !content.starts_with("Error: Error:"),
            "the error prefix must be applied exactly once: {content}"
        );
    }

    #[tokio::test]
    async fn a_hallucinated_tool_name_gets_an_error_result() {
        // An unknown tool name is an error tool result on the same plane
        // as a failed execution. The model is told and the harness lives.

        // Arrange
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                tool_call_turn(&[(
                    "call_abc",
                    "write_file",
                    json!({ "path": "x.txt", "content": "y" }),
                )]),
                text_turn("I can't write files."),
            ],
        )
        .await;

        // Act
        let events = run_agent("Write y to x.txt", &server).await;

        // Assert
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnComplete {
                stop_reason: StopReason::EndTurn
            })
        ));
        let messages = nth_request_messages(&server, 1).await;
        assert_eq!(
            messages[2],
            json!({
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "Error: unknown tool: write_file"
            })
        );
    }

    #[tokio::test]
    async fn every_tool_call_gets_a_result_including_failures() {
        // Arrange
        let file = temp_file_with(b"alpha");
        let file_path = file.path().to_str().unwrap();
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                tool_call_turn(&[
                    ("call_a", "read_file", json!({ "path": file_path })),
                    (
                        "call_b",
                        "read_file",
                        json!({ "path": "/definitely/not/a/real/file" }),
                    ),
                ]),
                text_turn("done"),
            ],
        )
        .await;

        // Act
        let events = run_agent("Read both files", &server).await;

        // Assert
        let names: Vec<_> = events
            .iter()
            .map(|event| match event {
                AgentEvent::ToolStarted { .. } => "started",
                AgentEvent::ToolFinished { .. } => "finished",
                AgentEvent::TextDelta(_) => "text",
                AgentEvent::TurnComplete { .. } => "complete",
                AgentEvent::Error(_) => "error",
            })
            .collect();
        assert_eq!(
            names,
            vec![
                "started", "finished", "started", "finished", "text", "complete"
            ]
        );

        // Assert
        let messages = nth_request_messages(&server, 1).await;
        assert_eq!(messages[2]["tool_call_id"], "call_a");
        assert_eq!(messages[2]["content"], "alpha");
        assert_eq!(messages[3]["tool_call_id"], "call_b");
        assert!(
            messages[3]["content"]
                .as_str()
                .unwrap()
                .starts_with("Error:")
        );
    }

    #[tokio::test]
    async fn a_provider_error_becomes_an_error_event() {
        // Provider errors are the *other* plane, they abort the turn and
        // surface as a single Error event, then the channel closes.

        // Arrange
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![ResponseTemplate::new(401).set_body_string("bad key")],
        )
        .await;

        // Act
        let events = run_agent("Say hi", &server).await;

        // Assert
        assert!(
            matches!(&events[..], [AgentEvent::Error(msg)] if msg.contains("401")),
            "expected a single Error event carrying the status, got {events:?}"
        );
    }

    #[tokio::test]
    async fn cancelling_aborts_the_turn_promptly() {
        // Cancellation is non-negotiable. Tripping the token must abort
        // the in-flight stream without waiting for the server.

        // Arrange
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(text_turn("too late").set_delay(Duration::from_secs(30)))
            .mount(&server)
            .await;
        let mut handle = spawn_agent("Say hi".to_string(), test_provider(&server));

        // Act
        handle.cancel.cancel();
        let mut events = Vec::new();
        timeout(Duration::from_secs(2), async {
            while let Some(event) = handle.events.recv().await {
                events.push(event);
            }
        })
        .await
        .expect("cancel did not abort the in-flight turn promptly");

        // Assert
        assert!(
            matches!(&events[..], [AgentEvent::Error(msg)] if msg.contains("cancel")),
            "expected a single cancellation Error event, got {events:?}"
        );
    }
}
