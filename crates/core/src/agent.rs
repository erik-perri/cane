use crate::message::{AgentEvent, ContentBlock, Message, Role, StopReason, TurnOutcome};
use crate::provider::{OpenAiClient, ProviderConfig, ProviderError};
use crate::{FileTool, Tool, ToolDefinition, Workspace, dispatch};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, PartialEq)]
pub enum AgentCommand {
    UserInput(String),
}

pub struct AgentHandle {
    pub cancel: CancellationToken,
    pub commands: mpsc::Sender<AgentCommand>,
    pub events: mpsc::Receiver<AgentEvent>,
}

pub fn spawn_agent(provider: ProviderConfig, workspace: Workspace) -> AgentHandle {
    let (events_tx, events_rx) = mpsc::channel(64);
    let (commands_tx, commands_rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();

    let task_cancel = cancel.clone();
    tokio::spawn(async move {
        if let Err(e) = agent_loop(
            provider,
            workspace,
            events_tx.clone(),
            task_cancel,
            commands_rx,
        )
        .await
        {
            let _ = events_tx.send(AgentEvent::Error(e.to_string())).await;
        }
    });

    AgentHandle {
        cancel,
        commands: commands_tx,
        events: events_rx,
    }
}

async fn agent_loop(
    provider: ProviderConfig,
    workspace: Workspace,
    events: mpsc::Sender<AgentEvent>,
    cancel: CancellationToken,
    mut commands: mpsc::Receiver<AgentCommand>,
) -> Result<(), ProviderError> {
    let workspace = Arc::new(workspace);
    let client = OpenAiClient::new(
        provider.base_url,
        provider.api_key,
        provider.model,
        provider.max_tokens,
    )?;

    let mut history = Vec::new();

    let tools: Vec<Box<dyn Tool>> = vec![Box::new(FileTool::new(Arc::clone(&workspace)))];
    let tool_definitions = tools
        .iter()
        .map(|tool| tool.definition())
        .collect::<Vec<ToolDefinition>>();

    loop {
        let command = tokio::select! {
            _ = cancel.cancelled() => {
                return Err(ProviderError::Cancelled);
            }
            _ = events.closed() => {
                return Ok(());
            }
            command = commands.recv() => {
                match command {
                    Some(command) => command,
                    None => return Ok(()),
                }
            }
        };

        let turn_start = history.len();

        match command {
            AgentCommand::UserInput(prompt) => {
                history.push(Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text { text: prompt }],
                });
            }
        }

        loop {
            let stream_result = tokio::select! {
                _ = events.closed() => return Ok(()),
                result = client.stream_message(&history, &tool_definitions, &events, &cancel) => {
                    result
                }
            };

            let (assistant_msg, stop_reason) = match stream_result {
                Ok(result) => result,
                Err(error) => {
                    let cancelled = matches!(&error, ProviderError::Cancelled);

                    // History is committed only when the turn completes. A
                    // failed stream may have already appended messages, and
                    // retaining them would leave an incomplete turn in the
                    // next request.
                    history.truncate(turn_start);

                    let _ = events.send(AgentEvent::Error(error.to_string())).await;
                    let _ = events
                        .send(AgentEvent::TurnComplete {
                            outcome: if cancelled {
                                TurnOutcome::Cancelled
                            } else {
                                TurnOutcome::Failed
                            },
                        })
                        .await;

                    // A cancellation token cannot be reset, so this session
                    // cannot accept another turn after its active turn ends.
                    // End cleanly after delivering the terminal event rather
                    // than returning an error to the task wrapper, which only
                    // emits Error and would leave frontends waiting.
                    if cancelled {
                        return Ok(());
                    }
                    break;
                }
            };

            tracing::debug!(history_len = history.len(), ?stop_reason);

            history.push(assistant_msg);

            if stop_reason != StopReason::ToolUse {
                let _ = events
                    .send(AgentEvent::TurnComplete {
                        outcome: TurnOutcome::Completed { stop_reason },
                    })
                    .await;
                break;
            }

            let mut results = Vec::new();

            for block in &history.last().expect("just pushed").content {
                match block {
                    ContentBlock::ToolUse {
                        id, input, name, ..
                    } => {
                        if events
                            .send(AgentEvent::ToolStarted {
                                input: input.clone(),
                                name: name.clone(),
                            })
                            .await
                            .is_err()
                        {
                            return Ok(());
                        }

                        let (content, is_error) = match dispatch(&tools, name, input.clone()).await
                        {
                            Ok(output) => (output, false),
                            Err(err) => (err, true),
                        };

                        if events
                            .send(AgentEvent::ToolFinished {
                                name: name.clone(),
                                output: content.clone(),
                                is_error,
                            })
                            .await
                            .is_err()
                        {
                            return Ok(());
                        }

                        results.push(ContentBlock::ToolResult {
                            content,
                            is_error,
                            tool_use_id: id.clone(),
                        });
                    }
                    ContentBlock::Text { .. } => {}
                    ContentBlock::ToolResult { .. } => {
                        tracing::warn!("unexpected tool result content block")
                    }
                }
            }

            if results.is_empty() {
                history.truncate(turn_start);

                let _ = events
                    .send(AgentEvent::Error(
                        "no tool results were generated".to_string(),
                    ))
                    .await;
                let _ = events
                    .send(AgentEvent::TurnComplete {
                        outcome: TurnOutcome::Failed,
                    })
                    .await;
                break;
            }

            history.push(Message {
                role: Role::User,
                content: results,
            });
        }
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

    fn test_workspace() -> Workspace {
        Workspace::new(std::env::temp_dir()).unwrap()
    }

    /// Send one input, close the command channel, and collect every event
    /// until the agent exits. This keeps the old one-shot tests focused on a
    /// single turn while exercising the session-shaped public API.
    async fn run_agent(prompt: &str, server: &MockServer) -> Vec<AgentEvent> {
        let mut handle = spawn_agent(test_provider(server), test_workspace());
        handle
            .commands
            .send(AgentCommand::UserInput(prompt.to_string()))
            .await
            .expect("agent command channel closed before accepting input");
        drop(handle.commands);

        collect_until_events_close(&mut handle.events).await
    }

    /// The timeout turns a non-terminating loop into a test failure instead
    /// of a hung suite.
    async fn collect_until_events_close(
        events_rx: &mut mpsc::Receiver<AgentEvent>,
    ) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        timeout(Duration::from_secs(5), async {
            while let Some(event) = events_rx.recv().await {
                events.push(event);
            }
        })
        .await
        .expect("event channel never closed; is the agent loop terminating?");
        events
    }

    /// Collect precisely one completed turn, leaving the session alive for a
    /// follow-up command.
    async fn collect_turn(events_rx: &mut mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
        timeout(Duration::from_secs(5), async {
            let mut events = Vec::new();
            loop {
                let event = events_rx
                    .recv()
                    .await
                    .expect("event channel closed before TurnComplete");
                let complete = matches!(event, AgentEvent::TurnComplete { .. });
                events.push(event);
                if complete {
                    return events;
                }
            }
        })
        .await
        .expect("agent turn never completed")
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
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn
                    }
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
    async fn agent_waits_for_input_and_exits_when_command_channel_closes() {
        // A session must not contact the provider until its frontend supplies
        // input. Closing that frontend command channel is clean shutdown.

        // Arrange
        let server = MockServer::start().await;
        let mut handle = spawn_agent(test_provider(&server), test_workspace());

        // Act
        drop(handle.commands);
        let events = collect_until_events_close(&mut handle.events).await;

        // Assert
        assert!(
            events.is_empty(),
            "idle session emitted unexpected events: {events:?}"
        );
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "idle session made a provider request"
        );
    }

    #[tokio::test]
    async fn dropping_events_stops_agent_even_when_command_sender_remains_alive() {
        // Arrange
        let server = MockServer::start().await;
        let handle = spawn_agent(test_provider(&server), test_workspace());
        let AgentHandle {
            cancel: _cancel,
            commands,
            events,
        } = handle;

        // Act
        drop(events);

        // Assert
        timeout(Duration::from_secs(5), commands.closed())
            .await
            .expect("agent remained alive after its event receiver was dropped");
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "idle session made a provider request"
        );
    }

    #[tokio::test]
    async fn two_user_inputs_produce_two_completed_turns_and_preserve_history() {
        // Arrange
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![text_turn("Hello!"), text_turn("Yes, I remember.")],
        )
        .await;
        let mut handle = spawn_agent(test_provider(&server), test_workspace());

        // Act
        handle
            .commands
            .send(AgentCommand::UserInput("My name is Ada.".to_string()))
            .await
            .unwrap();
        let first_turn = collect_turn(&mut handle.events).await;

        handle
            .commands
            .send(AgentCommand::UserInput(
                "Do you remember my name?".to_string(),
            ))
            .await
            .unwrap();
        let second_turn = collect_turn(&mut handle.events).await;
        drop(handle.commands);
        let shutdown_events = collect_until_events_close(&mut handle.events).await;

        // Assert
        assert_eq!(
            first_turn,
            vec![
                AgentEvent::TextDelta("Hello!".to_string()),
                AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn
                    }
                },
            ]
        );
        assert_eq!(
            second_turn,
            vec![
                AgentEvent::TextDelta("Yes, I remember.".to_string()),
                AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn
                    }
                },
            ]
        );
        assert!(
            shutdown_events.is_empty(),
            "clean shutdown emitted unexpected events: {shutdown_events:?}"
        );
        assert_eq!(
            nth_request_messages(&server, 1).await,
            json!([
                { "role": "user", "content": "My name is Ada." },
                { "role": "assistant", "content": "Hello!" },
                { "role": "user", "content": "Do you remember my name?" },
            ])
        );
    }

    #[tokio::test]
    async fn a_completed_tool_turn_is_preserved_for_the_next_user_input() {
        // Tool-use history needs to outlive its turn too: the assistant echo
        // and tool result must remain paired when the next turn is sent.

        // Arrange
        let file = temp_file_with(b"alpha");
        let file_path = file.path().to_str().unwrap();
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                tool_call_turn(&[("call_abc", "read_file", json!({ "path": file_path }))]),
                text_turn("The file says alpha."),
                text_turn("Still alpha."),
            ],
        )
        .await;
        let mut handle = spawn_agent(test_provider(&server), test_workspace());

        // Act
        handle
            .commands
            .send(AgentCommand::UserInput("Read the file.".to_string()))
            .await
            .unwrap();
        let first_turn = collect_turn(&mut handle.events).await;

        handle
            .commands
            .send(AgentCommand::UserInput("What did it say?".to_string()))
            .await
            .unwrap();
        let second_turn = collect_turn(&mut handle.events).await;
        drop(handle.commands);
        let shutdown_events = collect_until_events_close(&mut handle.events).await;

        // Assert
        assert!(matches!(
            first_turn.last(),
            Some(AgentEvent::TurnComplete {
                outcome: TurnOutcome::Completed {
                    stop_reason: StopReason::EndTurn
                }
            })
        ));
        assert!(matches!(
            second_turn.as_slice(),
            [
                AgentEvent::TextDelta(text),
                AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn
                    }
                },
            ] if text == "Still alpha."
        ));
        assert!(
            shutdown_events.is_empty(),
            "clean shutdown emitted unexpected events: {shutdown_events:?}"
        );
        assert_eq!(
            nth_request_messages(&server, 2).await,
            json!([
                { "role": "user", "content": "Read the file." },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json!({ "path": file_path }).to_string(),
                        },
                    }],
                },
                { "role": "tool", "tool_call_id": "call_abc", "content": "alpha" },
                { "role": "assistant", "content": "The file says alpha." },
                { "role": "user", "content": "What did it say?" },
            ])
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
                    is_error: false,
                },
                AgentEvent::TextDelta("It has one member.".to_string()),
                AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn
                    }
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
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolFinished { is_error: true, .. }))
        );
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnComplete {
                outcome: TurnOutcome::Completed {
                    stop_reason: StopReason::EndTurn
                }
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
                outcome: TurnOutcome::Completed {
                    stop_reason: StopReason::EndTurn
                }
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
        // Provider errors are the *other* plane: they abort the current turn
        // and surface as an Error event. This helper closes the command
        // channel after that one input, so the session then exits cleanly.

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
            matches!(
                &events[..],
                [
                    AgentEvent::Error(msg),
                    AgentEvent::TurnComplete {
                        outcome: TurnOutcome::Failed
                    }
                ] if msg.contains("401")
            ),
            "expected an Error followed by a failed TurnComplete, got {events:?}"
        );
    }

    #[tokio::test]
    async fn provider_error_does_not_prevent_a_later_user_input() {
        // Provider errors end the affected turn, not the long-lived session.

        // Arrange
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                ResponseTemplate::new(401).set_body_string("bad key"),
                text_turn("This turn succeeded."),
            ],
        )
        .await;
        let mut handle = spawn_agent(test_provider(&server), test_workspace());

        // Act
        handle
            .commands
            .send(AgentCommand::UserInput("This will fail.".to_string()))
            .await
            .unwrap();
        let failed_turn = collect_turn(&mut handle.events).await;

        handle
            .commands
            .send(AgentCommand::UserInput("Try again.".to_string()))
            .await
            .expect("session did not accept input after provider error");
        let recovery_turn = collect_turn(&mut handle.events).await;
        drop(handle.commands);
        let shutdown_events = collect_until_events_close(&mut handle.events).await;

        // Assert
        assert!(
            matches!(
                &failed_turn[..],
                [
                    AgentEvent::Error(msg),
                    AgentEvent::TurnComplete {
                        outcome: TurnOutcome::Failed
                    }
                ] if msg.contains("401")
            ),
            "expected an Error followed by a failed TurnComplete, got {failed_turn:?}"
        );
        assert_eq!(
            recovery_turn,
            vec![
                AgentEvent::TextDelta("This turn succeeded.".to_string()),
                AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn
                    }
                },
            ]
        );
        assert!(
            shutdown_events.is_empty(),
            "clean shutdown emitted unexpected events: {shutdown_events:?}"
        );
        assert_eq!(
            nth_request_messages(&server, 1).await,
            json!([{ "role": "user", "content": "Try again." }])
        );
    }

    #[tokio::test]
    async fn cancelling_aborts_the_turn_promptly() {
        // Cancellation is non-negotiable. Tripping the token must abort
        // the in-flight stream without waiting for the server.

        // Arrange
        let server = MockServer::start().await;
        let request = Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(text_turn("too late").set_delay(Duration::from_secs(30)))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let mut handle = spawn_agent(test_provider(&server), test_workspace());

        // Act
        handle
            .commands
            .send(AgentCommand::UserInput("Say hi".to_string()))
            .await
            .unwrap();
        timeout(Duration::from_secs(2), request.wait_until_satisfied())
            .await
            .expect("agent did not start its provider request promptly");
        handle.cancel.cancel();
        let events = collect_until_events_close(&mut handle.events).await;

        // Assert
        assert!(
            matches!(
                &events[..],
                [
                    AgentEvent::Error(msg),
                    AgentEvent::TurnComplete {
                        outcome: TurnOutcome::Cancelled
                    }
                ] if msg.contains("cancel")
            ),
            "expected an Error followed by a cancelled TurnComplete, got {events:?}"
        );
    }
}
