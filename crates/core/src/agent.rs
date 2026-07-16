use crate::Workspace;
use crate::approval::{ApprovalAuthorization, ApprovalGate};
use crate::message::{ContentBlock, Message, Role, StopReason, ToolResultData};
use crate::protocol::{AgentCommand, AgentEvent, AgentExit, EventSink, HostHandle, TurnOutcome};
use crate::provider::{OpenAiClient, ProviderConfig, ProviderError};
use crate::tools::{PreparedInvocation, ToolExecutionError, ToolSet};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct AgentSession {
    client: OpenAiClient,
    host_handle: HostHandle,
    tool_set: ToolSet,
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
        let events = EventSink::new(events_tx.clone());

        let host_handle = HostHandle {
            cancel: task_cancel,
            commands: commands_rx,
            events: events.clone(),
        };

        let session = match AgentSession::new(host_handle, provider, workspace) {
            Ok(s) => s,
            Err(e) => {
                let _ = events.emit(AgentEvent::Error(e.to_string())).await;
                return;
            }
        };

        match session.run().await {
            Ok(()) | Err(AgentExit::Disconnected) => {
                // Clean shutdown: nothing to say, no one to say it to.
            }
            Err(AgentExit::Cancelled) => {
                // Already surfaced as Error + TurnComplete inside the loop.
            }
        }
    });

    AgentHandle {
        cancel,
        commands: commands_tx,
        events: events_rx,
    }
}

impl AgentSession {
    fn new(
        host_handle: HostHandle,
        provider: ProviderConfig,
        workspace: Workspace,
    ) -> Result<AgentSession, ProviderError> {
        let workspace = Arc::new(workspace);
        let client = OpenAiClient::new(
            provider.base_url,
            provider.api_key,
            provider.model,
            provider.max_tokens,
        )?;

        Ok(AgentSession {
            host_handle,
            client,
            tool_set: ToolSet::new(workspace),
        })
    }

    async fn run(mut self) -> Result<(), AgentExit> {
        let mut history = Vec::new();
        let mut approval_gate = ApprovalGate::new();

        loop {
            let command = tokio::select! {
                _ = self.host_handle.cancel.cancelled() => return Err(AgentExit::Cancelled),
                _ = self.host_handle.events.closed() => return Ok(()),
                command = self.host_handle.commands.recv() => {
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

            match self.run_turn(&mut history, &mut approval_gate).await {
                Ok(outcome) => {
                    let session_over = matches!(outcome, TurnOutcome::Cancelled);
                    let needs_rollback = session_over || matches!(outcome, TurnOutcome::Failed);

                    if needs_rollback {
                        // Truncate the history on failure so we don't leave an incomplete
                        // turn in the next request.
                        history.truncate(turn_start);
                    }

                    self.host_handle
                        .events
                        .emit(AgentEvent::TurnComplete { outcome })
                        .await?;

                    if session_over {
                        return Err(AgentExit::Cancelled);
                    }
                }
                Err(AgentExit::Cancelled) => {
                    // If a cancel is tripped mid-approval, the turn still gets
                    // its one marker before the session ends.
                    let _ = self
                        .host_handle
                        .events
                        .emit(AgentEvent::TurnComplete {
                            outcome: TurnOutcome::Cancelled,
                        })
                        .await;

                    return Err(AgentExit::Cancelled);
                }
                Err(exit) => return Err(exit),
            }
        }
    }

    async fn run_turn(
        &mut self,
        history: &mut Vec<Message>,
        gate: &mut ApprovalGate,
    ) -> Result<TurnOutcome, AgentExit> {
        loop {
            let stream_result = tokio::select! {
                _ = self.host_handle.events.closed() => return Err(AgentExit::Disconnected),
                result = self.client.stream_message(history, self.tool_set.definitions(), self.host_handle.events.sender(), &self.host_handle.cancel) => {
                    result
                }
            };

            let (assistant_msg, stop_reason) = match stream_result {
                Ok(result) => result,
                Err(error) => {
                    let cancelled = matches!(&error, ProviderError::Cancelled);

                    self.host_handle
                        .events
                        .emit(AgentEvent::Error(error.to_string()))
                        .await?;

                    return if cancelled {
                        Ok(TurnOutcome::Cancelled)
                    } else {
                        Ok(TurnOutcome::Failed)
                    };
                }
            };

            tracing::debug!(history_len = history.len(), ?stop_reason);

            history.push(assistant_msg);

            if stop_reason != StopReason::ToolUse {
                return Ok(TurnOutcome::Completed { stop_reason });
            }

            let mut results = Vec::new();

            for block in &history.last().expect("just pushed").content {
                match block {
                    ContentBlock::ToolUse {
                        id, input, name, ..
                    } => {
                        let tool_result = self.execute_tool_call(id, name, input, gate).await?;

                        results.push(ContentBlock::ToolResult(tool_result));
                    }
                    ContentBlock::Text { .. } => {
                        //
                    }
                    ContentBlock::ToolResult { .. } => {
                        tracing::warn!("unexpected tool result content block")
                    }
                }
            }

            if results.is_empty() {
                self.host_handle
                    .events
                    .emit(AgentEvent::Error(
                        "no tool results were generated".to_string(),
                    ))
                    .await?;

                return Ok(TurnOutcome::Failed);
            }

            history.push(Message {
                role: Role::User,
                content: results,
            });
        }
    }

    async fn execute_tool_call(
        &mut self,
        id: &str,
        name: &str,
        input: &Value,
        gate: &mut ApprovalGate,
    ) -> Result<ToolResultData, AgentExit> {
        let invocation = match prepare_tool_call(&self.tool_set, id, name, input).await {
            Ok(invocation) => invocation,
            Err(result) => {
                self.host_handle
                    .events
                    .emit(AgentEvent::ToolRejected {
                        name: name.to_string(),
                        error: result.content.clone(),
                    })
                    .await?;
                return Ok(result);
            }
        };

        match gate
            .authorize(
                invocation.approval_requirement(),
                name,
                id,
                input,
                &self.host_handle.events,
                &self.host_handle.cancel,
            )
            .await?
        {
            ApprovalAuthorization::Denied { reason } => {
                self.host_handle
                    .events
                    .emit(AgentEvent::ToolDenied {
                        name: name.to_string(),
                        reason: reason.to_string(),
                    })
                    .await?;

                Ok(ToolResultData {
                    content: format!(
                        "The user declined this tool call and said: \"{reason}\". Do not assume the tool ran. Address their feedback, then retry if appropriate."
                    ),
                    is_error: false,
                    tool_use_id: id.to_string(),
                })
            }

            ApprovalAuthorization::Approved => {
                execute_invocation(&self.host_handle, id, name, input, invocation).await
            }
        }
    }
}

async fn prepare_tool_call(
    tool_set: &ToolSet,
    id: &str,
    name: &str,
    input: &Value,
) -> Result<Box<dyn PreparedInvocation>, ToolResultData> {
    let tool = tool_set
        .locate(name)
        .map_err(|error| failed_tool_result(id, error))?;
    tool.prepare(input.clone())
        .await
        .map_err(|error| failed_tool_result(id, error))
}

fn failed_tool_result(id: &str, error: String) -> ToolResultData {
    ToolResultData {
        content: error,
        is_error: true,
        tool_use_id: id.to_string(),
    }
}

async fn execute_invocation(
    host_handle: &HostHandle,
    id: &str,
    name: &str,
    input: &Value,
    invocation: Box<dyn PreparedInvocation>,
) -> Result<ToolResultData, AgentExit> {
    host_handle
        .events
        .emit(AgentEvent::ToolStarted {
            input: input.clone(),
            name: name.to_string(),
        })
        .await?;

    let execution_cancel = host_handle.cancel.child_token();
    let tool_future = invocation.execute(execution_cancel.clone());

    let execution_result = tokio::select! {
        _ = host_handle.events.closed() => {
            execution_cancel.cancel();
            return Err(AgentExit::Disconnected);
        }
        _ = host_handle.cancel.cancelled() => {
            execution_cancel.cancel();
            return Err(AgentExit::Cancelled);
        }
        result = tool_future => result,
    };

    let (content, is_error) = match execution_result {
        Ok(content) => (content, false),
        Err(ToolExecutionError::ToolError(error)) => (error, true),
        Err(ToolExecutionError::Cancelled) => {
            return Err(AgentExit::Cancelled);
        }
    };

    host_handle
        .events
        .emit(AgentEvent::ToolFinished {
            is_error,
            name: name.to_string(),
            output: content.clone(),
        })
        .await?;

    Ok(ToolResultData {
        content,
        is_error,
        tool_use_id: id.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApprovalDecision;
    use crate::protocol::ApprovalRequirement;
    use crate::tools::{Tool, ToolDefinition};
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::io::Write;
    use std::time::Duration;
    use tempfile::NamedTempFile;
    use tokio::sync::Notify;
    use tokio::time::timeout;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn stream_chunk(delta: Value, finish_reason: Option<&str>) -> Value {
        json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1751980000,
            "model": "test-model",
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }]
        })
    }

    fn sse_response(chunks: &[Value]) -> ResponseTemplate {
        let mut body = String::new();
        for chunk in chunks {
            body.push_str(&format!("data: {chunk}\n\n"));
        }
        body.push_str("data: [DONE]\n\n");
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(body)
    }

    fn text_turn(text: &str) -> ResponseTemplate {
        sse_response(&[
            stream_chunk(json!({ "role": "assistant", "content": text }), None),
            stream_chunk(json!({}), Some("stop")),
        ])
    }

    fn tool_call_turn(calls: &[(&str, &str, Value)]) -> ResponseTemplate {
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

    fn test_session(
        server: &MockServer,
        host_handle: HostHandle,
        tools: Vec<Box<dyn Tool>>,
    ) -> AgentSession {
        let provider = test_provider(server);

        let client = OpenAiClient::new(
            provider.base_url,
            provider.api_key,
            provider.model,
            provider.max_tokens,
        )
        .unwrap();

        AgentSession {
            host_handle,
            client,
            tool_set: ToolSet::from_tools(tools),
        }
    }

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

    async fn nth_request_messages(server: &MockServer, n: usize) -> Value {
        let requests = server.received_requests().await.unwrap();
        let body: Value = requests[n].body_json().unwrap();
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
        assert_eq!(2, events.len());

        let Some(AgentEvent::TextDelta(text)) = events.first() else {
            panic!("expected the first event to be TextDelta");
        };
        assert_eq!(text, "Hello world");

        let Some(AgentEvent::TurnComplete { outcome }) = events.get(1) else {
            panic!("expected the second event to be TurnComplete");
        };
        assert_eq!(
            outcome,
            &TurnOutcome::Completed {
                stop_reason: StopReason::EndTurn
            }
        );

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: Value = requests[0].body_json().unwrap();
        assert_eq!(
            body["messages"],
            json!([{ "role": "user", "content": "Say hi" }])
        );
    }

    #[tokio::test]
    async fn requests_advertise_all_registered_tools() {
        // Arrange
        let server = MockServer::start().await;
        mount_turns(&server, vec![text_turn("Hello world")]).await;

        // Act
        run_agent("Say hi", &server).await;

        // Assert
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: Value = requests[0].body_json().unwrap();
        let mut names = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        names.sort_unstable();

        assert_eq!(names, vec!["edit_file", "glob", "read_file", "write_file"]);
    }

    #[tokio::test]
    async fn agent_waits_for_input_and_exits_when_command_channel_closes() {
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
        assert_eq!(2, first_turn.len());

        let Some(AgentEvent::TextDelta(text)) = first_turn.first() else {
            panic!("Expected first turn to contain a TextDelta event");
        };
        assert_eq!("Hello!", text);

        let Some(AgentEvent::TurnComplete { outcome }) = first_turn.get(1) else {
            panic!("Expected first turn to contain a TurnComplete event");
        };
        assert_eq!(
            outcome,
            &TurnOutcome::Completed {
                stop_reason: StopReason::EndTurn
            }
        );

        assert_eq!(2, second_turn.len());

        let Some(AgentEvent::TextDelta(text)) = second_turn.first() else {
            panic!("Expected second turn to contain a TextDelta event");
        };
        assert_eq!("Yes, I remember.", text);

        let Some(AgentEvent::TurnComplete { outcome }) = second_turn.get(1) else {
            panic!("Expected second turn to contain a TurnComplete event");
        };
        assert_eq!(
            outcome,
            &TurnOutcome::Completed {
                stop_reason: StopReason::EndTurn
            }
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
    async fn user_input_queued_while_approval_is_pending_becomes_the_next_turn() {
        // Arrange
        let file = temp_file_with(b"alpha");
        let file_path = file.path().to_str().unwrap();

        let server = MockServer::start().await;

        mount_turns(
            &server,
            vec![
                tool_call_turn(&[(
                    "write-1",
                    "write_file",
                    json!({ "path": file_path, "content": "what" }),
                )]),
                text_turn("what1"),
                text_turn("what2"),
            ],
        )
        .await;

        let mut handle = spawn_agent(test_provider(&server), test_workspace());

        // Act
        handle
            .commands
            .send(AgentCommand::UserInput("what".to_string()))
            .await
            .unwrap();

        let respond_to = loop {
            let event = handle.events.recv().await.unwrap();

            if let AgentEvent::ApprovalRequest {
                respond_to: new_respond_to,
                ..
            } = event
            {
                break new_respond_to;
            }
        };

        handle
            .commands
            .send(AgentCommand::UserInput("what3".to_string()))
            .await
            .unwrap();

        respond_to.send(ApprovalDecision::Allow).unwrap();

        let first_turn = collect_turn(&mut handle.events).await;
        let second_turn = collect_turn(&mut handle.events).await;

        drop(handle.commands);
        let shutdown_events = collect_until_events_close(&mut handle.events).await;

        let queued_message = nth_request_messages(&server, 2).await;

        // Assert
        assert_eq!(4, first_turn.len());
        assert_eq!(2, second_turn.len());
        assert_eq!(0, shutdown_events.len());

        let Some(AgentEvent::ToolStarted { name, .. }) = first_turn.first() else {
            panic!("Expected first event to be a ToolStarted event");
        };
        assert_eq!("write_file", name);

        let Some(AgentEvent::ToolFinished { name, .. }) = first_turn.get(1) else {
            panic!("Expected second event to be a ToolFinished event");
        };
        assert_eq!("write_file", name);

        let Some(AgentEvent::TextDelta(text)) = first_turn.get(2) else {
            panic!("Expected third event to be a TextDelta event");
        };
        assert_eq!("what1", text);

        let Some(AgentEvent::TurnComplete { outcome }) = first_turn.get(3) else {
            panic!("Expected fourth event to be a TurnComplete event");
        };
        assert_eq!(
            TurnOutcome::Completed {
                stop_reason: StopReason::EndTurn
            },
            *outcome,
        );

        let Some(AgentEvent::TextDelta(text)) = second_turn.first() else {
            panic!("Expected first event to be a TextDelta event");
        };
        assert_eq!("what2", text);

        let Some(AgentEvent::TurnComplete { outcome }) = second_turn.get(1) else {
            panic!("Expected second event to be a TurnComplete event");
        };
        assert_eq!(
            TurnOutcome::Completed {
                stop_reason: StopReason::EndTurn
            },
            *outcome,
        );

        let Some(last_message) = queued_message
            .as_array()
            .and_then(|messages| messages.last())
        else {
            panic!("expected provider messages to be a non-empty array");
        };

        assert_eq!(
            last_message,
            &json!({
                "role": "user",
                "content": "what3"
            })
        );
    }

    #[tokio::test]
    async fn denied_tool_is_not_started_or_executed() {
        // Arrange
        let file = temp_file_with(b"original");
        let file_path = file.path().to_str().unwrap();
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                tool_call_turn(&[(
                    "write-1",
                    "write_file",
                    json!({ "path": file_path, "content": "changed" }),
                )]),
                text_turn("I did not change it."),
            ],
        )
        .await;
        let mut handle = spawn_agent(test_provider(&server), test_workspace());
        handle
            .commands
            .send(AgentCommand::UserInput("Change the file.".to_string()))
            .await
            .unwrap();

        // Act
        let respond_to = loop {
            let event = handle.events.recv().await.unwrap();
            match event {
                AgentEvent::ApprovalRequest { respond_to, .. } => break respond_to,
                AgentEvent::ToolStarted { .. } => {
                    panic!("tool was reported as started before approval")
                }
                _ => {}
            }
        };
        respond_to
            .send(ApprovalDecision::Deny {
                reason: "not this file".to_string(),
            })
            .unwrap();

        let events = collect_turn(&mut handle.events).await;

        // Assert
        assert!(matches!(
            events.as_slice(),
            [
                AgentEvent::ToolDenied { name, reason },
                AgentEvent::TextDelta(text),
                AgentEvent::TurnComplete { .. },
            ] if name == "write_file"
                && reason == "not this file"
                && text == "I did not change it."
        ));
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "original");
    }

    #[tokio::test]
    async fn a_tool_turn_executes_the_tool_and_round_trips_the_result() {
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
        assert_eq!(4, events.len());

        let Some(AgentEvent::ToolStarted { name, input }) = events.first() else {
            panic!("Expected first event to be a ToolStarted event");
        };
        assert_eq!("read_file", name);
        assert_eq!(json!({ "path": file_path }), *input);

        let Some(AgentEvent::ToolFinished {
            name,
            output,
            is_error,
        }) = events.get(1)
        else {
            panic!("Expected second event to be a ToolFinished event");
        };
        assert_eq!("read_file", name);
        assert!(!is_error);
        assert_eq!("[workspace]\nmembers = [\"crates/core\"]", output);

        let Some(AgentEvent::TextDelta(text)) = events.get(2) else {
            panic!("Expected third event to be a TextDelta event");
        };
        assert_eq!("It has one member.", text);

        let Some(AgentEvent::TurnComplete { outcome }) = events.get(3) else {
            panic!("Expected fourth event to be a TurnComplete event");
        };
        assert_eq!(
            outcome,
            &TurnOutcome::Completed {
                stop_reason: StopReason::EndTurn
            }
        );

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
                .any(|event| matches!(event, AgentEvent::ToolRejected { .. }))
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
    async fn an_unknown_tool_name_gets_an_error_result() {
        // Arrange
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                tool_call_turn(&[(
                    "call_abc",
                    "write_the_file_at_the_path",
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
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolRejected { name, .. }
                if name == "write_the_file_at_the_path"
        )));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolStarted { .. }))
        );
        let messages = nth_request_messages(&server, 1).await;
        assert_eq!(
            messages[2],
            json!({
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "Error: unknown tool: `write_the_file_at_the_path`"
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
                AgentEvent::ApprovalRequest { .. } => "approval",
                AgentEvent::ToolStarted { .. } => "started",
                AgentEvent::ToolFinished { .. } => "finished",
                AgentEvent::ToolDenied { .. } => "denied",
                AgentEvent::ToolRejected { .. } => "rejected",
                AgentEvent::TextDelta(_) => "text",
                AgentEvent::TurnComplete { .. } => "complete",
                AgentEvent::Error(_) => "error",
            })
            .collect();
        assert_eq!(
            names,
            vec!["started", "finished", "rejected", "text", "complete"]
        );

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

        assert_eq!(2, recovery_turn.len());

        let Some(AgentEvent::TextDelta(text)) = recovery_turn.first() else {
            panic!("expected the first event to be TextDelta");
        };
        assert_eq!("This turn succeeded.", text);

        let Some(AgentEvent::TurnComplete { outcome }) = recovery_turn.get(1) else {
            panic!("expected the second event to be TurnComplete");
        };
        assert_eq!(
            outcome,
            &TurnOutcome::Completed {
                stop_reason: StopReason::EndTurn
            }
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
    async fn an_allowed_mutating_tool_executes_and_round_trips_its_result() {
        // Arrange
        let file = temp_file_with(b"original");
        let file_path = file.path().to_str().unwrap();
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                tool_call_turn(&[(
                    "write-1",
                    "write_file",
                    json!({ "path": file_path, "content": "changed" }),
                )]),
                text_turn("I changed it."),
            ],
        )
        .await;
        let mut handle = spawn_agent(test_provider(&server), test_workspace());
        handle
            .commands
            .send(AgentCommand::UserInput("Change the file.".to_string()))
            .await
            .unwrap();

        // Act
        let respond_to = loop {
            let event = handle.events.recv().await.unwrap();
            match event {
                AgentEvent::ApprovalRequest { respond_to, .. } => break respond_to,
                AgentEvent::ToolStarted { .. } => {
                    panic!("tool was reported as started before approval")
                }
                _ => {}
            }
        };
        respond_to.send(ApprovalDecision::Allow).unwrap();

        let events = collect_turn(&mut handle.events).await;

        // Assert
        assert_eq!(4, events.len());

        let Some(AgentEvent::ToolStarted { name, input }) = events.first() else {
            panic!("Expected first event to be a ToolStarted event");
        };
        assert_eq!("write_file", name);
        assert_eq!(json!({ "content": "changed", "path": file_path }), *input);

        let Some(AgentEvent::ToolFinished {
            name,
            output,
            is_error,
        }) = events.get(1)
        else {
            panic!("Expected second event to be a ToolFinished event");
        };
        assert_eq!("write_file", name);
        assert!(!is_error);
        assert_eq!(format!("updated `{file_path}`; 7 bytes written"), *output);

        let Some(AgentEvent::TextDelta(text)) = events.get(2) else {
            panic!("Expected third event to be a TextDelta event");
        };
        assert_eq!("I changed it.", text);

        let Some(AgentEvent::TurnComplete { outcome }) = events.get(3) else {
            panic!("Expected fourth event to be a TurnComplete event");
        };
        assert_eq!(
            outcome,
            &TurnOutcome::Completed {
                stop_reason: StopReason::EndTurn
            }
        );

        let messages = nth_request_messages(&server, 1).await;
        assert_eq!(
            messages[1],
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "write-1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json!({ "content": "changed", "path": file_path }).to_string()
                    }
                }]
            }),
            "assistant echo must keep its tool_calls intact"
        );
        assert_eq!(
            messages[2],
            json!({
                "role": "tool",
                "tool_call_id": "write-1",
                "content": format!("updated `{file_path}`; 7 bytes written")
            })
        );

        assert_eq!(std::fs::read_to_string(file_path).unwrap(), "changed");
    }

    #[tokio::test]
    async fn a_denied_tool_round_trips_the_reason_as_a_non_error_result() {
        // Arrange
        let file = temp_file_with(b"initial");
        let file_path = file.path().to_str().unwrap();
        let server = MockServer::start().await;

        let deny_reason = "I changed my mind".to_string();

        mount_turns(
            &server,
            vec![
                tool_call_turn(&[(
                    "write-1",
                    "write_file",
                    json!({ "path": file_path, "content": "updated" }),
                )]),
                text_turn(&deny_reason),
            ],
        )
        .await;

        let mut handle = spawn_agent(test_provider(&server), test_workspace());

        handle
            .commands
            .send(AgentCommand::UserInput("Change the file.".to_string()))
            .await
            .unwrap();

        // Act
        let respond_to = loop {
            let event = handle.events.recv().await.unwrap();
            match event {
                AgentEvent::ApprovalRequest { respond_to, .. } => break respond_to,
                AgentEvent::ToolStarted { .. } => {
                    panic!("tool was reported as started before approval")
                }
                _ => {}
            }
        };
        respond_to
            .send(ApprovalDecision::Deny {
                reason: deny_reason.clone(),
            })
            .unwrap();

        let _ = collect_turn(&mut handle.events).await;
        let provider_request = nth_request_messages(&server, 1).await;

        // Assert
        assert_eq!(
            provider_request[2],
            json!({
                "content": format!("The user declined this tool call and said: \"{deny_reason}\". Do not assume the tool ran. Address their feedback, then retry if appropriate."),
                "role": "tool",
                "tool_call_id": "write-1",
            })
        );
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "initial");
    }

    #[tokio::test]
    async fn denied_and_allowed_sibling_tool_calls_both_produce_results() {
        // Arrange
        let file_one = temp_file_with(b"one");
        let file_two = temp_file_with(b"two");

        let file_one_path = file_one.path().to_str().unwrap();
        let file_two_path = file_two.path().to_str().unwrap();

        let server = MockServer::start().await;

        mount_turns(
            &server,
            vec![
                tool_call_turn(&[
                    (
                        "write-1",
                        "write_file",
                        json!({ "path": file_one_path, "content": "eno" }),
                    ),
                    (
                        "write-2",
                        "write_file",
                        json!({ "path": file_two_path, "content": "owt" }),
                    ),
                ]),
                text_turn("Finished"),
            ],
        )
        .await;

        let mut handle = spawn_agent(test_provider(&server), test_workspace());

        handle
            .commands
            .send(AgentCommand::UserInput("Make the change".to_string()))
            .await
            .unwrap();

        // Act & Assert
        let write_one_event = handle.events.recv().await.unwrap();
        let AgentEvent::ApprovalRequest {
            id: write_one_id,
            respond_to,
            ..
        } = write_one_event
        else {
            panic!("expected first approval request");
        };
        assert_eq!("write-1", write_one_id);

        respond_to
            .send(ApprovalDecision::Deny {
                reason: "Changed my mind".to_string(),
            })
            .unwrap();
        let write_one_response = handle.events.recv().await.unwrap();

        let write_two_event = handle.events.recv().await.unwrap();
        let AgentEvent::ApprovalRequest {
            id: write_two_id,
            respond_to,
            ..
        } = write_two_event
        else {
            panic!("expected second approval request");
        };
        assert_eq!("write-2", write_two_id);

        respond_to.send(ApprovalDecision::Allow).unwrap();

        let events = collect_turn(&mut handle.events).await;

        assert_eq!(4, events.len());

        let AgentEvent::ToolDenied { reason, .. } = write_one_response else {
            panic!("expected write_one_response to contain ToolDenied");
        };
        assert_eq!("Changed my mind", reason);

        assert!(
            matches!(
                &events[..],
                [
                    AgentEvent::ToolStarted{ name: start_name, .. },
                    AgentEvent::ToolFinished{ name: finished_name, .. },
                    AgentEvent::TextDelta(text),
                    AgentEvent::TurnComplete {
                        outcome: TurnOutcome::Completed { stop_reason },
                    }
                ] if start_name == "write_file"
                  && finished_name == "write_file"
                  && text == "Finished"
                  && *stop_reason == StopReason::EndTurn
            ),
            "expected a ToolStarted->ToolFinished->TextDelta->TurnComplete, got {events:?}"
        );

        assert_eq!(std::fs::read_to_string(file_one_path).unwrap(), "one");
        assert_eq!(std::fs::read_to_string(file_two_path).unwrap(), "owt");
    }

    #[tokio::test]
    async fn cancelling_during_approval_completes_the_turn_as_cancelled_without_executing() {
        // Arrange
        let file = temp_file_with(b"contents");
        let file_path = file.path().to_str().unwrap();
        let server = MockServer::start().await;

        mount_turns(
            &server,
            vec![tool_call_turn(&[(
                "write-1",
                "write_file",
                json!({ "path": file_path, "content": "changed" }),
            )])],
        )
        .await;

        let mut handle = spawn_agent(test_provider(&server), test_workspace());

        handle
            .commands
            .send(AgentCommand::UserInput("Make the change".to_string()))
            .await
            .unwrap();

        // Act
        let event = handle.events.recv().await.unwrap();
        let _respond_to = match event {
            AgentEvent::ApprovalRequest { respond_to, .. } => respond_to,
            _ => {
                panic!("unexpected event")
            }
        };

        handle.cancel.cancel();

        let events = collect_until_events_close(&mut handle.events).await;

        // Assert
        assert!(
            matches!(
                &events[..],
                [AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Cancelled
                }]
            ),
            "expected TurnComplete, got {events:?}"
        );

        assert_eq!(std::fs::read_to_string(file_path).unwrap(), "contents");
    }

    #[tokio::test]
    async fn cancelling_aborts_the_turn_promptly() {
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

    enum TestExecution {
        Fail(String),
        Cancel,
        Block { started: Arc<Notify> },
    }

    struct TestInvocation {
        execution: TestExecution,
    }

    #[async_trait]
    impl PreparedInvocation for TestInvocation {
        fn approval_requirement(&self) -> ApprovalRequirement {
            ApprovalRequirement::None
        }

        async fn execute(
            self: Box<Self>,
            cancel: CancellationToken,
        ) -> Result<String, ToolExecutionError> {
            match self.execution {
                TestExecution::Fail(error) => Err(ToolExecutionError::ToolError(error)),
                TestExecution::Cancel => Err(ToolExecutionError::Cancelled),
                TestExecution::Block { started } => {
                    started.notify_one();
                    cancel.cancelled().await;
                    Err(ToolExecutionError::Cancelled)
                }
            }
        }
    }

    struct BlockingTestTool {
        started: Arc<Notify>,
    }

    #[async_trait]
    impl Tool for BlockingTestTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "test_tool".to_string(),
                description: "Blocks until cancelled".to_string(),
                input_schema: json!({
                    "type": "object",
                    "additionalProperties": false,
                }),
            }
        }

        async fn prepare(&self, _input: Value) -> Result<Box<dyn PreparedInvocation>, String> {
            Ok(Box::new(TestInvocation {
                execution: TestExecution::Block {
                    started: Arc::clone(&self.started),
                },
            }))
        }
    }

    #[tokio::test]
    async fn cancelling_during_tool_execution_completes_turn_once_without_finishing_tool() {
        // Arrange
        let server = MockServer::start().await;

        mount_turns(
            &server,
            vec![tool_call_turn(&[("tool-1", "test_tool", json!({}))])],
        )
        .await;

        let (events_tx, mut events_rx) = mpsc::channel(64);
        let (commands_tx, commands_rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();

        let host_handle = HostHandle {
            cancel: cancel.clone(),
            commands: commands_rx,
            events: EventSink::new(events_tx),
        };

        let started = Arc::new(Notify::new());

        let session = test_session(
            &server,
            host_handle,
            vec![Box::new(BlockingTestTool {
                started: Arc::clone(&started),
            })],
        );

        let session_task = tokio::spawn(session.run());

        commands_tx
            .send(AgentCommand::UserInput("Run the test tool".to_string()))
            .await
            .unwrap();

        let started_event = timeout(Duration::from_secs(1), events_rx.recv())
            .await
            .expect("ToolStarted was not emitted")
            .expect("event channel closed unexpectedly");

        timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("tool invocation never started");

        // Act
        cancel.cancel();

        let remaining_events = collect_until_events_close(&mut events_rx).await;

        let session_result = timeout(Duration::from_secs(1), session_task)
            .await
            .expect("session did not stop promptly")
            .expect("session task panicked");

        // Assert
        assert_eq!(session_result, Err(AgentExit::Cancelled));
        assert!(matches!(
            started_event,
            AgentEvent::ToolStarted { ref name, .. } if name == "test_tool"
        ));

        assert!(matches!(
            remaining_events.as_slice(),
            [AgentEvent::TurnComplete {
                outcome: TurnOutcome::Cancelled,
            }]
        ));
    }

    #[tokio::test]
    async fn ordinary_tool_failure_returns_error_result_instead_of_cancelling() {
        // Arrange
        let (events_tx, _events_rx) = mpsc::channel(64);
        let (_commands_tx, commands_rx) = mpsc::channel(64);
        let event_sink = EventSink::new(events_tx.clone());

        let host_handle = HostHandle {
            cancel: CancellationToken::new(),
            commands: commands_rx,
            events: event_sink,
        };

        let mock_error = "Mock error".to_string();
        let mock_id = "call-1".to_string();

        // Act
        let result = execute_invocation(
            &host_handle,
            mock_id.as_str(),
            "test",
            &json!({}),
            Box::new(TestInvocation {
                execution: TestExecution::Fail(mock_error.clone()),
            }),
        )
        .await;

        // Assert
        assert_eq!(
            result,
            Ok(ToolResultData {
                content: mock_error,
                is_error: true,
                tool_use_id: mock_id,
            }),
        );
    }

    #[tokio::test]
    async fn tool_cancellation_becomes_agent_cancellation() {
        // Arrange
        let (events_tx, mut events_rx) = mpsc::channel(64);
        let (_commands_tx, commands_rx) = mpsc::channel(64);
        let event_sink = EventSink::new(events_tx.clone());

        let host_handle = HostHandle {
            cancel: CancellationToken::new(),
            commands: commands_rx,
            events: event_sink,
        };

        // Act
        let result = execute_invocation(
            &host_handle,
            "call-1",
            "test",
            &json!({}),
            Box::new(TestInvocation {
                execution: TestExecution::Cancel,
            }),
        )
        .await
        .unwrap_err();

        // Assert
        assert_eq!(result, AgentExit::Cancelled);

        let event = timeout(Duration::from_secs(1), events_rx.recv())
            .await
            .expect("event was not emitted")
            .expect("event channel closed");

        assert!(matches!(
            event,
            AgentEvent::ToolStarted { ref name, .. } if name == "test"
        ));
        assert!(matches!(
            events_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn frontend_disconnect_during_tool_execution_returns_disconnected() {
        // Arrange
        let (events_tx, mut events_rx) = mpsc::channel(64);
        let (_commands_tx, commands_rx) = mpsc::channel(64);
        let event_sink = EventSink::new(events_tx.clone());

        let host_handle = HostHandle {
            cancel: CancellationToken::new(),
            commands: commands_rx,
            events: event_sink,
        };

        let started = Arc::new(Notify::new());
        let invocation_started = Arc::clone(&started);

        let execution = tokio::spawn(async move {
            execute_invocation(
                &host_handle,
                "call-1",
                "test",
                &json!({}),
                Box::new(TestInvocation {
                    execution: TestExecution::Block {
                        started: invocation_started,
                    },
                }),
            )
            .await
        });

        let start_event = timeout(Duration::from_secs(1), events_rx.recv())
            .await
            .expect("ToolStarted was not emitted")
            .expect("event channel closed unexpectedly");

        // Confirm the invocation itself is executing.
        timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("tool invocation never started");

        // Act
        // Drop the receiver to disconnect the frontend.
        drop(events_rx);

        let result = timeout(Duration::from_secs(1), execution)
            .await
            .expect("tool execution did not stop promptly")
            .expect("execution task panicked")
            .unwrap_err();

        // Assert
        assert_eq!(result, AgentExit::Disconnected);

        assert!(matches!(
            start_event,
            AgentEvent::ToolStarted { ref name, .. } if name == "test"
        ));
    }

    #[tokio::test]
    async fn session_cancellation_during_tool_execution_returns_cancelled_without_finishing() {
        // Arrange
        let (events_tx, mut events_rx) = mpsc::channel(64);
        let (_commands_tx, commands_rx) = mpsc::channel(64);
        let event_sink = EventSink::new(events_tx.clone());
        let cancel = CancellationToken::new();

        let host_handle = HostHandle {
            cancel: cancel.clone(),
            commands: commands_rx,
            events: event_sink,
        };

        let started = Arc::new(Notify::new());
        let invocation_started = Arc::clone(&started);

        let execution = tokio::spawn(async move {
            execute_invocation(
                &host_handle,
                "call-1",
                "test",
                &json!({}),
                Box::new(TestInvocation {
                    execution: TestExecution::Block {
                        started: invocation_started,
                    },
                }),
            )
            .await
        });

        let started_event = timeout(Duration::from_secs(1), events_rx.recv())
            .await
            .expect("ToolStarted was not emitted")
            .expect("event channel closed unexpectedly");

        // Confirm the invocation itself is executing.
        timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("tool invocation never started");

        // Act
        cancel.cancel();

        let result = timeout(Duration::from_secs(1), execution)
            .await
            .expect("tool execution did not stop promptly")
            .expect("execution task panicked")
            .unwrap_err();

        // Assert
        assert_eq!(result, AgentExit::Cancelled);
        assert!(matches!(
            started_event,
            AgentEvent::ToolStarted { ref name, .. } if name == "test"
        ));
        assert!(matches!(
            events_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }
}
