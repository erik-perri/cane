use crate::Workspace;
use crate::approval::{
    ApprovalAuthorization, ApprovalCheck, ApprovalGate, ApprovalOutcome, request_approval,
};
use crate::journal::{
    ApprovalId, ErrorDetail, JournalApprovalDecision, JournalEntry, JournalError, RunEndReason,
    RunId, RunJournal, RunStarted, SessionId, SessionJournal, SessionStarted, ToolAuthorization,
    TurnAbortOutcome, TurnCommitOutcome, TurnId,
};
use crate::message::{ContentBlock, Message, Role, StopReason, ToolInput, ToolResultData};
use crate::protocol::{
    AgentCommand, AgentEvent, AgentExit, ApprovalSubject, EventSink, HostHandle, ShutdownReason,
    TurnOutcome,
};
use crate::provider::{ModelTurn, OpenAiClient, ProviderConfig, ProviderError};
use crate::session::SessionConfig;
use crate::tools::{PreparedInvocation, ToolExecutionError, ToolSet};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const MAX_PROVIDER_ROUNDS_PER_TURN: usize = 24;

pub struct AgentSession {
    client: OpenAiClient,
    host_handle: HostHandle,
    journal: RunJournal,
    tool_set: ToolSet,
}

pub struct AgentHandle {
    pub cancel: CancellationToken,
    pub commands: mpsc::Sender<AgentCommand>,
    pub events: mpsc::Receiver<AgentEvent>,
    journal_path: PathBuf,
    session_id: SessionId,
    task: JoinHandle<()>,
}

#[derive(Debug, Error)]
pub enum AgentStartError {
    #[error(transparent)]
    Journal(#[from] JournalError),

    #[error(transparent)]
    Provider(#[from] ProviderError),

    #[error("workspace path is not valid UTF-8: '{}'", path.display())]
    WorkspacePath { path: PathBuf },
}

#[derive(Default)]
struct TurnBudget {
    provider_rounds: usize,
}

struct AuthorizedToolCall<'a> {
    authorization: ApprovalAuthorization,
    id: &'a str,
    input: &'a Value,
    name: &'a str,
    turn_id: TurnId,
}

enum TurnResult {
    Completed { stop_reason: StopReason },
    Paused { reason: String },
    Failed { error: Option<ErrorDetail> },
    Cancelled,
}

impl TurnResult {
    fn outcome(&self) -> TurnOutcome {
        match self {
            Self::Completed { stop_reason } => TurnOutcome::Completed {
                stop_reason: stop_reason.clone(),
            },
            Self::Paused { reason } => TurnOutcome::Paused {
                reason: reason.clone(),
            },
            Self::Failed { .. } => TurnOutcome::Failed,
            Self::Cancelled => TurnOutcome::Cancelled,
        }
    }

    fn failed(category: &str, message: impl Into<String>) -> Self {
        Self::Failed {
            error: Some(ErrorDetail {
                category: category.to_string(),
                message: message.into(),
            }),
        }
    }

    fn unattributed_failure() -> Self {
        Self::Failed { error: None }
    }
}

impl TurnBudget {
    fn begin_provider_round(&mut self) -> Result<(), String> {
        if self.provider_rounds >= MAX_PROVIDER_ROUNDS_PER_TURN {
            return Err(format!(
                "turn paused after {MAX_PROVIDER_ROUNDS_PER_TURN} provider rounds; send another message to continue with the existing context"
            ));
        }

        self.provider_rounds += 1;
        Ok(())
    }
}

pub async fn spawn_agent(
    provider: ProviderConfig,
    workspace: Workspace,
    sessions: SessionConfig,
) -> Result<AgentHandle, AgentStartError> {
    let workspace_path = workspace
        .root()
        .to_str()
        .ok_or_else(|| AgentStartError::WorkspacePath {
            path: workspace.root().to_path_buf(),
        })?
        .to_string();

    let client = OpenAiClient::new(
        provider.base_url,
        provider.api_key,
        provider.model.clone(),
        provider.max_tokens,
    )?;
    let tool_set = ToolSet::new(Arc::new(workspace), None);
    let session_id = SessionId::generate();
    let run_id = RunId::generate();
    let mut journal = SessionJournal::create(sessions.sessions_directory(), session_id).await?;

    journal
        .append(JournalEntry::SessionStarted(SessionStarted {
            cane_version: sessions.cane_version().to_string(),
            instructions: sessions.instructions().to_string(),
            workspace: workspace_path,
        }))
        .await?;

    let model = provider.model;
    let provider_descriptor = client.provider_descriptor();

    journal
        .append(JournalEntry::RunStarted(RunStarted {
            approval_grants: Vec::new(),
            git: None,
            max_output_tokens: provider.max_tokens,
            model: model.clone(),
            provider: provider_descriptor.clone(),
            run_id,
            shell_policy: None,
            tool_catalog: tool_set.definitions().to_vec(),
        }))
        .await?;

    let journal = RunJournal::new(journal, model, provider_descriptor, run_id);
    let journal_path = journal.path().to_path_buf();
    let (events_tx, events_rx) = mpsc::channel(64);
    let (commands_tx, commands_rx) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task = tokio::spawn(async move {
        let events = EventSink::new(events_tx.clone());
        let host_handle = HostHandle {
            cancel: task_cancel,
            commands: commands_rx,
            events: events.clone(),
        };

        let session = AgentSession {
            client,
            host_handle,
            journal,
            tool_set,
        };

        run_and_report(session, &events).await;
    });

    Ok(AgentHandle {
        cancel,
        commands: commands_tx,
        events: events_rx,
        journal_path,
        session_id,
        task,
    })
}

async fn run_and_report(session: AgentSession, events: &EventSink) {
    if let Err(AgentExit::JournalFailed(message)) = session.run().await {
        let _ = events
            .emit(AgentEvent::Error(format!(
                "session journal failed: {message}"
            )))
            .await;
    }
}

impl AgentHandle {
    pub fn journal_path(&self) -> &Path {
        &self.journal_path
    }

    pub async fn join(self) -> Result<(), tokio::task::JoinError> {
        let Self {
            cancel: _,
            commands,
            events,
            journal_path: _,
            session_id: _,
            task,
        } = self;

        drop(commands);
        drop(events);
        task.await
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }
}

impl AgentSession {
    async fn run(mut self) -> Result<(), AgentExit> {
        tracing::debug!(
            journal_path = %self.journal.path().display(),
            session_id = %self.journal.session_id(),
            "agent run started"
        );

        match self.run_until_end().await {
            Ok(reason) => {
                self.journal.end_run(reason).await?;
                Ok(())
            }
            Err(exit @ AgentExit::Cancelled) => {
                self.journal
                    .end_run(RunEndReason::ActiveTurnCancelled)
                    .await?;
                Err(exit)
            }
            Err(exit @ AgentExit::Disconnected) => {
                self.journal
                    .end_run(RunEndReason::FrontendDisconnected)
                    .await?;
                Err(exit)
            }
            Err(exit @ AgentExit::JournalFailed(_)) => Err(exit),
        }
    }

    async fn run_until_end(&mut self) -> Result<RunEndReason, AgentExit> {
        let mut history = Vec::new();
        let mut approval_gate = ApprovalGate::new();

        loop {
            let prompt = tokio::select! {
                biased;
                _ = self.host_handle.cancel.cancelled() => {
                    return Ok(RunEndReason::IdleCancelled);
                }
                command = self.host_handle.commands.recv() => {
                    match command {
                        Some(AgentCommand::Shutdown(reason)) => {
                            return Ok(shutdown_reason(reason));
                        }
                        Some(AgentCommand::UserInput(prompt)) => prompt,
                        None => return Err(AgentExit::Disconnected),
                    }
                }
                _ = self.host_handle.events.closed() => {
                    return Err(AgentExit::Disconnected);
                }
            };

            let turn_start = history.len();
            let user_message = Message {
                role: Role::User,
                content: vec![ContentBlock::Text { text: prompt }],
            };
            let turn_id = self.journal.start_turn(&user_message).await?;
            history.push(user_message);

            match self
                .run_turn(turn_id, &mut history, &mut approval_gate)
                .await
            {
                Ok(result) => {
                    let session_over = matches!(result, TurnResult::Cancelled);
                    let needs_rollback =
                        session_over || matches!(result, TurnResult::Failed { .. });
                    let outcome = result.outcome();

                    self.record_turn_outcome(turn_id, &result).await?;

                    if needs_rollback {
                        // Truncate the history on failure so we don't leave an incomplete
                        // turn in the next request.
                        history.truncate(turn_start);
                    }

                    if session_over {
                        let _ = self
                            .host_handle
                            .events
                            .emit(AgentEvent::TurnComplete { outcome })
                            .await;
                        return Err(AgentExit::Cancelled);
                    }

                    self.host_handle
                        .events
                        .emit(AgentEvent::TurnComplete { outcome })
                        .await?;
                }
                Err(AgentExit::Cancelled) => {
                    self.journal
                        .abort_turn(turn_id, TurnAbortOutcome::Cancelled)
                        .await?;

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
                Err(AgentExit::Disconnected) => {
                    let result = TurnResult::failed(
                        "frontend_disconnected",
                        "frontend disconnected during the active turn",
                    );
                    self.record_turn_outcome(turn_id, &result).await?;
                    history.truncate(turn_start);
                    return Err(AgentExit::Disconnected);
                }
                Err(exit @ AgentExit::JournalFailed(_)) => return Err(exit),
            }
        }
    }

    async fn record_turn_outcome(
        &mut self,
        turn_id: TurnId,
        result: &TurnResult,
    ) -> Result<(), AgentExit> {
        match result {
            TurnResult::Completed { stop_reason } => {
                self.journal
                    .commit_turn(
                        turn_id,
                        TurnCommitOutcome::Completed {
                            stop_reason: stop_reason.clone(),
                        },
                    )
                    .await?;
            }
            TurnResult::Paused { reason } => {
                self.journal
                    .commit_turn(
                        turn_id,
                        TurnCommitOutcome::Paused {
                            reason: reason.clone(),
                        },
                    )
                    .await?;
            }
            TurnResult::Failed { error } => {
                self.journal
                    .abort_turn(
                        turn_id,
                        TurnAbortOutcome::Failed {
                            error: error.clone(),
                        },
                    )
                    .await?;
            }
            TurnResult::Cancelled => {
                self.journal
                    .abort_turn(turn_id, TurnAbortOutcome::Cancelled)
                    .await?;
            }
        }

        Ok(())
    }

    async fn run_turn(
        &mut self,
        turn_id: TurnId,
        history: &mut Vec<Message>,
        gate: &mut ApprovalGate,
    ) -> Result<TurnResult, AgentExit> {
        let mut budget = TurnBudget::default();

        loop {
            if let Err(reason) = budget.begin_provider_round() {
                return Ok(TurnResult::Paused { reason });
            }

            let provider_round_id = self.journal.provider_round_started(turn_id).await?;
            let provider_started = Instant::now();
            let stream_result = tokio::select! {
                _ = self.host_handle.events.closed() => return Err(AgentExit::Disconnected),
                result = self.client.stream_message(history, self.tool_set.definitions(), self.host_handle.events.sender(), &self.host_handle.cancel) => {
                    result
                }
            };
            let latency_ms = elapsed_ms(provider_started);

            let model_turn = match stream_result {
                Ok(result) => {
                    self.journal
                        .provider_round_completed(latency_ms, provider_round_id, &result, turn_id)
                        .await?;
                    result
                }
                Err(error) => {
                    let cancelled = matches!(&error, ProviderError::Cancelled);

                    if cancelled {
                        self.journal
                            .provider_round_cancelled(latency_ms, provider_round_id, turn_id)
                            .await?;
                    } else {
                        self.journal
                            .provider_round_failed(
                                provider_error_detail(&error),
                                latency_ms,
                                provider_round_id,
                                None,
                                turn_id,
                            )
                            .await?;
                    }

                    self.host_handle
                        .events
                        .emit(AgentEvent::Error(error.to_string()))
                        .await?;

                    return if cancelled {
                        Ok(TurnResult::Cancelled)
                    } else {
                        Ok(TurnResult::unattributed_failure())
                    };
                }
            };

            let ModelTurn {
                message: assistant_msg,
                stop_reason,
                ..
            } = model_turn;

            tracing::debug!(history_len = history.len(), ?stop_reason);

            if let Err(error) = validate_assistant_message(&assistant_msg, &stop_reason) {
                self.host_handle
                    .events
                    .emit(AgentEvent::Error(error.to_string()))
                    .await?;
                return Ok(TurnResult::failed("invalid_assistant_message", error));
            }

            self.journal.message_added(&assistant_msg, turn_id).await?;
            history.push(assistant_msg);

            if stop_reason != StopReason::ToolUse {
                return Ok(TurnResult::Completed { stop_reason });
            }

            let mut results = Vec::new();

            for block in &history.last().expect("just pushed").content {
                match block {
                    ContentBlock::ToolUse {
                        id, input, name, ..
                    } => {
                        let tool_result = match input {
                            ToolInput::Valid(input) => {
                                self.execute_tool_call(turn_id, id, name, input, gate)
                                    .await?
                            }
                            ToolInput::Invalid(raw) => {
                                let error = format!(
                                    "invalid input for tool `{name}`: arguments are not valid JSON: {raw}"
                                );

                                self.journal
                                    .tool_rejected("invalid_input", id, name, turn_id)
                                    .await?;
                                self.host_handle
                                    .events
                                    .emit(AgentEvent::ToolRejected {
                                        name: name.to_string(),
                                        error: error.clone(),
                                    })
                                    .await?;

                                failed_tool_result(id, error)
                            }
                        };

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

                return Ok(TurnResult::failed(
                    "agent_invariant",
                    "no tool results were generated",
                ));
            }

            let result_message = Message {
                role: Role::User,
                content: results,
            };
            self.journal.message_added(&result_message, turn_id).await?;
            history.push(result_message);
        }
    }

    async fn execute_tool_call(
        &mut self,
        turn_id: TurnId,
        id: &str,
        name: &str,
        input: &Value,
        gate: &mut ApprovalGate,
    ) -> Result<ToolResultData, AgentExit> {
        let invocation = match prepare_tool_call(&self.tool_set, id, name, input).await {
            Ok(invocation) => invocation,
            Err(result) => {
                self.journal
                    .tool_rejected("preparation_failed", id, name, turn_id)
                    .await?;
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

        let subject = ApprovalSubject::tool_call(id, name);
        let available_lifetimes = invocation.available_grant_lifetimes().to_vec();
        let authorization = match gate.check(invocation.approval_requirement(), &subject) {
            ApprovalCheck::Authorized(authorization) => authorization,
            ApprovalCheck::RequiresDecision => {
                let approval_id = ApprovalId::generate();
                self.journal
                    .approval_requested(
                        approval_id,
                        available_lifetimes.clone(),
                        subject.clone(),
                        turn_id,
                    )
                    .await?;
                let decision = request_approval(
                    available_lifetimes.clone(),
                    input,
                    subject.clone(),
                    &self.host_handle.events,
                    &self.host_handle.cancel,
                )
                .await?;

                match gate.apply_decision(&available_lifetimes, approval_id, &decision, &subject) {
                    ApprovalOutcome::Authorized(authorization) => {
                        self.journal
                            .approval_decided(approval_id, journal_approval_decision(&decision))
                            .await?;
                        authorization
                    }
                    ApprovalOutcome::Denied { reason } => {
                        self.journal
                            .approval_decided(approval_id, journal_approval_decision(&decision))
                            .await?;
                        self.journal
                            .tool_rejected("approval_denied", id, name, turn_id)
                            .await?;
                        self.host_handle
                            .events
                            .emit(AgentEvent::ToolDenied {
                                name: name.to_string(),
                                reason: reason.to_string(),
                            })
                            .await?;

                        return Ok(denied_tool_result(id, &reason));
                    }
                    ApprovalOutcome::InvalidGrant { reason } => {
                        self.journal
                            .tool_rejected("invalid_approval_grant", id, name, turn_id)
                            .await?;
                        self.host_handle
                            .events
                            .emit(AgentEvent::ToolRejected {
                                error: reason.clone(),
                                name: name.to_string(),
                            })
                            .await?;

                        return Ok(failed_tool_result(id, reason));
                    }
                }
            }
        };

        tracing::debug!(
            ?authorization,
            tool_call_id = id,
            tool_name = name,
            "tool call authorized"
        );

        execute_invocation(
            &self.host_handle,
            &mut self.journal,
            AuthorizedToolCall {
                authorization,
                id,
                input,
                name,
                turn_id,
            },
            invocation,
        )
        .await
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

fn denied_tool_result(id: &str, reason: &str) -> ToolResultData {
    let content = if reason.trim().is_empty() {
        "The user declined this tool call. Do not assume the tool ran. Continue without it or ask the user what they prefer."
            .to_string()
    } else {
        format!(
            "The user declined this tool call and said: \"{reason}\". Do not assume the tool ran. Address their feedback, then retry if appropriate."
        )
    };

    ToolResultData {
        content,
        is_error: false,
        tool_use_id: id.to_string(),
    }
}

fn validate_assistant_message(
    message: &Message,
    stop_reason: &StopReason,
) -> Result<(), &'static str> {
    let mut has_tool_use = false;
    let mut tool_call_ids = HashSet::new();

    for block in &message.content {
        match block {
            ContentBlock::Text { .. } => {}
            ContentBlock::ToolUse { id, .. } => {
                has_tool_use = true;
                if !tool_call_ids.insert(id) {
                    return Err("assistant message repeats a tool call ID");
                }
            }
            ContentBlock::ToolResult { .. } => {
                return Err("assistant message contains a tool result");
            }
        }
    }

    if has_tool_use != (*stop_reason == StopReason::ToolUse) {
        return Err("assistant tool calls do not agree with the provider stop reason");
    }

    Ok(())
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn journal_approval_decision(decision: &crate::ApprovalDecision) -> JournalApprovalDecision {
    match decision {
        crate::ApprovalDecision::Grant(grant) => JournalApprovalDecision::Grant {
            grant: grant.clone(),
        },
        crate::ApprovalDecision::Deny { reason } => JournalApprovalDecision::Deny {
            reason: reason.clone(),
        },
    }
}

fn shutdown_reason(reason: ShutdownReason) -> RunEndReason {
    match reason {
        ShutdownReason::InputClosed => RunEndReason::InputClosed,
        ShutdownReason::UserQuit => RunEndReason::UserQuit,
    }
}

fn provider_error_detail(error: &ProviderError) -> ErrorDetail {
    let category = match error {
        ProviderError::Api { .. } => "api",
        ProviderError::Cancelled => "cancelled",
        ProviderError::InvalidBaseUrl { .. } => "invalid_base_url",
        ProviderError::Network(_) => "network",
        ProviderError::Parsing(_) => "parsing",
        ProviderError::Protocol { .. } => "protocol",
    };

    ErrorDetail {
        category: category.to_string(),
        message: error.to_string(),
    }
}

fn tool_authorization(authorization: ApprovalAuthorization) -> ToolAuthorization {
    match authorization {
        ApprovalAuthorization::Granted { approval_id, grant } => {
            ToolAuthorization::Granted { approval_id, grant }
        }
        ApprovalAuthorization::NotRequired => ToolAuthorization::NotRequired,
    }
}

async fn execute_invocation(
    host_handle: &HostHandle,
    journal: &mut RunJournal,
    call: AuthorizedToolCall<'_>,
    invocation: Box<dyn PreparedInvocation>,
) -> Result<ToolResultData, AgentExit> {
    journal
        .tool_started(
            tool_authorization(call.authorization),
            None,
            call.id,
            call.name,
            call.turn_id,
        )
        .await?;

    host_handle
        .events
        .emit(AgentEvent::ToolStarted {
            input: call.input.clone(),
            name: call.name.to_string(),
        })
        .await?;

    let execution_cancel = host_handle.cancel.child_token();
    let execution_started = Instant::now();
    let tool_future = invocation.execute(execution_cancel.clone());

    let execution_result = tokio::select! {
        _ = host_handle.events.closed() => {
            execution_cancel.cancel();
            journal
                .tool_cancelled(
                    elapsed_ms(execution_started),
                    call.id,
                    call.turn_id,
                )
                .await?;
            return Err(AgentExit::Disconnected);
        }
        _ = host_handle.cancel.cancelled() => {
            execution_cancel.cancel();
            journal
                .tool_cancelled(
                    elapsed_ms(execution_started),
                    call.id,
                    call.turn_id,
                )
                .await?;
            return Err(AgentExit::Cancelled);
        }
        result = tool_future => result,
    };
    let duration_ms = elapsed_ms(execution_started);

    let (content, is_error) = match execution_result {
        Ok(content) => {
            journal
                .tool_completed(duration_ms, None, call.id, call.turn_id)
                .await?;
            (content, false)
        }
        Err(ToolExecutionError::ToolError(error)) => {
            journal
                .tool_failed(duration_ms, "tool_error", call.id, call.turn_id)
                .await?;
            (error, true)
        }
        Err(ToolExecutionError::Cancelled) => {
            journal
                .tool_cancelled(duration_ms, call.id, call.turn_id)
                .await?;
            return Err(AgentExit::Cancelled);
        }
    };

    host_handle
        .events
        .emit(AgentEvent::ToolFinished {
            is_error,
            name: call.name.to_string(),
            output: content.clone(),
        })
        .await?;

    Ok(ToolResultData {
        content,
        is_error,
        tool_use_id: call.id.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::{
        InjectedFlushFailure, JournalRecord, SessionProjection, parse_journal, project_journal,
    };
    use crate::protocol::ApprovalRequirement;
    use crate::tools::{Tool, ToolDefinition};
    use crate::{ApprovalDecision, ApprovalLifetime};
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tempfile::{NamedTempFile, TempDir};
    use tokio::sync::Notify;
    use tokio::time::timeout;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct FaultInjectedSession {
        commands: mpsc::Sender<AgentCommand>,
        events: mpsc::Receiver<AgentEvent>,
        journal_path: PathBuf,
        sessions: TempDir,
        task: JoinHandle<()>,
    }

    struct CountingInvocation {
        executions: Arc<AtomicUsize>,
    }

    struct CountingTool {
        executions: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl PreparedInvocation for CountingInvocation {
        fn approval_requirement(&self) -> ApprovalRequirement {
            ApprovalRequirement::None
        }

        async fn execute(
            self: Box<Self>,
            _cancel: CancellationToken,
        ) -> Result<String, ToolExecutionError> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            Ok("executed".to_string())
        }
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                description: "Counts executions".to_string(),
                input_schema: json!({
                    "type": "object",
                    "additionalProperties": false,
                }),
                name: "counting_tool".to_string(),
            }
        }

        async fn prepare(&self, _input: Value) -> Result<Box<dyn PreparedInvocation>, String> {
            Ok(Box::new(CountingInvocation {
                executions: Arc::clone(&self.executions),
            }))
        }
    }

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

    fn malformed_tool_call_turn(id: &str, name: &str, arguments: &str) -> ResponseTemplate {
        sse_response(&[
            stream_chunk(
                json!({
                    "tool_calls": [{
                        "index": 0,
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": arguments }
                    }]
                }),
                None,
            ),
            stream_chunk(json!({}), Some("tool_calls")),
        ])
    }

    fn assistant_tool_call(id: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_string(),
            input: ToolInput::Valid(json!({})),
            name: "test_tool".to_string(),
        }
    }

    #[test]
    fn assistant_tool_calls_require_a_tool_use_stop_reason() {
        // Arrange
        let message = Message {
            content: vec![assistant_tool_call("call_1")],
            role: Role::Assistant,
        };

        // Act
        let result = validate_assistant_message(&message, &StopReason::MaxTokens);

        // Assert
        assert_eq!(
            result,
            Err("assistant tool calls do not agree with the provider stop reason")
        );
    }

    #[test]
    fn assistant_tool_call_ids_must_be_unique_within_a_message() {
        // Arrange
        let message = Message {
            content: vec![assistant_tool_call("call_1"), assistant_tool_call("call_1")],
            role: Role::Assistant,
        };

        // Act
        let result = validate_assistant_message(&message, &StopReason::ToolUse);

        // Assert
        assert_eq!(result, Err("assistant message repeats a tool call ID"));
    }

    #[test]
    fn denied_tool_results_only_quote_a_nonempty_reason() {
        // Arrange
        let tool_call_id = "call_1";

        // Act
        let unexplained = denied_tool_result(tool_call_id, " \t");
        let explained = denied_tool_result(tool_call_id, "use the existing build output");

        // Assert
        assert_eq!(
            unexplained,
            ToolResultData {
                content: "The user declined this tool call. Do not assume the tool ran. Continue without it or ask the user what they prefer."
                    .to_string(),
                is_error: false,
                tool_use_id: tool_call_id.to_string(),
            }
        );
        assert_eq!(
            explained,
            ToolResultData {
                content: "The user declined this tool call and said: \"use the existing build output\". Do not assume the tool ran. Address their feedback, then retry if appropriate."
                    .to_string(),
                is_error: false,
                tool_use_id: tool_call_id.to_string(),
            }
        );
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

    async fn test_run_journal() -> (RunJournal, TempDir) {
        let sessions = TempDir::new().unwrap();
        let journal = SessionJournal::create(sessions.path(), SessionId::generate())
            .await
            .unwrap();
        let journal = RunJournal::new(
            journal,
            "test-model".to_string(),
            crate::ProviderDescriptor {
                adapter: crate::ProviderAdapter::OpenAiCompatible,
                endpoint: "https://example.test/chat/completions".to_string(),
            },
            RunId::generate(),
        );

        (journal, sessions)
    }

    async fn test_session(
        server: &MockServer,
        host_handle: HostHandle,
        tools: Vec<Box<dyn Tool>>,
    ) -> (AgentSession, TempDir) {
        let provider = test_provider(server);
        let model = provider.model.clone();
        let client = OpenAiClient::new(
            provider.base_url,
            provider.api_key,
            provider.model,
            provider.max_tokens,
        )
        .unwrap();
        let provider_descriptor = client.provider_descriptor();
        let tool_set = ToolSet::from_tools(tools);
        let sessions = TempDir::new().unwrap();
        let session_id = SessionId::generate();
        let run_id = RunId::generate();
        let mut journal = SessionJournal::create(sessions.path(), session_id)
            .await
            .unwrap();
        journal
            .append(JournalEntry::SessionStarted(SessionStarted {
                cane_version: "test-cane-version".to_string(),
                instructions: String::new(),
                workspace: std::env::temp_dir().to_string_lossy().into_owned(),
            }))
            .await
            .unwrap();
        journal
            .append(JournalEntry::RunStarted(RunStarted {
                approval_grants: Vec::new(),
                git: None,
                max_output_tokens: provider.max_tokens,
                model: model.clone(),
                provider: provider_descriptor.clone(),
                run_id,
                shell_policy: None,
                tool_catalog: tool_set.definitions().to_vec(),
            }))
            .await
            .unwrap();
        let journal = RunJournal::new(journal, model, provider_descriptor, run_id);

        (
            AgentSession {
                client,
                host_handle,
                journal,
                tool_set,
            },
            sessions,
        )
    }

    async fn spawn_test_agent(server: &MockServer) -> (AgentHandle, TempDir) {
        let sessions = TempDir::new().unwrap();
        let config = SessionConfig::new("test-cane-version", "", sessions.path());
        let handle = spawn_agent(test_provider(server), test_workspace(), config)
            .await
            .unwrap();

        (handle, sessions)
    }

    async fn spawn_fault_injected_session(
        server: &MockServer,
        failure: InjectedFlushFailure,
        tools: Vec<Box<dyn Tool>>,
    ) -> FaultInjectedSession {
        let (events_tx, events_rx) = mpsc::channel(64);
        let (commands_tx, commands_rx) = mpsc::channel(64);
        let events = EventSink::new(events_tx);
        let (mut session, sessions) = test_session(
            server,
            HostHandle {
                cancel: CancellationToken::new(),
                commands: commands_rx,
                events: events.clone(),
            },
            tools,
        )
        .await;
        session.journal.inject_flush_failure(failure);
        let journal_path = session.journal.path().to_path_buf();
        let task = tokio::spawn(async move {
            run_and_report(session, &events).await;
        });

        FaultInjectedSession {
            commands: commands_tx,
            events: events_rx,
            journal_path,
            sessions,
            task,
        }
    }

    async fn run_agent(prompt: &str, server: &MockServer) -> Vec<AgentEvent> {
        let (mut handle, _sessions) = spawn_test_agent(server).await;
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

    async fn finish_and_project(handle: AgentHandle) -> (Vec<JournalRecord>, SessionProjection) {
        let journal_path = handle.journal_path().to_path_buf();
        handle
            .commands
            .send(AgentCommand::Shutdown(ShutdownReason::InputClosed))
            .await
            .unwrap();
        timeout(Duration::from_secs(1), handle.join())
            .await
            .expect("agent task did not join")
            .expect("agent task panicked");
        let records = read_journal_records(&journal_path).await;
        let projection = project_journal(&records).unwrap();
        (records, projection)
    }

    async fn read_journal_records(path: &Path) -> Vec<JournalRecord> {
        let contents = tokio::fs::read(path).await.unwrap();
        parse_journal(&contents).unwrap()
    }

    #[tokio::test]
    async fn startup_flushes_session_and_run_metadata_before_returning() {
        // Arrange
        let server = MockServer::start().await;
        let sessions = TempDir::new().unwrap();
        let workspace_directory = TempDir::new().unwrap();
        let workspace = Workspace::new(workspace_directory.path().to_path_buf()).unwrap();
        let expected_workspace = workspace.root().to_str().unwrap().to_string();
        let config = SessionConfig::new("test-cane-version", "Be precise.", sessions.path());

        // Act
        let handle = spawn_agent(test_provider(&server), workspace, config)
            .await
            .unwrap();
        let contents = tokio::fs::read(handle.journal_path()).await.unwrap();
        let records = parse_journal(&contents).unwrap();

        // Assert
        assert_eq!(records.len(), 2);
        assert_eq!(handle.session_id(), records[0].session_id);
        assert_eq!(
            handle.journal_path(),
            sessions
                .path()
                .join(format!("{}.jsonl", handle.session_id()))
        );

        let JournalEntry::SessionStarted(started) = &records[0].entry else {
            panic!("expected session_started");
        };
        assert_eq!(started.cane_version, "test-cane-version");
        assert_eq!(started.instructions, "Be precise.");
        assert_eq!(started.workspace, expected_workspace);

        let JournalEntry::RunStarted(started) = &records[1].entry else {
            panic!("expected run_started");
        };
        assert_eq!(started.max_output_tokens, 1234);
        assert_eq!(started.model, "test-model");
        assert_eq!(
            started.provider,
            crate::ProviderDescriptor {
                adapter: crate::ProviderAdapter::OpenAiCompatible,
                endpoint: format!("{}/chat/completions", server.uri()),
            }
        );
        assert_eq!(
            started
                .tool_catalog
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec!["edit_file", "glob", "grep", "read_file", "write_file"]
        );
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "startup made a provider request"
        );

        timeout(Duration::from_secs(1), handle.join())
            .await
            .expect("agent task did not join")
            .expect("agent task panicked");
    }

    #[tokio::test]
    async fn journal_creation_failure_prevents_agent_startup() {
        // Arrange
        let server = MockServer::start().await;
        let sessions_file = NamedTempFile::new().unwrap();
        let config = SessionConfig::new("test-cane-version", "", sessions_file.path());

        // Act
        let result = spawn_agent(test_provider(&server), test_workspace(), config).await;

        // Assert
        assert!(matches!(result, Err(AgentStartError::Journal(_))));
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "failed startup made a provider request"
        );
    }

    #[tokio::test]
    async fn provider_validation_failure_does_not_create_a_journal() {
        // Arrange
        let parent = TempDir::new().unwrap();
        let sessions_directory = parent.path().join("sessions");
        let config = SessionConfig::new("test-cane-version", "", &sessions_directory);
        let provider = ProviderConfig {
            base_url: "not a URL".to_string(),
            api_key: "test-key".to_string(),
            model: "test-model".to_string(),
            max_tokens: 1234,
        };

        // Act
        let result = spawn_agent(provider, test_workspace(), config).await;

        // Assert
        assert!(matches!(result, Err(AgentStartError::Provider(_))));
        assert!(!sessions_directory.exists());
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
    async fn a_completed_text_turn_is_committed_to_the_projected_history() {
        // Arrange
        let server = MockServer::start().await;
        mount_turns(&server, vec![text_turn("Hello world")]).await;
        let (mut handle, _sessions) = spawn_test_agent(&server).await;
        handle
            .commands
            .send(AgentCommand::UserInput("Say hi".to_string()))
            .await
            .unwrap();

        // Act
        let events = collect_turn(&mut handle.events).await;
        let (records, projection) = finish_and_project(handle).await;

        // Assert
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnComplete {
                outcome: TurnOutcome::Completed {
                    stop_reason: StopReason::EndTurn,
                },
            })
        ));
        assert_eq!(
            projection.messages,
            vec![
                Message {
                    content: vec![ContentBlock::Text {
                        text: "Say hi".to_string(),
                    }],
                    role: Role::User,
                },
                Message {
                    content: vec![ContentBlock::Text {
                        text: "Hello world".to_string(),
                    }],
                    role: Role::Assistant,
                },
            ]
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| { matches!(&record.entry, JournalEntry::ProviderRoundStarted(_)) })
                .count(),
            1
        );
        assert!(
            records
                .iter()
                .any(|record| matches!(&record.entry, JournalEntry::TurnCommitted(_)))
        );
        assert!(matches!(
            records.last().map(|record| &record.entry),
            Some(JournalEntry::RunEnded(crate::journal::RunEnded {
                reason: RunEndReason::InputClosed,
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn a_failed_provider_round_aborts_without_projecting_the_user_message() {
        // Arrange
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("provider unavailable"))
            .expect(1)
            .mount(&server)
            .await;
        let (mut handle, _sessions) = spawn_test_agent(&server).await;
        handle
            .commands
            .send(AgentCommand::UserInput("Say hi".to_string()))
            .await
            .unwrap();

        // Act
        let events = collect_turn(&mut handle.events).await;
        let (records, projection) = finish_and_project(handle).await;

        // Assert
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnComplete {
                outcome: TurnOutcome::Failed,
            })
        ));
        assert!(projection.messages.is_empty());
        assert!(records.iter().any(|record| matches!(
            &record.entry,
            JournalEntry::ProviderRoundFailed(failed)
                if failed.error.category == "api"
                    && failed.error.message.contains("provider unavailable")
        )));
        assert!(matches!(
            records.iter().rev().nth(1).map(|record| &record.entry),
            Some(JournalEntry::TurnAborted(crate::journal::TurnAborted {
                outcome: TurnAbortOutcome::Failed { error: None },
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn an_agent_validation_failure_is_preserved_on_the_turn_abort() {
        // Arrange
        let server = MockServer::start().await;
        let response = sse_response(&[
            stream_chunk(
                json!({
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{}"
                        }
                    }]
                }),
                None,
            ),
            stream_chunk(json!({}), Some("length")),
        ]);
        mount_turns(&server, vec![response]).await;
        let (mut handle, _sessions) = spawn_test_agent(&server).await;
        handle
            .commands
            .send(AgentCommand::UserInput("Read something".to_string()))
            .await
            .unwrap();

        // Act
        let events = collect_turn(&mut handle.events).await;
        let (records, projection) = finish_and_project(handle).await;

        // Assert
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnComplete {
                outcome: TurnOutcome::Failed,
            })
        ));
        assert!(projection.messages.is_empty());
        assert!(
            records
                .iter()
                .any(|record| matches!(&record.entry, JournalEntry::ProviderRoundCompleted(_)))
        );
        assert!(matches!(
            records.iter().rev().nth(1).map(|record| &record.entry),
            Some(JournalEntry::TurnAborted(crate::journal::TurnAborted {
                outcome: TurnAbortOutcome::Failed {
                    error: Some(ErrorDetail { category, message }),
                },
                ..
            })) if category == "invalid_assistant_message"
                && message == "assistant tool calls do not agree with the provider stop reason"
        ));
    }

    #[tokio::test]
    async fn a_flush_failure_before_tool_start_prevents_execution_and_stops_the_run() {
        // Arrange
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![tool_call_turn(&[("call-1", "counting_tool", json!({}))])],
        )
        .await;
        let executions = Arc::new(AtomicUsize::new(0));
        let FaultInjectedSession {
            commands,
            mut events,
            journal_path,
            sessions: _sessions,
            task,
        } = spawn_fault_injected_session(
            &server,
            InjectedFlushFailure::ToolStarted,
            vec![Box::new(CountingTool {
                executions: Arc::clone(&executions),
            })],
        )
        .await;

        // Act
        commands
            .send(AgentCommand::UserInput("Run the tool".to_string()))
            .await
            .unwrap();
        commands
            .send(AgentCommand::UserInput(
                "This turn must not start".to_string(),
            ))
            .await
            .unwrap();
        let observed_events = collect_until_events_close(&mut events).await;
        task.await.unwrap();
        let records = read_journal_records(&journal_path).await;
        let projection = project_journal(&records).unwrap();
        let requests = server.received_requests().await.unwrap();

        // Assert
        assert_eq!(executions.load(Ordering::SeqCst), 0);
        assert!(matches!(
            observed_events.as_slice(),
            [AgentEvent::Error(error)]
                if error.contains("session journal failed")
                    && error.contains("injected flush failure")
        ));
        assert_eq!(requests.len(), 1);
        assert_eq!(
            records
                .iter()
                .filter(|record| matches!(&record.entry, JournalEntry::TurnStarted(_)))
                .count(),
            1,
            "a queued user input must not start after the journal failure"
        );
        assert!(projection.messages.is_empty());
        assert_eq!(projection.warnings.len(), 2);
        assert!(matches!(
            records.last().map(|record| &record.entry),
            Some(JournalEntry::ToolStarted(_))
        ));
        assert!(!records.iter().any(|record| matches!(
            &record.entry,
            JournalEntry::ToolCompleted(_) | JournalEntry::RunEnded(_)
        )));
    }

    #[tokio::test]
    async fn a_flush_failure_after_tool_execution_stops_without_retrying_the_tool() {
        // Arrange
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![tool_call_turn(&[("call-1", "counting_tool", json!({}))])],
        )
        .await;
        let executions = Arc::new(AtomicUsize::new(0));
        let FaultInjectedSession {
            commands,
            mut events,
            journal_path,
            sessions: _sessions,
            task,
        } = spawn_fault_injected_session(
            &server,
            InjectedFlushFailure::ToolCompleted,
            vec![Box::new(CountingTool {
                executions: Arc::clone(&executions),
            })],
        )
        .await;

        // Act
        commands
            .send(AgentCommand::UserInput("Run the tool".to_string()))
            .await
            .unwrap();
        let observed_events = collect_until_events_close(&mut events).await;
        task.await.unwrap();
        let records = read_journal_records(&journal_path).await;
        let projection = project_journal(&records).unwrap();
        let requests = server.received_requests().await.unwrap();

        // Assert
        assert_eq!(executions.load(Ordering::SeqCst), 1);
        assert!(matches!(
            observed_events.as_slice(),
            [
                AgentEvent::ToolStarted { name, .. },
                AgentEvent::Error(error),
            ] if name == "counting_tool"
                && error.contains("session journal failed")
                && error.contains("injected flush failure")
        ));
        assert_eq!(requests.len(), 1);
        assert!(projection.messages.is_empty());
        assert_eq!(projection.warnings.len(), 2);
        assert!(matches!(
            records.last().map(|record| &record.entry),
            Some(JournalEntry::ToolCompleted(_))
        ));
        assert!(
            !records
                .iter()
                .any(|record| matches!(&record.entry, JournalEntry::RunEnded(_)))
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| matches!(&record.entry, JournalEntry::MessageAdded(_)))
                .count(),
            2,
            "the failed tool terminal write must not be followed by a tool-result message"
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

        assert_eq!(
            names,
            vec!["edit_file", "glob", "grep", "read_file", "write_file"]
        );
    }

    #[tokio::test]
    async fn agent_waits_for_input_and_exits_when_command_channel_closes() {
        // Arrange
        let server = MockServer::start().await;
        let (mut handle, _sessions) = spawn_test_agent(&server).await;
        let journal_path = handle.journal_path().to_path_buf();

        // Act
        tokio::task::yield_now().await;
        let waited_for_input = !handle.task.is_finished();
        drop(handle.commands);
        let events = collect_until_events_close(&mut handle.events).await;
        let records = read_journal_records(&journal_path).await;

        // Assert
        assert!(waited_for_input, "agent exited while its frontend was live");
        assert!(
            events.is_empty(),
            "idle session emitted unexpected events: {events:?}"
        );
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "idle session made a provider request"
        );
        assert!(matches!(
            records.last().map(|record| &record.entry),
            Some(JournalEntry::RunEnded(crate::journal::RunEnded {
                reason: RunEndReason::FrontendDisconnected,
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn an_explicit_user_quit_ends_the_run_cleanly() {
        // Arrange
        let server = MockServer::start().await;
        let (mut handle, _sessions) = spawn_test_agent(&server).await;
        let journal_path = handle.journal_path().to_path_buf();

        // Act
        handle
            .commands
            .send(AgentCommand::Shutdown(ShutdownReason::UserQuit))
            .await
            .unwrap();
        let events = collect_until_events_close(&mut handle.events).await;
        let records = read_journal_records(&journal_path).await;
        let projection = project_journal(&records).unwrap();

        // Assert
        assert!(events.is_empty());
        assert!(projection.warnings.is_empty());
        assert!(matches!(
            records.last().map(|record| &record.entry),
            Some(JournalEntry::RunEnded(crate::journal::RunEnded {
                reason: RunEndReason::UserQuit,
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn cancelling_an_idle_session_exits_cleanly_without_a_turn_outcome() {
        // Arrange
        let server = MockServer::start().await;
        let (events_tx, mut events_rx) = mpsc::channel(64);
        let (_commands_tx, commands_rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let (session, _sessions) = test_session(
            &server,
            HostHandle {
                cancel: cancel.clone(),
                commands: commands_rx,
                events: EventSink::new(events_tx),
            },
            Vec::new(),
        )
        .await;
        let journal_path = session.journal.path().to_path_buf();
        let session_task = tokio::spawn(session.run());

        // Act
        cancel.cancel();
        let session_result = timeout(Duration::from_secs(1), session_task)
            .await
            .expect("idle session did not stop promptly")
            .expect("session task panicked");
        let events = collect_until_events_close(&mut events_rx).await;
        let records = read_journal_records(&journal_path).await;

        // Assert
        assert_eq!(session_result, Ok(()));
        assert!(
            events.is_empty(),
            "idle session emitted an unexpected turn outcome: {events:?}"
        );
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "idle session made a provider request"
        );
        assert!(matches!(
            records.last().map(|record| &record.entry),
            Some(JournalEntry::RunEnded(crate::journal::RunEnded {
                reason: RunEndReason::IdleCancelled,
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn dropping_events_stops_agent_even_when_command_sender_remains_alive() {
        // Arrange
        let server = MockServer::start().await;
        let (handle, _sessions) = spawn_test_agent(&server).await;
        let journal_path = handle.journal_path().to_path_buf();
        let AgentHandle {
            cancel: _cancel,
            commands,
            events,
            ..
        } = handle;

        // Act
        drop(events);

        // Assert
        timeout(Duration::from_secs(5), commands.closed())
            .await
            .expect("agent remained alive after its event receiver was dropped");
        let records = read_journal_records(&journal_path).await;
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "idle session made a provider request"
        );
        assert!(matches!(
            records.last().map(|record| &record.entry),
            Some(JournalEntry::RunEnded(crate::journal::RunEnded {
                reason: RunEndReason::FrontendDisconnected,
                ..
            }))
        ));
    }

    #[tokio::test]
    async fn frontend_disconnection_aborts_an_active_turn_before_ending_the_run() {
        // Arrange
        let server = MockServer::start().await;
        let request = Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(text_turn("too late").set_delay(Duration::from_secs(30)))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let (handle, _sessions) = spawn_test_agent(&server).await;
        let journal_path = handle.journal_path().to_path_buf();
        let AgentHandle {
            commands, events, ..
        } = handle;
        commands
            .send(AgentCommand::UserInput("Say hi".to_string()))
            .await
            .unwrap();
        timeout(Duration::from_secs(2), request.wait_until_satisfied())
            .await
            .expect("agent did not start its provider request promptly");

        // Act
        drop(events);
        timeout(Duration::from_secs(5), commands.closed())
            .await
            .expect("agent remained alive after its frontend disconnected");
        let records = read_journal_records(&journal_path).await;
        let projection = project_journal(&records).unwrap();

        // Assert
        assert!(projection.messages.is_empty());
        assert!(projection.warnings.is_empty());
        assert!(matches!(
            records.iter().rev().nth(1).map(|record| &record.entry),
            Some(JournalEntry::TurnAborted(crate::journal::TurnAborted {
                outcome: TurnAbortOutcome::Failed {
                    error: Some(ErrorDetail { category, message }),
                },
                ..
            })) if category == "frontend_disconnected"
                && message == "frontend disconnected during the active turn"
        ));
        assert!(matches!(
            records.last().map(|record| &record.entry),
            Some(JournalEntry::RunEnded(crate::journal::RunEnded {
                reason: RunEndReason::FrontendDisconnected,
                ..
            }))
        ));
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
        let (mut handle, _sessions) = spawn_test_agent(&server).await;

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
    async fn provider_round_limit_preserves_history_and_resets_for_the_next_turn() {
        // Arrange
        let file = temp_file_with(b"alpha");
        let file_path = file.path().to_str().unwrap();
        let server = MockServer::start().await;
        let mut turns = Vec::new();
        for index in 0..MAX_PROVIDER_ROUNDS_PER_TURN {
            let id = format!("read-{index}");
            turns.push(tool_call_turn(&[(
                &id,
                "read_file",
                json!({ "path": file_path }),
            )]));
        }
        turns.push(text_turn("Finished after continuing."));
        mount_turns(&server, turns).await;
        let (mut handle, _sessions) = spawn_test_agent(&server).await;

        // Act
        handle
            .commands
            .send(AgentCommand::UserInput("Inspect repeatedly.".to_string()))
            .await
            .unwrap();
        let paused_turn = collect_turn(&mut handle.events).await;
        handle
            .commands
            .send(AgentCommand::UserInput("Continue".to_string()))
            .await
            .unwrap();
        let continued_turn = collect_turn(&mut handle.events).await;
        drop(handle.commands);
        let shutdown_events = collect_until_events_close(&mut handle.events).await;
        let requests = server.received_requests().await.unwrap();
        let continued_request: Value = requests[MAX_PROVIDER_ROUNDS_PER_TURN].body_json().unwrap();
        let messages = continued_request["messages"].as_array().unwrap();

        // Assert
        assert!(matches!(
            paused_turn.last(),
            Some(AgentEvent::TurnComplete {
                outcome: TurnOutcome::Paused { reason }
            }) if reason
                == "turn paused after 24 provider rounds; send another message to continue with the existing context"
        ));
        assert!(
            !paused_turn
                .iter()
                .any(|event| matches!(event, AgentEvent::Error(_))),
            "a resumable pause must not be reported as an error: {paused_turn:?}"
        );
        assert!(matches!(
            continued_turn.as_slice(),
            [
                AgentEvent::TextDelta(text),
                AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn
                    }
                }
            ] if text == "Finished after continuing."
        ));
        assert!(
            shutdown_events.is_empty(),
            "clean shutdown emitted unexpected events: {shutdown_events:?}"
        );
        assert_eq!(requests.len(), MAX_PROVIDER_ROUNDS_PER_TURN + 1);
        assert_eq!(messages.len(), 2 * MAX_PROVIDER_ROUNDS_PER_TURN + 2);
        assert_eq!(
            messages.first().unwrap(),
            &json!({ "role": "user", "content": "Inspect repeatedly." })
        );
        assert_eq!(
            messages.last().unwrap(),
            &json!({ "role": "user", "content": "Continue" })
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
        let (mut handle, _sessions) = spawn_test_agent(&server).await;

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

        let (mut handle, _sessions) = spawn_test_agent(&server).await;

        // Act
        handle
            .commands
            .send(AgentCommand::UserInput("what".to_string()))
            .await
            .unwrap();

        let (respond_to, subject) = loop {
            let event = handle.events.recv().await.unwrap();

            if let AgentEvent::ApprovalRequest {
                respond_to: new_respond_to,
                subject,
                ..
            } = event
            {
                break (new_respond_to, subject);
            }
        };

        handle
            .commands
            .send(AgentCommand::UserInput("what3".to_string()))
            .await
            .unwrap();

        respond_to
            .send(ApprovalDecision::Grant(
                subject.grant(ApprovalLifetime::Invocation),
            ))
            .unwrap();

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
        let (mut handle, _sessions) = spawn_test_agent(&server).await;
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
    async fn a_grep_tool_turn_runs_without_approval_and_round_trips_matches() {
        // Arrange
        let file = temp_file_with(b"before\nneedle\nafter\n");
        let file_path = file.path().to_str().unwrap();
        let rendered_path = file.path().file_name().unwrap().to_str().unwrap();
        let server = MockServer::start().await;
        mount_turns(
            &server,
            vec![
                tool_call_turn(&[(
                    "grep-1",
                    "grep",
                    json!({ "path": file_path, "pattern": "needle" }),
                )]),
                text_turn("Found it."),
            ],
        )
        .await;

        // Act
        let events = run_agent("Find needle.", &server).await;

        // Assert
        assert!(matches!(
            events.as_slice(),
            [
                AgentEvent::ToolStarted { name: started, .. },
                AgentEvent::ToolFinished {
                    name: finished,
                    output,
                    is_error: false,
                },
                AgentEvent::TextDelta(text),
                AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn
                    }
                },
            ] if started == "grep"
                && finished == "grep"
                && output == &format!("{rendered_path}:\n  2: needle")
                && text == "Found it."
        ));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::ApprovalRequest { .. }))
        );

        let messages = nth_request_messages(&server, 1).await;
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "grep-1");
        assert_eq!(
            messages[2]["content"],
            format!("{rendered_path}:\n  2: needle")
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
    async fn malformed_tool_input_is_rejected_without_starting_the_tool() {
        // Arrange
        let server = MockServer::start().await;
        let malformed_input = "{\"path\": unclosed";
        mount_turns(
            &server,
            vec![
                malformed_tool_call_turn("call_bad", "read_file", malformed_input),
                text_turn("I need to retry that call."),
            ],
        )
        .await;

        // Act
        let events = run_agent("Read the file", &server).await;
        let messages = nth_request_messages(&server, 1).await;

        // Assert
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolRejected { name, error }
                if name == "read_file" && error.contains("arguments are not valid JSON")
        )));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolStarted { .. }))
        );
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            malformed_input
        );
        assert_eq!(
            messages[2],
            json!({
                "role": "tool",
                "tool_call_id": "call_bad",
                "content": format!(
                    "Error: invalid input for tool `read_file`: arguments are not valid JSON: {malformed_input}"
                )
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
                ] if msg == "api error (401): bad key"
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
        let (mut handle, _sessions) = spawn_test_agent(&server).await;

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
                ] if msg == "api error (401): bad key"
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
        let (mut handle, _sessions) = spawn_test_agent(&server).await;
        handle
            .commands
            .send(AgentCommand::UserInput("Change the file.".to_string()))
            .await
            .unwrap();

        // Act
        let (respond_to, subject) = loop {
            let event = handle.events.recv().await.unwrap();
            match event {
                AgentEvent::ApprovalRequest {
                    respond_to,
                    subject,
                    ..
                } => break (respond_to, subject),
                AgentEvent::ToolStarted { .. } => {
                    panic!("tool was reported as started before approval")
                }
                _ => {}
            }
        };
        respond_to
            .send(ApprovalDecision::Grant(
                subject.grant(ApprovalLifetime::Invocation),
            ))
            .unwrap();

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

        let (records, projection) = finish_and_project(handle).await;
        let requested = records
            .iter()
            .find_map(|record| match &record.entry {
                JournalEntry::ApprovalRequested(requested) => {
                    Some((record.sequence, requested.approval_id))
                }
                _ => None,
            })
            .expect("approval request was not journaled");
        let decided = records
            .iter()
            .find_map(|record| match &record.entry {
                JournalEntry::ApprovalDecided(decided) => Some((record.sequence, decided)),
                _ => None,
            })
            .expect("approval decision was not journaled");
        let started = records
            .iter()
            .find_map(|record| match &record.entry {
                JournalEntry::ToolStarted(started) => Some((record.sequence, started)),
                _ => None,
            })
            .expect("tool start was not journaled");

        assert_eq!(projection.messages.len(), 4);
        assert_eq!(decided.1.approval_id, requested.1);
        let expected_grant =
            ApprovalSubject::tool_call("write-1", "write_file").grant(ApprovalLifetime::Invocation);
        assert_eq!(
            decided.1.decision,
            JournalApprovalDecision::Grant {
                grant: expected_grant.clone(),
            }
        );
        assert_eq!(
            started.1.authorization,
            ToolAuthorization::Granted {
                approval_id: requested.1,
                grant: expected_grant,
            }
        );
        assert!(requested.0 < decided.0);
        assert!(decided.0 < started.0);
        assert!(records.iter().any(|record| matches!(
            &record.entry,
            JournalEntry::ToolCompleted(completed)
                if completed.tool_call_id == "write-1"
        )));
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

        let (mut handle, _sessions) = spawn_test_agent(&server).await;

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

        let (mut handle, _sessions) = spawn_test_agent(&server).await;

        handle
            .commands
            .send(AgentCommand::UserInput("Make the change".to_string()))
            .await
            .unwrap();

        // Act & Assert
        let write_one_event = handle.events.recv().await.unwrap();
        let AgentEvent::ApprovalRequest {
            respond_to,
            subject: write_one_subject,
            ..
        } = write_one_event
        else {
            panic!("expected first approval request");
        };
        assert_eq!("write-1", write_one_subject.tool_call_id());

        respond_to
            .send(ApprovalDecision::Deny {
                reason: "Changed my mind".to_string(),
            })
            .unwrap();
        let write_one_response = handle.events.recv().await.unwrap();

        let write_two_event = handle.events.recv().await.unwrap();
        let AgentEvent::ApprovalRequest {
            respond_to,
            subject: write_two_subject,
            ..
        } = write_two_event
        else {
            panic!("expected second approval request");
        };
        assert_eq!("write-2", write_two_subject.tool_call_id());

        respond_to
            .send(ApprovalDecision::Grant(
                write_two_subject.grant(ApprovalLifetime::Invocation),
            ))
            .unwrap();

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

        let (mut handle, _sessions) = spawn_test_agent(&server).await;

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
        let (mut handle, _sessions) = spawn_test_agent(&server).await;
        let journal_path = handle.journal_path().to_path_buf();

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
        let records = read_journal_records(&journal_path).await;

        // Assert
        assert!(
            matches!(
                &events[..],
                [
                    AgentEvent::Error(msg),
                    AgentEvent::TurnComplete {
                        outcome: TurnOutcome::Cancelled
                    }
                ] if msg == "cancelled"
            ),
            "expected an Error followed by a cancelled TurnComplete, got {events:?}"
        );
        assert!(matches!(
            records.iter().rev().nth(1).map(|record| &record.entry),
            Some(JournalEntry::TurnAborted(crate::journal::TurnAborted {
                outcome: TurnAbortOutcome::Cancelled,
                ..
            }))
        ));
        assert!(matches!(
            records.last().map(|record| &record.entry),
            Some(JournalEntry::RunEnded(crate::journal::RunEnded {
                reason: RunEndReason::ActiveTurnCancelled,
                ..
            }))
        ));
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
    async fn session_cancellation_completes_the_turn_once_without_tool_finished() {
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

        let (session, _sessions) = test_session(
            &server,
            host_handle,
            vec![Box::new(BlockingTestTool {
                started: Arc::clone(&started),
            })],
        )
        .await;

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
        let (mut journal, _sessions) = test_run_journal().await;

        // Act
        let result = execute_invocation(
            &host_handle,
            &mut journal,
            AuthorizedToolCall {
                authorization: ApprovalAuthorization::NotRequired,
                id: mock_id.as_str(),
                input: &json!({}),
                name: "test",
                turn_id: TurnId::generate(),
            },
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
        let (mut journal, _sessions) = test_run_journal().await;

        // Act
        let result = execute_invocation(
            &host_handle,
            &mut journal,
            AuthorizedToolCall {
                authorization: ApprovalAuthorization::NotRequired,
                id: "call-1",
                input: &json!({}),
                name: "test",
                turn_id: TurnId::generate(),
            },
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
        let (mut journal, _sessions) = test_run_journal().await;
        let turn_id = TurnId::generate();

        let execution = tokio::spawn(async move {
            execute_invocation(
                &host_handle,
                &mut journal,
                AuthorizedToolCall {
                    authorization: ApprovalAuthorization::NotRequired,
                    id: "call-1",
                    input: &json!({}),
                    name: "test",
                    turn_id,
                },
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
    async fn session_cancellation_returns_cancelled_without_tool_finished() {
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
        let (mut journal, _sessions) = test_run_journal().await;
        let turn_id = TurnId::generate();

        let execution = tokio::spawn(async move {
            execute_invocation(
                &host_handle,
                &mut journal,
                AuthorizedToolCall {
                    authorization: ApprovalAuthorization::NotRequired,
                    id: "call-1",
                    input: &json!({}),
                    name: "test",
                    turn_id,
                },
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
