use super::{
    ApprovalId, JournalApprovalDecision, JournalEntry, JournalRecord, ProviderRoundId, RunId,
    SessionId, ToolAuthorization, TurnAbortOutcome, TurnCommitOutcome, TurnId,
};
use crate::{ContentBlock, Message, Role, StopReason};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionProjection {
    pub instructions: String,
    pub messages: Vec<Message>,
    pub session_id: SessionId,
    pub warnings: Vec<ProjectionWarning>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectionWarning {
    UnterminatedRun { run_id: RunId },
    UnterminatedTurn { turn_id: TurnId },
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("invalid journal lifecycle at sequence {sequence}: {detail}")]
pub struct ProjectionError {
    pub detail: String,
    pub sequence: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ApprovalDecision {
    AllowForRun,
    AllowOnce,
    Deny,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolState {
    Pending,
    Started,
    Terminal,
}

struct ToolCallState {
    name: String,
    state: ToolState,
}

struct ApprovalState {
    decision: Option<ApprovalDecision>,
    tool_call_id: String,
    tool_name: String,
}

struct TurnState {
    approvals: HashMap<ApprovalId, ApprovalState>,
    awaiting_assistant: Option<StopReason>,
    id: TurnId,
    last_assistant_stop: Option<StopReason>,
    messages: Vec<Message>,
    provider_round: Option<ProviderRoundId>,
    provider_rounds: HashSet<ProviderRoundId>,
    tool_calls: HashMap<String, ToolCallState>,
}

struct RunState {
    id: RunId,
    run_approvals: HashMap<ApprovalId, String>,
    turn: Option<TurnState>,
}

pub fn project_journal(records: &[JournalRecord]) -> Result<SessionProjection, ProjectionError> {
    let Some(first) = records.first() else {
        return Err(invalid(0, "journal is empty"));
    };
    let JournalEntry::SessionStarted(started) = &first.entry else {
        return Err(invalid(
            first.sequence,
            "first record is not session_started",
        ));
    };

    let mut active_run: Option<RunState> = None;
    let mut messages = Vec::new();

    for record in &records[1..] {
        match &record.entry {
            JournalEntry::SessionStarted(_) => {
                return Err(invalid(
                    record.sequence,
                    "session_started appears more than once",
                ));
            }
            JournalEntry::RunStarted(started) => {
                if active_run.is_some() {
                    return Err(invalid(
                        record.sequence,
                        "run_started appears while another run is active",
                    ));
                }
                active_run = Some(RunState {
                    id: started.run_id,
                    run_approvals: HashMap::new(),
                    turn: None,
                });
            }
            JournalEntry::TurnStarted(started) => {
                let run = require_run(&mut active_run, started.run_id, record.sequence)?;
                if run.turn.is_some() {
                    return Err(invalid(
                        record.sequence,
                        "turn_started appears while another turn is active",
                    ));
                }
                run.turn = Some(TurnState {
                    approvals: HashMap::new(),
                    awaiting_assistant: None,
                    id: started.turn_id,
                    last_assistant_stop: None,
                    messages: Vec::new(),
                    provider_round: None,
                    provider_rounds: HashSet::new(),
                    tool_calls: HashMap::new(),
                });
            }
            JournalEntry::MessageAdded(added) => {
                let turn = require_turn(
                    &mut active_run,
                    added.run_id,
                    added.turn_id,
                    record.sequence,
                )?;
                add_message(turn, &added.message, record.sequence)?;
            }
            JournalEntry::ProviderRoundStarted(started) => {
                let turn = require_turn(
                    &mut active_run,
                    started.run_id,
                    started.turn_id,
                    record.sequence,
                )?;
                if turn.provider_round.is_some() || turn.awaiting_assistant.is_some() {
                    return Err(invalid(
                        record.sequence,
                        "provider round started before the previous round was resolved",
                    ));
                }
                if !matches!(
                    turn.messages.last(),
                    Some(Message {
                        role: Role::User,
                        ..
                    })
                ) {
                    return Err(invalid(
                        record.sequence,
                        "provider round started without a preceding user message",
                    ));
                }
                if !turn.provider_rounds.insert(started.provider_round_id) {
                    return Err(invalid(
                        record.sequence,
                        "provider round ID is reused within a turn",
                    ));
                }
                turn.provider_round = Some(started.provider_round_id);
            }
            JournalEntry::ProviderRoundCompleted(completed) => {
                let turn = require_turn(
                    &mut active_run,
                    completed.run_id,
                    completed.turn_id,
                    record.sequence,
                )?;
                finish_provider_round(turn, completed.provider_round_id, record.sequence)?;
                turn.awaiting_assistant = Some(completed.stop_reason.clone());
            }
            JournalEntry::ProviderRoundFailed(failed) => {
                let turn = require_turn(
                    &mut active_run,
                    failed.run_id,
                    failed.turn_id,
                    record.sequence,
                )?;
                finish_provider_round(turn, failed.provider_round_id, record.sequence)?;
            }
            JournalEntry::ProviderRoundCancelled(cancelled) => {
                let turn = require_turn(
                    &mut active_run,
                    cancelled.run_id,
                    cancelled.turn_id,
                    record.sequence,
                )?;
                finish_provider_round(turn, cancelled.provider_round_id, record.sequence)?;
            }
            JournalEntry::ApprovalRequested(requested) => {
                let turn =
                    require_active_turn(&mut active_run, requested.turn_id, record.sequence)?;
                let Some(tool) = turn.tool_calls.get(&requested.tool_call_id) else {
                    return Err(invalid(
                        record.sequence,
                        "approval refers to an unknown tool call",
                    ));
                };
                if tool.name != requested.tool_name {
                    return Err(invalid(
                        record.sequence,
                        "approval tool name does not match its tool call",
                    ));
                }
                if turn
                    .approvals
                    .insert(
                        requested.approval_id,
                        ApprovalState {
                            decision: None,
                            tool_call_id: requested.tool_call_id.clone(),
                            tool_name: requested.tool_name.clone(),
                        },
                    )
                    .is_some()
                {
                    return Err(invalid(record.sequence, "approval ID is reused"));
                }
            }
            JournalEntry::ApprovalDecided(decided) => {
                let run = active_run_mut(&mut active_run, record.sequence)?;
                let run_approval = {
                    let Some(turn) = run.turn.as_mut() else {
                        return Err(invalid(record.sequence, "record has no active turn"));
                    };
                    let Some(current) = turn.approvals.get_mut(&decided.approval_id) else {
                        return Err(invalid(
                            record.sequence,
                            "approval decision has no matching request",
                        ));
                    };
                    if current.decision.is_some() {
                        return Err(invalid(
                            record.sequence,
                            "approval is decided more than once",
                        ));
                    }
                    let decision = match decided.decision {
                        JournalApprovalDecision::AllowForRun => ApprovalDecision::AllowForRun,
                        JournalApprovalDecision::AllowOnce => ApprovalDecision::AllowOnce,
                        JournalApprovalDecision::Deny { .. } => ApprovalDecision::Deny,
                    };
                    current.decision = Some(decision);
                    (decision == ApprovalDecision::AllowForRun).then(|| current.tool_name.clone())
                };
                if let Some(tool_name) = run_approval {
                    run.run_approvals.insert(decided.approval_id, tool_name);
                }
            }
            JournalEntry::ToolStarted(started) => {
                let run = active_run_mut(&mut active_run, record.sequence)?;
                let Some(turn) = run.turn.as_mut() else {
                    return Err(invalid(record.sequence, "record has no active turn"));
                };
                if turn.id != started.turn_id {
                    return Err(invalid(
                        record.sequence,
                        "record does not belong to the active turn",
                    ));
                }
                authorize_tool_start(
                    &run.run_approvals,
                    turn,
                    &started.authorization,
                    &started.tool_call_id,
                    &started.tool_name,
                    record.sequence,
                )?;
                let tool = require_tool_call(
                    turn,
                    &started.tool_call_id,
                    &started.tool_name,
                    record.sequence,
                )?;
                if tool.state != ToolState::Pending {
                    return Err(invalid(
                        record.sequence,
                        "tool call is started more than once",
                    ));
                }
                tool.state = ToolState::Started;
            }
            JournalEntry::ToolCompleted(completed) => {
                finish_tool(
                    &mut active_run,
                    completed.turn_id,
                    &completed.tool_call_id,
                    record.sequence,
                )?;
            }
            JournalEntry::ToolFailed(failed) => {
                finish_tool(
                    &mut active_run,
                    failed.turn_id,
                    &failed.tool_call_id,
                    record.sequence,
                )?;
            }
            JournalEntry::ToolCancelled(cancelled) => {
                finish_tool(
                    &mut active_run,
                    cancelled.turn_id,
                    &cancelled.tool_call_id,
                    record.sequence,
                )?;
            }
            JournalEntry::ToolRejected(rejected) => {
                let turn = require_active_turn(&mut active_run, rejected.turn_id, record.sequence)?;
                let tool = require_tool_call(
                    turn,
                    &rejected.tool_call_id,
                    &rejected.tool_name,
                    record.sequence,
                )?;
                if tool.state != ToolState::Pending {
                    return Err(invalid(
                        record.sequence,
                        "rejected tool call was already started or resolved",
                    ));
                }
                tool.state = ToolState::Terminal;
            }
            JournalEntry::TurnCommitted(committed) => {
                let run = active_run_mut(&mut active_run, record.sequence)?;
                let Some(turn) = run.turn.take() else {
                    return Err(invalid(record.sequence, "turn commit has no active turn"));
                };
                if turn.id != committed.turn_id {
                    return Err(invalid(
                        record.sequence,
                        "turn commit does not match the active turn",
                    ));
                }
                validate_commit(&turn, &committed.outcome, record.sequence)?;
                messages.extend(turn.messages);
            }
            JournalEntry::TurnAborted(aborted) => {
                let run = active_run_mut(&mut active_run, record.sequence)?;
                let Some(turn) = run.turn.take() else {
                    return Err(invalid(record.sequence, "turn abort has no active turn"));
                };
                if turn.id != aborted.turn_id {
                    return Err(invalid(
                        record.sequence,
                        "turn abort does not match the active turn",
                    ));
                }
                match &aborted.outcome {
                    TurnAbortOutcome::Failed { .. } | TurnAbortOutcome::Cancelled => {}
                }
            }
            JournalEntry::RunEnded(ended) => {
                let Some(run) = active_run.take() else {
                    return Err(invalid(record.sequence, "run end has no active run"));
                };
                if run.id != ended.run_id {
                    return Err(invalid(
                        record.sequence,
                        "run end does not match the active run",
                    ));
                }
                if run.turn.is_some() {
                    return Err(invalid(
                        record.sequence,
                        "run ended while a turn is still active",
                    ));
                }
            }
        }
    }

    let mut warnings = Vec::new();
    if let Some(run) = active_run {
        if let Some(turn) = run.turn {
            warnings.push(ProjectionWarning::UnterminatedTurn { turn_id: turn.id });
        }
        warnings.push(ProjectionWarning::UnterminatedRun { run_id: run.id });
    }

    Ok(SessionProjection {
        instructions: started.instructions.clone(),
        messages,
        session_id: first.session_id,
        warnings,
    })
}

fn add_message(
    turn: &mut TurnState,
    message: &Message,
    sequence: u64,
) -> Result<(), ProjectionError> {
    match message.role {
        Role::User if turn.messages.is_empty() => {
            if message
                .content
                .iter()
                .any(|block| !matches!(block, ContentBlock::Text { .. }))
            {
                return Err(invalid(
                    sequence,
                    "the first turn message must contain only user text",
                ));
            }
        }
        Role::User => add_tool_results(turn, message, sequence)?,
        Role::Assistant => add_assistant_message(turn, message, sequence)?,
    }

    turn.messages.push(message.clone());
    Ok(())
}

fn add_assistant_message(
    turn: &mut TurnState,
    message: &Message,
    sequence: u64,
) -> Result<(), ProjectionError> {
    let Some(stop_reason) = turn.awaiting_assistant.take() else {
        return Err(invalid(
            sequence,
            "assistant message has no completed provider round",
        ));
    };
    if !turn.tool_calls.is_empty() {
        return Err(invalid(
            sequence,
            "assistant message arrived before prior tool results",
        ));
    }

    let mut has_tool_use = false;
    for block in &message.content {
        match block {
            ContentBlock::Text { .. } => {}
            ContentBlock::ToolUse { id, name, .. } => {
                has_tool_use = true;
                if turn
                    .tool_calls
                    .insert(
                        id.clone(),
                        ToolCallState {
                            name: name.clone(),
                            state: ToolState::Pending,
                        },
                    )
                    .is_some()
                {
                    return Err(invalid(
                        sequence,
                        "assistant message repeats a tool call ID",
                    ));
                }
            }
            ContentBlock::ToolResult(_) => {
                return Err(invalid(
                    sequence,
                    "assistant message contains a tool result",
                ));
            }
        }
    }

    if has_tool_use != (stop_reason == StopReason::ToolUse) {
        return Err(invalid(
            sequence,
            "assistant tool calls do not agree with the provider stop reason",
        ));
    }
    turn.last_assistant_stop = Some(stop_reason);
    Ok(())
}

fn add_tool_results(
    turn: &mut TurnState,
    message: &Message,
    sequence: u64,
) -> Result<(), ProjectionError> {
    if turn.tool_calls.is_empty() {
        return Err(invalid(
            sequence,
            "additional user message has no pending tool calls",
        ));
    }

    let mut result_ids = HashSet::new();
    for block in &message.content {
        let ContentBlock::ToolResult(result) = block else {
            return Err(invalid(
                sequence,
                "tool-result message contains non-result content",
            ));
        };
        let Some(tool) = turn.tool_calls.get(&result.tool_use_id) else {
            return Err(invalid(
                sequence,
                "tool result refers to an unknown tool call",
            ));
        };
        if tool.state != ToolState::Terminal {
            return Err(invalid(
                sequence,
                "tool result was recorded before its terminal lifecycle record",
            ));
        }
        if !result_ids.insert(result.tool_use_id.as_str()) {
            return Err(invalid(sequence, "tool result ID is repeated"));
        }
    }

    if result_ids.len() != turn.tool_calls.len() {
        return Err(invalid(
            sequence,
            "tool-result message does not resolve every tool call",
        ));
    }
    turn.tool_calls.clear();
    Ok(())
}

fn authorize_tool_start(
    run_approvals: &HashMap<ApprovalId, String>,
    turn: &TurnState,
    authorization: &ToolAuthorization,
    tool_call_id: &str,
    tool_name: &str,
    sequence: u64,
) -> Result<(), ProjectionError> {
    match authorization {
        ToolAuthorization::ApprovedForRun { approval_id } => {
            if run_approvals.get(approval_id).map(String::as_str) != Some(tool_name) {
                return Err(invalid(
                    sequence,
                    "run approval does not authorize this tool",
                ));
            }
        }
        ToolAuthorization::ApprovedOnce { approval_id } => {
            let Some(approval) = turn.approvals.get(approval_id) else {
                return Err(invalid(
                    sequence,
                    "tool authorization has no matching approval decision",
                ));
            };
            if approval.decision != Some(ApprovalDecision::AllowOnce)
                || approval.tool_call_id != tool_call_id
                || approval.tool_name != tool_name
            {
                return Err(invalid(
                    sequence,
                    "one-time approval does not authorize this tool call",
                ));
            }
        }
        ToolAuthorization::NotRequired => {}
    }
    Ok(())
}

fn finish_provider_round(
    turn: &mut TurnState,
    provider_round_id: ProviderRoundId,
    sequence: u64,
) -> Result<(), ProjectionError> {
    if turn.provider_round != Some(provider_round_id) {
        return Err(invalid(
            sequence,
            "provider terminal record does not match the active round",
        ));
    }
    turn.provider_round = None;
    Ok(())
}

fn finish_tool(
    active_run: &mut Option<RunState>,
    turn_id: TurnId,
    tool_call_id: &str,
    sequence: u64,
) -> Result<(), ProjectionError> {
    let turn = require_active_turn(active_run, turn_id, sequence)?;
    let Some(tool) = turn.tool_calls.get_mut(tool_call_id) else {
        return Err(invalid(
            sequence,
            "tool terminal record refers to an unknown tool call",
        ));
    };
    if tool.state != ToolState::Started {
        return Err(invalid(
            sequence,
            "tool terminal record has no matching tool start",
        ));
    }
    tool.state = ToolState::Terminal;
    Ok(())
}

fn validate_commit(
    turn: &TurnState,
    outcome: &TurnCommitOutcome,
    sequence: u64,
) -> Result<(), ProjectionError> {
    if turn.provider_round.is_some() || turn.awaiting_assistant.is_some() {
        return Err(invalid(
            sequence,
            "turn committed with an unresolved provider round",
        ));
    }
    if !turn.tool_calls.is_empty() {
        return Err(invalid(
            sequence,
            "turn committed with unresolved tool calls",
        ));
    }
    if turn
        .approvals
        .values()
        .any(|approval| approval.decision.is_none())
    {
        return Err(invalid(
            sequence,
            "turn committed with an undecided approval",
        ));
    }

    if let TurnCommitOutcome::Completed { stop_reason } = outcome
        && turn.last_assistant_stop.as_ref() != Some(stop_reason)
    {
        return Err(invalid(
            sequence,
            "completed turn outcome does not match its assistant stop reason",
        ));
    }
    Ok(())
}

fn require_tool_call<'a>(
    turn: &'a mut TurnState,
    tool_call_id: &str,
    tool_name: &str,
    sequence: u64,
) -> Result<&'a mut ToolCallState, ProjectionError> {
    let Some(tool) = turn.tool_calls.get_mut(tool_call_id) else {
        return Err(invalid(sequence, "record refers to an unknown tool call"));
    };
    if tool.name != tool_name {
        return Err(invalid(sequence, "tool name does not match its tool call"));
    }
    Ok(tool)
}

fn require_run(
    active_run: &mut Option<RunState>,
    run_id: RunId,
    sequence: u64,
) -> Result<&mut RunState, ProjectionError> {
    let run = active_run_mut(active_run, sequence)?;
    if run.id != run_id {
        return Err(invalid(
            sequence,
            "record does not belong to the active run",
        ));
    }
    Ok(run)
}

fn require_turn(
    active_run: &mut Option<RunState>,
    run_id: RunId,
    turn_id: TurnId,
    sequence: u64,
) -> Result<&mut TurnState, ProjectionError> {
    let run = require_run(active_run, run_id, sequence)?;
    let Some(turn) = run.turn.as_mut() else {
        return Err(invalid(sequence, "record has no active turn"));
    };
    if turn.id != turn_id {
        return Err(invalid(
            sequence,
            "record does not belong to the active turn",
        ));
    }
    Ok(turn)
}

fn require_active_turn(
    active_run: &mut Option<RunState>,
    turn_id: TurnId,
    sequence: u64,
) -> Result<&mut TurnState, ProjectionError> {
    let turn = active_turn_mut(active_run, sequence)?;
    if turn.id != turn_id {
        return Err(invalid(
            sequence,
            "record does not belong to the active turn",
        ));
    }
    Ok(turn)
}

fn active_run_mut(
    active_run: &mut Option<RunState>,
    sequence: u64,
) -> Result<&mut RunState, ProjectionError> {
    active_run
        .as_mut()
        .ok_or_else(|| invalid(sequence, "record has no active run"))
}

fn active_turn_mut(
    active_run: &mut Option<RunState>,
    sequence: u64,
) -> Result<&mut TurnState, ProjectionError> {
    active_run_mut(active_run, sequence)?
        .turn
        .as_mut()
        .ok_or_else(|| invalid(sequence, "record has no active turn"))
}

fn invalid(sequence: u64, detail: impl Into<String>) -> ProjectionError {
    ProjectionError {
        detail: detail.into(),
        sequence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::{
        ErrorDetail, MessageAdded, ProviderRoundCompleted, ProviderRoundFailed,
        ProviderRoundStarted, RunEndReason, RunEnded, RunStarted, SessionStarted, ToolCompleted,
        ToolStarted, TurnAborted, TurnCommitted, TurnStarted,
    };
    use crate::{ProviderAdapter, ProviderDescriptor, ToolInput, ToolResultData};

    fn session_id() -> SessionId {
        "sess_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap()
    }

    fn run_id() -> RunId {
        "run_01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap()
    }

    fn turn_id() -> TurnId {
        "turn_01ARZ3NDEKTSV4RRFFQ69G5FAX".parse().unwrap()
    }

    fn round_id() -> ProviderRoundId {
        "round_01ARZ3NDEKTSV4RRFFQ69G5FAY".parse().unwrap()
    }

    fn records(entries: Vec<JournalEntry>) -> Vec<JournalRecord> {
        entries
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                JournalRecord::new(
                    (index + 1) as u64,
                    "2026-07-27T12:00:00.123Z".parse().unwrap(),
                    session_id(),
                    entry,
                )
            })
            .collect()
    }

    fn session_started() -> JournalEntry {
        JournalEntry::SessionStarted(SessionStarted {
            cane_version: "0.1.0".to_string(),
            instructions: "Be helpful.".to_string(),
            workspace: "/workspace".to_string(),
        })
    }

    fn run_started() -> JournalEntry {
        JournalEntry::RunStarted(RunStarted {
            git: None,
            max_output_tokens: 32_000,
            model: "test-model".to_string(),
            provider: provider(),
            run_id: run_id(),
            tool_catalog: Vec::new(),
        })
    }

    fn turn_started() -> JournalEntry {
        JournalEntry::TurnStarted(TurnStarted {
            run_id: run_id(),
            turn_id: turn_id(),
        })
    }

    fn user_text(text: &str) -> Message {
        Message {
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            role: Role::User,
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            role: Role::Assistant,
        }
    }

    fn message_added(message: Message) -> JournalEntry {
        JournalEntry::MessageAdded(MessageAdded {
            message,
            run_id: run_id(),
            turn_id: turn_id(),
        })
    }

    fn provider_started() -> JournalEntry {
        JournalEntry::ProviderRoundStarted(ProviderRoundStarted {
            model: "test-model".to_string(),
            provider: provider(),
            provider_round_id: round_id(),
            run_id: run_id(),
            turn_id: turn_id(),
        })
    }

    fn provider() -> ProviderDescriptor {
        ProviderDescriptor {
            adapter: ProviderAdapter::OpenAiCompatible,
            endpoint: "https://example.test/v1/chat/completions".to_string(),
        }
    }

    fn provider_completed(stop_reason: StopReason) -> JournalEntry {
        JournalEntry::ProviderRoundCompleted(ProviderRoundCompleted {
            latency_ms: 10,
            provider_cost: None,
            provider_round_id: round_id(),
            request_id: None,
            run_id: run_id(),
            stop_reason,
            turn_id: turn_id(),
            usage: None,
        })
    }

    fn run_ended() -> JournalEntry {
        JournalEntry::RunEnded(RunEnded {
            reason: RunEndReason::UserQuit,
            run_id: run_id(),
        })
    }

    #[test]
    fn a_committed_turn_projects_its_messages_and_session_instructions() {
        // Arrange
        let user = user_text("Hello");
        let assistant = assistant_text("Hi");
        let journal = records(vec![
            session_started(),
            run_started(),
            turn_started(),
            message_added(user.clone()),
            provider_started(),
            provider_completed(StopReason::EndTurn),
            message_added(assistant.clone()),
            JournalEntry::TurnCommitted(TurnCommitted {
                outcome: TurnCommitOutcome::Completed {
                    stop_reason: StopReason::EndTurn,
                },
                turn_id: turn_id(),
            }),
            run_ended(),
        ]);

        // Act
        let projection = project_journal(&journal).unwrap();

        // Assert
        assert_eq!(projection.instructions, "Be helpful.");
        assert_eq!(projection.messages, vec![user, assistant]);
        assert_eq!(projection.session_id, session_id());
        assert!(projection.warnings.is_empty());
    }

    #[test]
    fn a_paused_tool_turn_projects_only_after_its_commit() {
        // Arrange
        let user = user_text("Read Cargo.toml");
        let assistant = Message {
            content: vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                input: ToolInput::Valid(serde_json::json!({ "path": "Cargo.toml" })),
                name: "read_file".to_string(),
            }],
            role: Role::Assistant,
        };
        let result = Message {
            content: vec![ContentBlock::ToolResult(ToolResultData {
                content: "contents".to_string(),
                is_error: false,
                tool_use_id: "call-1".to_string(),
            })],
            role: Role::User,
        };
        let journal = records(vec![
            session_started(),
            run_started(),
            turn_started(),
            message_added(user.clone()),
            provider_started(),
            provider_completed(StopReason::ToolUse),
            message_added(assistant.clone()),
            JournalEntry::ToolStarted(ToolStarted {
                authorization: ToolAuthorization::NotRequired,
                tool_call_id: "call-1".to_string(),
                tool_name: "read_file".to_string(),
                turn_id: turn_id(),
            }),
            JournalEntry::ToolCompleted(ToolCompleted {
                duration_ms: 5,
                tool_call_id: "call-1".to_string(),
                turn_id: turn_id(),
            }),
            message_added(result.clone()),
            JournalEntry::TurnCommitted(TurnCommitted {
                outcome: TurnCommitOutcome::Paused {
                    reason: "provider round limit".to_string(),
                },
                turn_id: turn_id(),
            }),
            run_ended(),
        ]);

        // Act
        let projection = project_journal(&journal).unwrap();

        // Assert
        assert_eq!(projection.messages, vec![user, assistant, result]);
        assert!(projection.warnings.is_empty());
    }

    #[test]
    fn aborted_and_unterminated_turns_do_not_project_their_messages() {
        // Arrange
        let aborted = records(vec![
            session_started(),
            run_started(),
            turn_started(),
            message_added(user_text("Do work")),
            provider_started(),
            JournalEntry::ProviderRoundFailed(ProviderRoundFailed {
                error: ErrorDetail {
                    category: "provider".to_string(),
                    message: "failed".to_string(),
                },
                latency_ms: 10,
                provider_round_id: round_id(),
                request_id: None,
                run_id: run_id(),
                turn_id: turn_id(),
            }),
            JournalEntry::TurnAborted(TurnAborted {
                outcome: TurnAbortOutcome::Failed { error: None },
                turn_id: turn_id(),
            }),
            run_ended(),
        ]);
        let unterminated = records(vec![
            session_started(),
            run_started(),
            turn_started(),
            message_added(user_text("Do unfinished work")),
        ]);

        // Act
        let aborted_projection = project_journal(&aborted).unwrap();
        let unterminated_projection = project_journal(&unterminated).unwrap();

        // Assert
        assert!(aborted_projection.messages.is_empty());
        assert!(aborted_projection.warnings.is_empty());
        assert!(unterminated_projection.messages.is_empty());
        assert_eq!(
            unterminated_projection.warnings,
            vec![
                ProjectionWarning::UnterminatedTurn { turn_id: turn_id() },
                ProjectionWarning::UnterminatedRun { run_id: run_id() },
            ]
        );
    }

    #[test]
    fn malformed_lifecycle_records_are_rejected() {
        // Arrange
        let message_without_turn = records(vec![
            session_started(),
            run_started(),
            message_added(user_text("Hello")),
        ]);
        let mismatched_provider_terminal = records(vec![
            session_started(),
            run_started(),
            turn_started(),
            message_added(user_text("Hello")),
            provider_started(),
            JournalEntry::ProviderRoundCompleted(ProviderRoundCompleted {
                latency_ms: 10,
                provider_cost: None,
                provider_round_id: "round_01ARZ3NDEKTSV4RRFFQ69G5FAZ".parse().unwrap(),
                request_id: None,
                run_id: run_id(),
                stop_reason: StopReason::EndTurn,
                turn_id: turn_id(),
                usage: None,
            }),
        ]);

        // Act
        let missing_turn_error = project_journal(&message_without_turn).unwrap_err();
        let provider_error = project_journal(&mismatched_provider_terminal).unwrap_err();

        // Assert
        assert_eq!(missing_turn_error.sequence, 3);
        assert!(missing_turn_error.detail.contains("active turn"));
        assert_eq!(provider_error.sequence, 6);
        assert!(provider_error.detail.contains("active round"));
    }

    #[test]
    fn run_approvals_cross_turns_but_one_time_approvals_remain_call_specific() {
        // Arrange
        let approval_id: ApprovalId = "appr_01ARZ3NDEKTSV4RRFFQ69G5FAZ".parse().unwrap();
        let run_approvals = HashMap::from([(approval_id, "write_file".to_string())]);
        let turn = TurnState {
            approvals: HashMap::from([(
                approval_id,
                ApprovalState {
                    decision: Some(ApprovalDecision::AllowOnce),
                    tool_call_id: "call-1".to_string(),
                    tool_name: "write_file".to_string(),
                },
            )]),
            awaiting_assistant: None,
            id: turn_id(),
            last_assistant_stop: None,
            messages: Vec::new(),
            provider_round: None,
            provider_rounds: HashSet::new(),
            tool_calls: HashMap::new(),
        };

        // Act
        let run_approval_result = authorize_tool_start(
            &run_approvals,
            &turn,
            &ToolAuthorization::ApprovedForRun { approval_id },
            "call-2",
            "write_file",
            1,
        );
        let reused_once_result = authorize_tool_start(
            &run_approvals,
            &turn,
            &ToolAuthorization::ApprovedOnce { approval_id },
            "call-2",
            "write_file",
            2,
        );

        // Assert
        assert!(run_approval_result.is_ok());
        assert!(reused_once_result.is_err());
    }
}
