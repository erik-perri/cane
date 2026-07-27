use crate::{Message, ModelUsage, ReportedCost, StopReason, ToolDefinition};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

pub const JOURNAL_SCHEMA_VERSION: u32 = 1;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

string_id!(SessionId);
string_id!(RunId);
string_id!(TurnId);
string_id!(ProviderRoundId);
string_id!(ApprovalId);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct JournalRecord {
    #[serde(flatten)]
    pub entry: JournalEntry,
    pub recorded_at: String,
    pub schema_version: u32,
    pub sequence: u64,
    pub session_id: SessionId,
}

impl JournalRecord {
    pub fn new(
        sequence: u64,
        recorded_at: impl Into<String>,
        session_id: SessionId,
        entry: JournalEntry,
    ) -> Self {
        Self {
            entry,
            recorded_at: recorded_at.into(),
            schema_version: JOURNAL_SCHEMA_VERSION,
            sequence,
            session_id,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum JournalEntry {
    SessionStarted(SessionStarted),
    RunStarted(RunStarted),
    TurnStarted(TurnStarted),
    MessageAdded(MessageAdded),
    ProviderRoundStarted(ProviderRoundStarted),
    ProviderRoundCompleted(ProviderRoundCompleted),
    ProviderRoundFailed(ProviderRoundFailed),
    ProviderRoundCancelled(ProviderRoundCancelled),
    ApprovalRequested(ApprovalRequested),
    ApprovalDecided(ApprovalDecided),
    ToolStarted(ToolStarted),
    ToolCompleted(ToolCompleted),
    ToolFailed(ToolFailed),
    ToolCancelled(ToolCancelled),
    ToolRejected(ToolRejected),
    TurnCommitted(TurnCommitted),
    TurnAborted(TurnAborted),
    RunEnded(RunEnded),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SessionStarted {
    pub cane_version: String,
    pub instructions: String,
    pub workspace: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RunStarted {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git: Option<GitContext>,
    pub max_output_tokens: u32,
    pub model: String,
    pub provider: String,
    pub run_id: RunId,
    pub tool_catalog: Vec<ToolDefinition>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GitContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dirty: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository_root: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TurnStarted {
    pub run_id: RunId,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MessageAdded {
    pub message: Message,
    pub run_id: RunId,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRoundStarted {
    pub model: String,
    pub provider: String,
    pub provider_round_id: ProviderRoundId,
    pub run_id: RunId,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRoundCompleted {
    pub latency_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_cost: Option<ReportedCost>,
    pub provider_round_id: ProviderRoundId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub run_id: RunId,
    pub stop_reason: StopReason,
    pub turn_id: TurnId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ModelUsage>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRoundFailed {
    pub error: ErrorDetail,
    pub latency_ms: u64,
    pub provider_round_id: ProviderRoundId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub run_id: RunId,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderRoundCancelled {
    pub latency_ms: u64,
    pub provider_round_id: ProviderRoundId,
    pub run_id: RunId,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRequested {
    pub approval_id: ApprovalId,
    pub tool_call_id: String,
    pub tool_name: String,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ApprovalDecided {
    pub approval_id: ApprovalId,
    pub decision: JournalApprovalDecision,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JournalApprovalDecision {
    AllowOnce,
    AllowRun,
    Deny { reason: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolStarted {
    pub authorization: ToolAuthorization,
    pub tool_call_id: String,
    pub tool_name: String,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolAuthorization {
    NotRequired,
    AllowOnce { approval_id: ApprovalId },
    RunGrant { approval_id: ApprovalId },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolCompleted {
    pub duration_ms: u64,
    pub tool_call_id: String,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolFailed {
    pub duration_ms: u64,
    pub error_category: String,
    pub tool_call_id: String,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolCancelled {
    pub duration_ms: u64,
    pub tool_call_id: String,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolRejected {
    pub error_category: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TurnCommitted {
    pub outcome: TurnCommitOutcome,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TurnCommitOutcome {
    Completed { stop_reason: StopReason },
    Paused { reason: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TurnAborted {
    pub outcome: TurnAbortOutcome,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TurnAbortOutcome {
    Failed {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<ErrorDetail>,
    },
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub category: String,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RunEnded {
    pub reason: RunEndReason,
    pub run_id: RunId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunEndReason {
    ActiveTurnCancelled,
    IdleCancelled,
    UserQuit,
    InputClosed,
    FrontendDisconnected,
    JournalFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentBlock, Role, ToolInput};
    use serde_json::json;

    fn session_id() -> SessionId {
        SessionId::new("01K1SESSION")
    }

    fn run_id() -> RunId {
        RunId::new("01K1RUN")
    }

    fn turn_id() -> TurnId {
        TurnId::new("01K1TURN")
    }

    #[test]
    fn message_record_serializes_to_the_versioned_envelope_and_round_trips() {
        // Arrange
        let record = JournalRecord::new(
            3,
            "2026-07-26T18:42:00.123Z",
            session_id(),
            JournalEntry::MessageAdded(MessageAdded {
                run_id: run_id(),
                turn_id: turn_id(),
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "call-1".to_string(),
                        name: "read_file".to_string(),
                        input: ToolInput::Invalid("{\"path\": unclosed".to_string()),
                    }],
                },
            }),
        );

        // Act
        let serialized = serde_json::to_value(&record).unwrap();
        let unserialized: JournalRecord = serde_json::from_value(serialized.clone()).unwrap();

        // Assert
        assert_eq!(unserialized, record);
        assert_eq!(
            serialized,
            json!({
                "schema_version": 1,
                "sequence": 3,
                "recorded_at": "2026-07-26T18:42:00.123Z",
                "session_id": "01K1SESSION",
                "type": "message_added",
                "data": {
                    "run_id": "01K1RUN",
                    "turn_id": "01K1TURN",
                    "message": {
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "read_file",
                            "input": {
                                "type": "invalid",
                                "value": "{\"path\": unclosed"
                            }
                        }]
                    }
                }
            })
        );
    }

    #[test]
    fn provider_completion_preserves_reported_usage_and_cost() {
        // Arrange
        let record = JournalRecord::new(
            5,
            "2026-07-26T18:42:01Z",
            session_id(),
            JournalEntry::ProviderRoundCompleted(ProviderRoundCompleted {
                run_id: run_id(),
                turn_id: turn_id(),
                provider_round_id: ProviderRoundId::new("01K1ROUND"),
                latency_ms: 1250,
                stop_reason: StopReason::EndTurn,
                usage: Some(ModelUsage {
                    input_tokens: Some(120),
                    output_tokens: Some(30),
                    total_tokens: Some(150),
                    cached_input_tokens: Some(40),
                    ..ModelUsage::default()
                }),
                request_id: Some("request-1".to_string()),
                provider_cost: Some(ReportedCost {
                    amount: "0.0042".to_string(),
                    currency: "USD".to_string(),
                    source: "openrouter".to_string(),
                }),
            }),
        );

        // Act
        let serialized = serde_json::to_value(&record).unwrap();
        let unserialized: JournalRecord = serde_json::from_value(serialized).unwrap();

        // Assert
        assert_eq!(unserialized, record);
    }
}
