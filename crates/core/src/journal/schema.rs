use crate::{Message, ModelUsage, ReportedCost, StopReason, ToolDefinition};
use jiff::Timestamp;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use thiserror::Error;
use ulid::{DecodeError, Ulid};

pub const JOURNAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum IdParseError {
    #[error("ID '{value}' must start with '{expected_prefix}'")]
    InvalidPrefix {
        expected_prefix: &'static str,
        value: String,
    },

    #[error("ID '{value}' does not contain a valid ULID: {source}")]
    InvalidUlid {
        #[source]
        source: DecodeError,
        value: String,
    },
}

macro_rules! string_id {
    ($name:ident, $prefix:literal) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(Ulid);

        impl $name {
            pub fn from_ulid(value: Ulid) -> Self {
                Self(value)
            }

            pub fn generate() -> Self {
                Self(Ulid::generate())
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}_{}", $prefix, self.0)
            }
        }

        impl FromStr for $name {
            type Err = IdParseError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                let expected_prefix = concat!($prefix, "_");
                let encoded = value.strip_prefix(expected_prefix).ok_or_else(|| {
                    IdParseError::InvalidPrefix {
                        expected_prefix,
                        value: value.to_string(),
                    }
                })?;
                let ulid =
                    Ulid::from_string(encoded).map_err(|source| IdParseError::InvalidUlid {
                        source,
                        value: value.to_string(),
                    })?;

                Ok(Self(ulid))
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.collect_str(self)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                value.parse().map_err(serde::de::Error::custom)
            }
        }
    };
}

string_id!(SessionId, "sess");
string_id!(RunId, "run");
string_id!(TurnId, "turn");
string_id!(ProviderRoundId, "round");
string_id!(ApprovalId, "appr");

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct JournalRecord {
    #[serde(flatten)]
    pub entry: JournalEntry,
    pub recorded_at: Timestamp,
    pub schema_version: u32,
    pub sequence: u64,
    pub session_id: SessionId,
}

impl JournalRecord {
    pub fn new(
        sequence: u64,
        recorded_at: Timestamp,
        session_id: SessionId,
        entry: JournalEntry,
    ) -> Self {
        Self {
            entry,
            recorded_at,
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
        "sess_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap()
    }

    fn run_id() -> RunId {
        "run_01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap()
    }

    fn turn_id() -> TurnId {
        "turn_01ARZ3NDEKTSV4RRFFQ69G5FAX".parse().unwrap()
    }

    #[test]
    fn message_record_serializes_to_the_versioned_envelope_and_round_trips() {
        // Arrange
        let record = JournalRecord::new(
            3,
            "2026-07-26T18:42:00.123Z".parse().unwrap(),
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
                "session_id": "sess_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "type": "message_added",
                "data": {
                    "run_id": "run_01ARZ3NDEKTSV4RRFFQ69G5FAW",
                    "turn_id": "turn_01ARZ3NDEKTSV4RRFFQ69G5FAX",
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
            "2026-07-26T18:42:01Z".parse().unwrap(),
            session_id(),
            JournalEntry::ProviderRoundCompleted(ProviderRoundCompleted {
                run_id: run_id(),
                turn_id: turn_id(),
                provider_round_id: "round_01ARZ3NDEKTSV4RRFFQ69G5FAY".parse().unwrap(),
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

    #[test]
    fn prefixed_ids_reject_the_wrong_domain_and_invalid_ulids() {
        // Arrange
        let wrong_domain = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";
        let invalid_ulid = "sess_not-a-ulid";

        // Act
        let wrong_domain_error = wrong_domain.parse::<SessionId>().unwrap_err();
        let invalid_ulid_error = invalid_ulid.parse::<SessionId>().unwrap_err();

        // Assert
        assert!(matches!(
            wrong_domain_error,
            IdParseError::InvalidPrefix {
                expected_prefix: "sess_",
                ..
            }
        ));
        assert!(matches!(
            invalid_ulid_error,
            IdParseError::InvalidUlid { .. }
        ));
    }

    #[test]
    fn generated_ids_use_the_domain_prefix_and_round_trip() {
        // Arrange
        let generated = ApprovalId::generate();

        // Act
        let serialized = serde_json::to_string(&generated).unwrap();
        let unserialized: ApprovalId = serde_json::from_str(&serialized).unwrap();

        // Assert
        assert!(serialized.starts_with("\"appr_"));
        assert_eq!(unserialized, generated);
    }
}
