use crate::{
    ApprovalGrant, ApprovalLifetime, ApprovalSubject, Message, ModelUsage, NamedCapability,
    ProviderDescriptor, ReportedCost, StopReason, ToolDefinition,
};
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub approval_grants: Vec<EffectiveApprovalGrant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git: Option<GitContext>,
    pub max_output_tokens: u32,
    pub model: String,
    pub provider: ProviderDescriptor,
    pub run_id: RunId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shell_policy: Option<ShellPolicy>,
    pub tool_catalog: Vec<ToolDefinition>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EffectiveApprovalGrant {
    pub grant: ApprovalGrant,
    pub source: ApprovalGrantSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalGrantSource {
    WorkspaceConfiguration,
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
    pub provider: ProviderDescriptor,
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
    #[serde(default = "legacy_approval_lifetimes")]
    pub available_lifetimes: Vec<ApprovalLifetime>,
    #[serde(flatten)]
    pub subject: ApprovalSubject,
    pub turn_id: TurnId,
}

fn legacy_approval_lifetimes() -> Vec<ApprovalLifetime> {
    vec![ApprovalLifetime::Invocation, ApprovalLifetime::Run]
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ApprovalDecided {
    pub approval_id: ApprovalId,
    pub decision: JournalApprovalDecision,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JournalApprovalDecision {
    AllowForRun,
    AllowOnce,
    Deny { reason: String },
    Grant { grant: ApprovalGrant },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolStarted {
    pub authorization: ToolAuthorization,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ToolExecutionStarted>,
    pub tool_call_id: String,
    pub tool_name: String,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolAuthorization {
    ApprovedForRun {
        approval_id: ApprovalId,
    },
    ApprovedOnce {
        approval_id: ApprovalId,
    },
    Granted {
        approval_id: ApprovalId,
        grant: ApprovalGrant,
    },
    NotRequired,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolCompleted {
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ToolExecutionCompleted>,
    pub tool_call_id: String,
    pub turn_id: TurnId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShellPolicy {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<ShellSandboxBackend>,
    pub boundaries: ShellBoundaries,
    pub exposed_roots: Vec<ShellExposedRoot>,
    pub mode: ShellMode,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShellSandboxBackend {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShellBoundaries {
    pub environment: BoundaryEnforcement,
    pub filesystem: BoundaryEnforcement,
    pub network: BoundaryEnforcement,
    pub process: BoundaryEnforcement,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BoundaryEnforcement {
    Enforced { mechanism: String },
    NotApplicable,
    Unrestricted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShellExposedRoot {
    pub access: FilesystemAccess,
    pub path: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilesystemAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellMode {
    Disabled,
    Sandboxed,
    Unsafe,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolExecutionStarted {
    Shell {
        capabilities: Vec<ExecutionCapability>,
    },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ExecutionCapability {
    pub capability: NamedCapability,
    pub source: CapabilityAuthorizationSource,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CapabilityAuthorizationSource {
    Approval { approval_id: ApprovalId },
    WorkspaceConfiguration,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolExecutionCompleted {
    Shell {
        stderr: CapturedStream,
        stdout: CapturedStream,
        termination: CommandTermination,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapturedStream {
    pub bytes: u64,
    pub truncated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CommandTermination {
    Exited { code: i32 },
    Signaled { signal: i32 },
    TimedOut,
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
    FrontendDisconnected,
    IdleCancelled,
    InputClosed,
    UserQuit,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentBlock, ProviderAdapter, Role, ToolInput};
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
    fn run_started_serializes_a_structured_provider_descriptor() {
        // Arrange
        let record = JournalRecord::new(
            2,
            "2026-07-26T18:42:00.500Z".parse().unwrap(),
            session_id(),
            JournalEntry::RunStarted(RunStarted {
                approval_grants: Vec::new(),
                git: None,
                max_output_tokens: 32_000,
                model: "test-model".to_string(),
                provider: ProviderDescriptor {
                    adapter: ProviderAdapter::OpenAiCompatible,
                    endpoint: "https://example.test/v1/chat/completions".to_string(),
                },
                run_id: run_id(),
                shell_policy: None,
                tool_catalog: Vec::new(),
            }),
        );

        // Act
        let serialized = serde_json::to_value(&record).unwrap();
        let unserialized: JournalRecord = serde_json::from_value(serialized.clone()).unwrap();

        // Assert
        assert_eq!(unserialized, record);
        assert_eq!(
            serialized["data"]["provider"],
            json!({
                "adapter": "openai_compatible",
                "endpoint": "https://example.test/v1/chat/completions"
            })
        );
    }

    #[test]
    fn legacy_tool_approval_requests_decode_into_typed_subjects() {
        // Arrange
        let serialized = json!({
            "approval_id": "appr_01ARZ3NDEKTSV4RRFFQ69G5FAY",
            "tool_call_id": "call-1",
            "tool_name": "write_file",
            "turn_id": "turn_01ARZ3NDEKTSV4RRFFQ69G5FAX"
        });

        // Act
        let request: ApprovalRequested = serde_json::from_value(serialized).unwrap();

        // Assert
        assert_eq!(
            request.available_lifetimes,
            vec![ApprovalLifetime::Invocation, ApprovalLifetime::Run]
        );
        assert_eq!(
            request.subject,
            ApprovalSubject::tool_call("call-1", "write_file")
        );
    }

    #[test]
    fn typed_capability_approvals_and_shell_diagnostics_round_trip() {
        // Arrange
        let capability = NamedCapability::docker_daemon("unix:///var/run/docker.sock");
        let subject = ApprovalSubject::capability(capability.clone(), "shell-1", "shell");
        let grant = subject.grant(ApprovalLifetime::Run);
        let request = ApprovalRequested {
            approval_id: "appr_01ARZ3NDEKTSV4RRFFQ69G5FAY".parse().unwrap(),
            available_lifetimes: vec![ApprovalLifetime::Run, ApprovalLifetime::Workspace],
            subject,
            turn_id: turn_id(),
        };
        let decision = JournalApprovalDecision::Grant {
            grant: grant.clone(),
        };
        let started = ToolExecutionStarted::Shell {
            capabilities: vec![ExecutionCapability {
                capability,
                source: CapabilityAuthorizationSource::Approval {
                    approval_id: "appr_01ARZ3NDEKTSV4RRFFQ69G5FAY".parse().unwrap(),
                },
            }],
        };
        let completed = ToolExecutionCompleted::Shell {
            stderr: CapturedStream {
                bytes: 7,
                truncated: false,
            },
            stdout: CapturedStream {
                bytes: 32_768,
                truncated: true,
            },
            termination: CommandTermination::Exited { code: 0 },
        };
        let policy = ShellPolicy {
            backend: Some(ShellSandboxBackend {
                name: "bubblewrap".to_string(),
                version: Some("1.0.0".to_string()),
            }),
            boundaries: ShellBoundaries {
                environment: BoundaryEnforcement::Enforced {
                    mechanism: "allowlist".to_string(),
                },
                filesystem: BoundaryEnforcement::Enforced {
                    mechanism: "mount_namespace".to_string(),
                },
                network: BoundaryEnforcement::Enforced {
                    mechanism: "network_namespace".to_string(),
                },
                process: BoundaryEnforcement::Enforced {
                    mechanism: "pid_namespace".to_string(),
                },
            },
            exposed_roots: vec![ShellExposedRoot {
                access: FilesystemAccess::ReadWrite,
                path: "/workspace".to_string(),
            }],
            mode: ShellMode::Sandboxed,
        };
        let effective_grant = EffectiveApprovalGrant {
            grant: ApprovalSubject::capability(
                NamedCapability::docker_daemon("unix:///var/run/docker.sock"),
                "shell-1",
                "shell",
            )
            .grant(ApprovalLifetime::Workspace),
            source: ApprovalGrantSource::WorkspaceConfiguration,
        };
        let values = [
            serde_json::to_value(&request).unwrap(),
            serde_json::to_value(&decision).unwrap(),
            serde_json::to_value(&started).unwrap(),
            serde_json::to_value(&completed).unwrap(),
            serde_json::to_value(&policy).unwrap(),
            serde_json::to_value(&effective_grant).unwrap(),
        ];

        // Act
        let decoded_request: ApprovalRequested = serde_json::from_value(values[0].clone()).unwrap();
        let decoded_decision: JournalApprovalDecision =
            serde_json::from_value(values[1].clone()).unwrap();
        let decoded_started: ToolExecutionStarted =
            serde_json::from_value(values[2].clone()).unwrap();
        let decoded_completed: ToolExecutionCompleted =
            serde_json::from_value(values[3].clone()).unwrap();
        let decoded_policy: ShellPolicy = serde_json::from_value(values[4].clone()).unwrap();
        let decoded_effective_grant: EffectiveApprovalGrant =
            serde_json::from_value(values[5].clone()).unwrap();

        // Assert
        assert_eq!(values[0]["capability"]["name"], "docker_daemon");
        assert_eq!(decoded_request, request);
        assert_eq!(decoded_decision, decision);
        assert_eq!(decoded_started, started);
        assert_eq!(decoded_completed, completed);
        assert_eq!(decoded_policy, policy);
        assert_eq!(decoded_effective_grant, effective_grant);
        assert_eq!(grant.lifetime(), ApprovalLifetime::Run);
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
