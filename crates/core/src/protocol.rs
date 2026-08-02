use crate::command::CommandOutputChunk;
use crate::{Checklist, StopReason};
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub enum AgentEvent {
    ApprovalRequest {
        available_lifetimes: Vec<ApprovalLifetime>,
        input: serde_json::Value,
        respond_to: oneshot::Sender<ApprovalDecision>,
        subject: ApprovalSubject,
    },
    ChecklistUpdated(Checklist),
    CommandOutput(CommandOutputChunk),
    TextDelta(String),
    ToolStarted {
        input: serde_json::Value,
        name: String,
    },
    ToolFinished {
        is_error: bool,
        name: String,
        output: String,
    },
    ToolDenied {
        name: String,
        reason: String,
    },
    ToolRejected {
        name: String,
        error: String,
    },
    TurnComplete {
        outcome: TurnOutcome,
    },
    Warning(String),
    Error(String),
}

#[derive(Debug, PartialEq)]
pub enum TurnOutcome {
    Completed { stop_reason: StopReason },
    Paused { reason: String },
    Failed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub(crate) struct EventSink(mpsc::Sender<AgentEvent>);

pub(crate) struct FrontendGone;

impl EventSink {
    pub fn new(sender: mpsc::Sender<AgentEvent>) -> Self {
        Self(sender)
    }

    pub async fn emit(&self, event: AgentEvent) -> Result<(), FrontendGone> {
        self.0.send(event).await.map_err(|_| FrontendGone)
    }

    pub async fn closed(&self) {
        self.0.closed().await
    }

    pub fn emit_best_effort(&self, event: AgentEvent) {
        let _ = self.0.try_send(event);
    }

    pub fn sender(&self) -> &mpsc::Sender<AgentEvent> {
        &self.0
    }
}

pub struct HostHandle {
    pub events: EventSink,
    pub commands: mpsc::Receiver<AgentCommand>,
    pub cancel: CancellationToken,
}

#[derive(Debug, PartialEq)]
pub enum AgentExit {
    /// Command channel or event channel closed; clean shutdown.
    Disconnected,
    /// The authoritative journal could no longer be written.
    JournalFailed(String),
    /// Cancellation token tripped while a turn was active.
    Cancelled,
}

impl Display for AgentExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentExit::Disconnected => write!(f, "Command channel or event channel closed"),
            AgentExit::JournalFailed(message) => write!(f, "Session journal failed: {message}"),
            AgentExit::Cancelled => write!(f, "Cancellation token tripped"),
        }
    }
}

impl From<FrontendGone> for AgentExit {
    fn from(_: FrontendGone) -> Self {
        AgentExit::Disconnected
    }
}

impl From<crate::journal::JournalError> for AgentExit {
    fn from(error: crate::journal::JournalError) -> Self {
        Self::JournalFailed(error.to_string())
    }
}

#[derive(Debug, PartialEq)]
pub enum AgentCommand {
    Shutdown(ShutdownReason),
    UserInput(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShutdownReason {
    InputClosed,
    UserQuit,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalLifetime {
    Invocation,
    Run,
    Workspace,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKind {
    DockerDaemon,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct NamedCapability {
    #[serde(rename = "name")]
    kind: CapabilityKind,
    resource: String,
}

impl NamedCapability {
    pub fn new(kind: CapabilityKind, resource: impl Into<String>) -> Self {
        Self {
            kind,
            resource: resource.into(),
        }
    }

    pub fn docker_daemon(resource: impl Into<String>) -> Self {
        Self::new(CapabilityKind::DockerDaemon, resource)
    }

    pub fn kind(&self) -> CapabilityKind {
        self.kind
    }

    pub fn resource(&self) -> &str {
        &self.resource
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ApprovalSubject {
    Capability {
        capability: NamedCapability,
        tool_call_id: String,
        tool_name: String,
    },
    ToolCall {
        tool_call_id: String,
        tool_name: String,
    },
}

impl ApprovalSubject {
    pub fn capability(
        capability: NamedCapability,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        Self::Capability {
            capability,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
        }
    }

    pub fn tool_call(tool_call_id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self::ToolCall {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
        }
    }

    pub fn tool_call_id(&self) -> &str {
        match self {
            Self::Capability { tool_call_id, .. } | Self::ToolCall { tool_call_id, .. } => {
                tool_call_id
            }
        }
    }

    pub fn tool_name(&self) -> &str {
        match self {
            Self::Capability { tool_name, .. } | Self::ToolCall { tool_name, .. } => tool_name,
        }
    }

    pub fn grant(&self, lifetime: ApprovalLifetime) -> ApprovalGrant {
        let matcher = self.matcher();
        let scope = match lifetime {
            ApprovalLifetime::Invocation => ApprovalScope::Invocation {
                tool_call_id: self.tool_call_id().to_string(),
                tool_name: self.tool_name().to_string(),
            },
            ApprovalLifetime::Run => ApprovalScope::Run,
            ApprovalLifetime::Workspace => ApprovalScope::Workspace,
        };

        ApprovalGrant { matcher, scope }
    }

    fn matcher(&self) -> ApprovalMatcher {
        match self {
            Self::Capability { capability, .. } => ApprovalMatcher::Capability {
                capability: capability.clone(),
            },
            Self::ToolCall { tool_name, .. } => ApprovalMatcher::Tool {
                tool_name: tool_name.clone(),
            },
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApprovalMatcher {
    Capability { capability: NamedCapability },
    Tool { tool_name: String },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ApprovalGrant {
    pub matcher: ApprovalMatcher,
    pub scope: ApprovalScope,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApprovalScope {
    Invocation {
        tool_call_id: String,
        tool_name: String,
    },
    Run,
    Workspace,
}

impl ApprovalGrant {
    pub fn authorizes(&self, subject: &ApprovalSubject) -> bool {
        self.matcher.matches(subject) && self.scope.includes(subject)
    }

    pub fn lifetime(&self) -> ApprovalLifetime {
        self.scope.lifetime()
    }
}

impl ApprovalMatcher {
    fn matches(&self, subject: &ApprovalSubject) -> bool {
        self == &subject.matcher()
    }
}

impl ApprovalScope {
    fn includes(&self, subject: &ApprovalSubject) -> bool {
        match self {
            Self::Invocation {
                tool_call_id,
                tool_name,
            } => tool_call_id == subject.tool_call_id() && tool_name == subject.tool_name(),
            Self::Run | Self::Workspace => true,
        }
    }

    pub fn lifetime(&self) -> ApprovalLifetime {
        match self {
            Self::Invocation { .. } => ApprovalLifetime::Invocation,
            Self::Run => ApprovalLifetime::Run,
            Self::Workspace => ApprovalLifetime::Workspace,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ApprovalDecision {
    Grant(ApprovalGrant),
    Deny { reason: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalRequirement {
    None,
    Required,
}
