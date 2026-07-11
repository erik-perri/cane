use crate::StopReason;
use std::fmt::Display;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, PartialEq)]
pub enum AgentEvent {
    ApprovalRequest {
        id: String,
        input: serde_json::Value,
        name: String,
    },
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
    TurnComplete {
        outcome: TurnOutcome,
    },
    Error(String),
}

#[derive(Debug, PartialEq)]
pub enum TurnOutcome {
    Completed { stop_reason: StopReason },
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

    pub fn sender(&self) -> &mpsc::Sender<AgentEvent> {
        &self.0
    }
}

pub struct HostHandle {
    pub events: EventSink,
    pub commands: mpsc::Receiver<AgentCommand>,
    pub cancel: CancellationToken,
}

#[derive(Debug)]
pub enum AgentExit {
    /// Command channel or event channel closed — clean shutdown.
    Disconnected,
    /// Cancellation token tripped.
    Cancelled,
}

impl Display for AgentExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentExit::Disconnected => write!(f, "Command channel or event channel closed"),
            AgentExit::Cancelled => write!(f, "Cancellation token tripped"),
        }
    }
}

impl From<FrontendGone> for AgentExit {
    fn from(_: FrontendGone) -> Self {
        AgentExit::Disconnected
    }
}

#[derive(Debug, PartialEq)]
pub enum AgentCommand {
    Approval {
        id: String,
        decision: ApprovalDecision,
    },
    UserInput(String),
}

#[derive(Debug, PartialEq)]
pub enum ApprovalDecision {
    Allow,
    AlwaysAllowSession,
    Deny { reason: String },
}
