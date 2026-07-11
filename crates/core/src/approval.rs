use crate::protocol::{AgentExit, EventSink};
use crate::tools::Tool;
use crate::{AgentEvent, ApprovalDecision};
use std::collections::HashSet;
use tokio::sync::oneshot;

pub struct ApprovalGate {
    always_allowed: HashSet<String>,
}

pub enum ApprovalAuthorization {
    Approved,
    Denied { reason: String },
}

impl ApprovalGate {
    pub fn new() -> Self {
        Self {
            always_allowed: HashSet::new(),
        }
    }

    pub async fn authorize(
        &mut self,
        tool: &dyn Tool,
        call_id: &str,
        input: &serde_json::Value,
        events: &EventSink,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<ApprovalAuthorization, AgentExit> {
        let definition = tool.definition();

        if tool.read_only() || self.always_allowed.contains(&definition.name) {
            return Ok(ApprovalAuthorization::Approved);
        }

        let (decision_tx, decision_rx) = oneshot::channel();

        events
            .emit(AgentEvent::ApprovalRequest {
                id: call_id.to_string(),
                input: input.clone(),
                name: definition.name.clone(),
                respond_to: decision_tx,
            })
            .await?;

        match wait_for_response(decision_rx, events, cancel).await? {
            ApprovalDecision::Allow => Ok(ApprovalAuthorization::Approved),
            ApprovalDecision::AlwaysAllowSession => {
                self.always_allowed.insert(definition.name);

                Ok(ApprovalAuthorization::Approved)
            }
            ApprovalDecision::Deny { reason } => Ok(ApprovalAuthorization::Denied { reason }),
        }
    }
}

async fn wait_for_response(
    receiver: oneshot::Receiver<ApprovalDecision>,
    events: &EventSink,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<ApprovalDecision, AgentExit> {
    tokio::select! {
        _ = cancel.cancelled() => Err(AgentExit::Cancelled),
        _ = events.closed() => Err(AgentExit::Disconnected),
        result = receiver => result.map_err(|_| AgentExit::Disconnected),
    }
}
