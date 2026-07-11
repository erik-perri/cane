use crate::protocol::{AgentCommand, AgentExit, EventSink};
use crate::tools::Tool;
use crate::{AgentEvent, ApprovalDecision};
use std::collections::HashSet;

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
        commands: &mut tokio::sync::mpsc::Receiver<AgentCommand>,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<ApprovalAuthorization, AgentExit> {
        let definition = tool.definition();

        if tool.read_only() || self.always_allowed.contains(&definition.name) {
            return Ok(ApprovalAuthorization::Approved);
        }

        events
            .emit(AgentEvent::ApprovalRequest {
                id: call_id.to_string(),
                input: input.clone(),
                name: definition.name.clone(),
            })
            .await?;

        match wait_for_response(call_id, events, commands, cancel).await? {
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
    call_id: &str,
    events: &EventSink,
    commands: &mut tokio::sync::mpsc::Receiver<AgentCommand>,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<ApprovalDecision, AgentExit> {
    loop {
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(AgentExit::Cancelled),
            _ = events.closed() => return Err(AgentExit::Disconnected),
            command = commands.recv() => {
                command.ok_or(AgentExit::Disconnected)?
            }
        };

        match response {
            AgentCommand::Approval {
                decision,
                id: approved_id,
            } => {
                if approved_id.eq(call_id) {
                    return Ok(decision);
                }

                tracing::warn!(
                    approval_id = %call_id,
                    "ignoring unexpected command approval"
                );
            }
            AgentCommand::UserInput(..) => {
                tracing::warn!(
                    approval_id = %call_id,
                    "ignoring user input while approval is pending"
                );
            }
        }
    }
}
