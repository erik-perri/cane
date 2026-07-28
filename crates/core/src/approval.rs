use crate::journal::ApprovalId;
use crate::protocol::{AgentExit, ApprovalRequirement, EventSink};
use crate::{AgentEvent, ApprovalDecision};
use std::collections::HashMap;
use tokio::sync::oneshot;

pub struct ApprovalGate {
    run_approvals: HashMap<String, ApprovalId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalAuthorization {
    ApprovedForRun { approval_id: ApprovalId },
    ApprovedOnce { approval_id: ApprovalId },
    NotRequired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalCheck {
    Authorized(ApprovalAuthorization),
    RequiresDecision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApprovalOutcome {
    Authorized(ApprovalAuthorization),
    Denied { reason: String },
}

impl ApprovalGate {
    pub fn new() -> Self {
        Self {
            run_approvals: HashMap::new(),
        }
    }

    pub fn check(&self, requirement: ApprovalRequirement, tool_name: &str) -> ApprovalCheck {
        if requirement == ApprovalRequirement::None {
            return ApprovalCheck::Authorized(ApprovalAuthorization::NotRequired);
        }

        match self.run_approvals.get(tool_name) {
            Some(approval_id) => ApprovalCheck::Authorized(ApprovalAuthorization::ApprovedForRun {
                approval_id: *approval_id,
            }),
            None => ApprovalCheck::RequiresDecision,
        }
    }

    pub fn record_run_approval(&mut self, tool_name: String, approval_id: ApprovalId) {
        self.run_approvals.insert(tool_name, approval_id);
    }

    pub fn apply_decision(
        &mut self,
        tool_name: &str,
        approval_id: ApprovalId,
        decision: ApprovalDecision,
    ) -> ApprovalOutcome {
        match decision {
            ApprovalDecision::AllowOnce => {
                ApprovalOutcome::Authorized(ApprovalAuthorization::ApprovedOnce { approval_id })
            }
            ApprovalDecision::AllowForRun => {
                self.record_run_approval(tool_name.to_string(), approval_id);
                ApprovalOutcome::Authorized(ApprovalAuthorization::ApprovedForRun { approval_id })
            }
            ApprovalDecision::Deny { reason } => ApprovalOutcome::Denied { reason },
        }
    }
}

pub async fn request_approval(
    tool_name: &str,
    call_id: &str,
    input: &serde_json::Value,
    events: &EventSink,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<ApprovalDecision, AgentExit> {
    let (decision_tx, decision_rx) = oneshot::channel();

    events
        .emit(AgentEvent::ApprovalRequest {
            id: call_id.to_string(),
            input: input.clone(),
            name: tool_name.to_string(),
            respond_to: decision_tx,
        })
        .await?;

    wait_for_response(decision_rx, events, cancel).await
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    fn approval_id() -> ApprovalId {
        "appr_01ARZ3NDEKTSV4RRFFQ69G5FAY".parse().unwrap()
    }

    #[test]
    fn check_returns_not_required_for_tools_that_do_not_require_approval() {
        // Arrange
        let gate = ApprovalGate::new();

        // Act
        let result = gate.check(ApprovalRequirement::None, "read_file");

        // Assert
        assert_eq!(
            result,
            ApprovalCheck::Authorized(ApprovalAuthorization::NotRequired)
        );
    }

    #[test]
    fn check_requires_a_decision_when_the_tool_has_no_run_approval() {
        // Arrange
        let gate = ApprovalGate::new();

        // Act
        let result = gate.check(ApprovalRequirement::Required, "write_file");

        // Assert
        assert_eq!(result, ApprovalCheck::RequiresDecision);
    }

    #[test]
    fn run_approvals_authorize_only_the_matching_tool() {
        // Arrange
        let mut gate = ApprovalGate::new();
        gate.record_run_approval("write_file".to_string(), approval_id());

        // Act
        let matching = gate.check(ApprovalRequirement::Required, "write_file");
        let other = gate.check(ApprovalRequirement::Required, "edit_file");

        // Assert
        assert_eq!(
            matching,
            ApprovalCheck::Authorized(ApprovalAuthorization::ApprovedForRun {
                approval_id: approval_id()
            })
        );
        assert_eq!(other, ApprovalCheck::RequiresDecision);
    }

    #[test]
    fn applying_decisions_preserves_the_authorizing_approval() {
        // Arrange
        let mut gate = ApprovalGate::new();

        // Act
        let once = gate.apply_decision("write_file", approval_id(), ApprovalDecision::AllowOnce);
        let run = gate.apply_decision("edit_file", approval_id(), ApprovalDecision::AllowForRun);
        let denied = gate.apply_decision(
            "glob",
            approval_id(),
            ApprovalDecision::Deny {
                reason: "not now".to_string(),
            },
        );

        // Assert
        assert_eq!(
            once,
            ApprovalOutcome::Authorized(ApprovalAuthorization::ApprovedOnce {
                approval_id: approval_id(),
            })
        );
        assert_eq!(
            run,
            ApprovalOutcome::Authorized(ApprovalAuthorization::ApprovedForRun {
                approval_id: approval_id(),
            })
        );
        assert_eq!(
            gate.check(ApprovalRequirement::Required, "edit_file"),
            ApprovalCheck::Authorized(ApprovalAuthorization::ApprovedForRun {
                approval_id: approval_id(),
            })
        );
        assert_eq!(
            denied,
            ApprovalOutcome::Denied {
                reason: "not now".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn request_emits_the_call_details_and_returns_the_decision() {
        // Arrange
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let payload = json!({ "file": "test.txt", "contents": "test" });

        // Act
        let (request_result, ()) = tokio::join!(
            request_approval("write_file", "write-1", &payload, &sink, &cancel),
            async {
                let event = events_rx.recv().await.unwrap();

                let AgentEvent::ApprovalRequest {
                    id,
                    input,
                    name,
                    respond_to,
                } = event
                else {
                    panic!("expected ApprovalRequest");
                };

                assert_eq!(id, "write-1");
                assert_eq!(input, payload);
                assert_eq!(name, "write_file");

                respond_to
                    .send(ApprovalDecision::Deny {
                        reason: "not this file".to_string(),
                    })
                    .unwrap();
            }
        );

        // Assert
        assert_eq!(
            request_result,
            Ok(ApprovalDecision::Deny {
                reason: "not this file".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn cancellation_while_waiting_for_approval_returns_cancelled() {
        // Arrange
        let (events_tx, _events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        cancel.cancel();

        // Act
        let result = request_approval(
            "write_file",
            "write-1",
            &json!({ "path": "test.txt" }),
            &sink,
            &cancel,
        )
        .await;

        // Assert
        assert!(matches!(result, Err(AgentExit::Cancelled)));
    }

    #[tokio::test]
    async fn dropping_the_approval_responder_returns_disconnected() {
        // Arrange
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let payload = json!({ "path": "test.txt" });

        // Act
        let (request_result, ()) = tokio::join!(
            request_approval("write_file", "write-1", &payload, &sink, &cancel),
            async {
                let event = events_rx.recv().await.unwrap();
                let AgentEvent::ApprovalRequest { respond_to, .. } = event else {
                    panic!("expected ApprovalRequest");
                };
                drop(respond_to);
            }
        );

        // Assert
        assert!(matches!(request_result, Err(AgentExit::Disconnected)));
    }

    #[tokio::test]
    async fn dropping_the_event_receiver_returns_disconnected() {
        // Arrange
        let (events_tx, events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        drop(events_rx);

        // Act
        let result = request_approval(
            "write_file",
            "write-1",
            &json!({ "path": "test.txt" }),
            &sink,
            &cancel,
        )
        .await;

        // Assert
        assert!(matches!(result, Err(AgentExit::Disconnected)));
    }
}
