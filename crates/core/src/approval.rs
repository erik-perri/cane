use crate::protocol::{AgentExit, ApprovalRequirement, EventSink};
use crate::{AgentEvent, ApprovalDecision};
use std::collections::HashSet;
use tokio::sync::oneshot;

pub struct ApprovalGate {
    always_allowed: HashSet<String>,
}

#[derive(Debug)]
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
        requirement: ApprovalRequirement,
        tool_name: &str,
        call_id: &str,
        input: &serde_json::Value,
        events: &EventSink,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<ApprovalAuthorization, AgentExit> {
        if requirement == ApprovalRequirement::None || self.always_allowed.contains(tool_name) {
            return Ok(ApprovalAuthorization::Approved);
        }

        let (decision_tx, decision_rx) = oneshot::channel();

        events
            .emit(AgentEvent::ApprovalRequest {
                id: call_id.to_string(),
                input: input.clone(),
                name: tool_name.to_string(),
                respond_to: decision_tx,
            })
            .await?;

        match wait_for_response(decision_rx, events, cancel).await? {
            ApprovalDecision::Allow => Ok(ApprovalAuthorization::Approved),
            ApprovalDecision::AlwaysAllowSession => {
                self.always_allowed.insert(tool_name.to_string());

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn approval_not_required_returns_approved_without_emitting_a_request() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();

        // Act
        let result = gate
            .authorize(
                ApprovalRequirement::None,
                "read_file",
                "read-1",
                &json!({ "file": "test.txt" }),
                &sink,
                &cancel,
            )
            .await;

        // Assert
        assert!(matches!(result, Ok(ApprovalAuthorization::Approved)));
        assert!(matches!(
            events_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn required_approval_emits_the_call_details_and_returns_approved_on_allow() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let payload = json!({ "file": "test.txt", "contents": "test" });

        // Act
        let (authorize_result, ()) = tokio::join!(
            gate.authorize(
                ApprovalRequirement::Required,
                "write_file",
                "write-1",
                &payload,
                &sink,
                &cancel,
            ),
            async {
                let event = events_rx.recv().await.unwrap();

                let AgentEvent::ApprovalRequest {
                    input,
                    id,
                    name,
                    respond_to,
                } = event
                else {
                    panic!("Expected ApprovalRequest event");
                };

                assert_eq!(input, payload);
                assert_eq!(id, "write-1");
                assert_eq!(name, "write_file");

                respond_to.send(ApprovalDecision::Allow).unwrap();
            }
        );

        // Assert
        assert!(matches!(
            authorize_result,
            Ok(ApprovalAuthorization::Approved)
        ));
    }

    #[tokio::test]
    async fn denial_returns_the_supplied_reason() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let deny_reason = "Mock deny reason".to_string();
        let payload = json!({ "file": "test.txt", "contents": "test" });

        // Act
        let (authorize_result, ()) = tokio::join!(
            gate.authorize(
                ApprovalRequirement::Required,
                "write_file",
                "write-1",
                &payload,
                &sink,
                &cancel,
            ),
            async {
                let event = events_rx.recv().await.unwrap();

                let AgentEvent::ApprovalRequest {
                    input,
                    id,
                    name,
                    respond_to,
                } = event
                else {
                    panic!("Expected ApprovalRequest event");
                };

                assert_eq!(input, payload);
                assert_eq!(id, "write-1");
                assert_eq!(name, "write_file");

                respond_to
                    .send(ApprovalDecision::Deny {
                        reason: deny_reason.clone(),
                    })
                    .unwrap();
            }
        );

        // Assert
        let Ok(ApprovalAuthorization::Denied {
            reason: provided_reason,
        }) = authorize_result
        else {
            panic!("Expected ApprovalAuthorization::Denied")
        };
        assert_eq!(provided_reason, deny_reason);
    }

    #[tokio::test]
    async fn always_allow_skips_later_requests_for_the_same_tool() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let payload_one = json!({ "file": "test.txt", "contents": "test 1" });
        let payload_two = json!({ "file": "test.txt", "contents": "test 2" });

        // Act
        let (authorize_result_one, ()) = tokio::join!(
            gate.authorize(
                ApprovalRequirement::Required,
                "write_file",
                "write-1",
                &payload_one,
                &sink,
                &cancel,
            ),
            async {
                let event = events_rx.recv().await.unwrap();

                let AgentEvent::ApprovalRequest {
                    input,
                    id,
                    name,
                    respond_to,
                } = event
                else {
                    panic!("Expected ApprovalRequest event");
                };

                assert_eq!(input, payload_one);
                assert_eq!(id, "write-1");
                assert_eq!(name, "write_file");

                respond_to
                    .send(ApprovalDecision::AlwaysAllowSession)
                    .unwrap();
            }
        );

        let authorize_result_two = gate
            .authorize(
                ApprovalRequirement::Required,
                "write_file",
                "write-2",
                &payload_two,
                &sink,
                &cancel,
            )
            .await;

        // Assert
        let Ok(ApprovalAuthorization::Approved) = authorize_result_one else {
            panic!("Expected ApprovalAuthorization::Approved")
        };

        let Ok(ApprovalAuthorization::Approved) = authorize_result_two else {
            panic!("Expected ApprovalAuthorization::Approved")
        };

        assert!(matches!(
            events_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn always_allow_does_not_skip_requests_for_another_tool() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let payload_one = json!({ "file": "test.txt", "contents": "test 1" });
        let payload_two = json!({ "file": "test.txt", "contents": "test 2" });

        // Act
        let (authorize_result_one, ()) = tokio::join!(
            gate.authorize(
                ApprovalRequirement::Required,
                "write_file",
                "write-1",
                &payload_one,
                &sink,
                &cancel,
            ),
            async {
                let event = events_rx.recv().await.unwrap();

                let AgentEvent::ApprovalRequest {
                    input,
                    id,
                    name,
                    respond_to,
                } = event
                else {
                    panic!("Expected ApprovalRequest event");
                };

                assert_eq!(input, payload_one);
                assert_eq!(id, "write-1");
                assert_eq!(name, "write_file");

                respond_to
                    .send(ApprovalDecision::AlwaysAllowSession)
                    .unwrap();
            }
        );

        let (authorize_result_two, ()) = tokio::join!(
            gate.authorize(
                ApprovalRequirement::Required,
                "write_file_copy",
                "write-2",
                &payload_two,
                &sink,
                &cancel,
            ),
            async {
                let event = events_rx.recv().await.unwrap();

                let AgentEvent::ApprovalRequest {
                    input,
                    id,
                    name,
                    respond_to,
                } = event
                else {
                    panic!("Expected ApprovalRequest event");
                };

                assert_eq!(input, payload_two);
                assert_eq!(id, "write-2");
                assert_eq!(name, "write_file_copy");

                respond_to.send(ApprovalDecision::Allow).unwrap();
            }
        );

        // Assert
        let Ok(ApprovalAuthorization::Approved) = authorize_result_one else {
            panic!("Expected ApprovalAuthorization::Approved")
        };

        let Ok(ApprovalAuthorization::Approved) = authorize_result_two else {
            panic!("Expected ApprovalAuthorization::Approved")
        };

        assert!(matches!(
            events_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn cancellation_while_waiting_for_approval_returns_cancelled() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let (events_tx, _events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let payload = json!({ "file": "test.txt", "contents": "test" });

        cancel.cancel();

        // Act
        let result = gate
            .authorize(
                ApprovalRequirement::Required,
                "write_file",
                "write-1",
                &payload,
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
        let mut gate = ApprovalGate::new();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let payload = json!({ "file": "test.txt", "contents": "test" });

        // Act
        let (authorize_result, ()) = tokio::join!(
            gate.authorize(
                ApprovalRequirement::Required,
                "write_file",
                "write-1",
                &payload,
                &sink,
                &cancel,
            ),
            async {
                let event = events_rx.recv().await.unwrap();

                let AgentEvent::ApprovalRequest {
                    input,
                    id,
                    name,
                    respond_to,
                } = event
                else {
                    panic!("Expected ApprovalRequest event");
                };

                assert_eq!(input, payload);
                assert_eq!(id, "write-1");
                assert_eq!(name, "write_file");

                drop(respond_to);
            }
        );

        // Assert
        assert!(matches!(authorize_result, Err(AgentExit::Disconnected)));
    }

    #[tokio::test]
    async fn dropping_the_event_receiver_returns_disconnected() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let (events_tx, _) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let payload = json!({ "file": "test.txt", "contents": "test" });

        // Act
        let result = gate
            .authorize(
                ApprovalRequirement::Required,
                "write_file",
                "write-1",
                &payload,
                &sink,
                &cancel,
            )
            .await;

        // Assert
        assert!(matches!(result, Err(AgentExit::Disconnected)));
    }
}
