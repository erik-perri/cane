use crate::journal::ApprovalId;
use crate::protocol::{AgentExit, ApprovalRequirement, EventSink};
use crate::{AgentEvent, ApprovalDecision, ApprovalGrant, ApprovalLifetime, ApprovalSubject};
use tokio::sync::oneshot;

pub struct ApprovalGate {
    run_approvals: Vec<ApprovalAuthorization>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApprovalAuthorization {
    Granted {
        approval_id: ApprovalId,
        grant: ApprovalGrant,
    },
    WorkspaceConfigured {
        grant: ApprovalGrant,
    },
    NotRequired,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApprovalCheck {
    Authorized(ApprovalAuthorization),
    RequiresDecision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApprovalOutcome {
    Authorized(ApprovalAuthorization),
    Denied { reason: String },
    InvalidGrant { reason: String },
}

impl ApprovalGate {
    pub fn new() -> Self {
        Self {
            run_approvals: Vec::new(),
        }
    }

    pub fn check(
        &self,
        requirement: ApprovalRequirement,
        subject: &ApprovalSubject,
    ) -> ApprovalCheck {
        if requirement == ApprovalRequirement::None {
            return ApprovalCheck::Authorized(ApprovalAuthorization::NotRequired);
        }

        self.run_approvals
            .iter()
            .find(|authorization| match authorization {
                ApprovalAuthorization::Granted { grant, .. }
                | ApprovalAuthorization::WorkspaceConfigured { grant } => grant.authorizes(subject),
                ApprovalAuthorization::NotRequired => false,
            })
            .cloned()
            .map_or(ApprovalCheck::RequiresDecision, ApprovalCheck::Authorized)
    }

    pub fn apply_decision(
        &mut self,
        available_lifetimes: &[ApprovalLifetime],
        approval_id: ApprovalId,
        decision: &ApprovalDecision,
        subject: &ApprovalSubject,
    ) -> ApprovalOutcome {
        match decision {
            ApprovalDecision::Grant(grant) => {
                if !available_lifetimes.contains(&grant.lifetime()) {
                    return ApprovalOutcome::InvalidGrant {
                        reason: "approval grant uses a lifetime that was not offered".to_string(),
                    };
                }
                if !grant.authorizes(subject) {
                    return ApprovalOutcome::InvalidGrant {
                        reason: "approval grant does not authorize the requested subject"
                            .to_string(),
                    };
                }

                let authorization = ApprovalAuthorization::Granted {
                    approval_id,
                    grant: grant.clone(),
                };
                if grant.lifetime() != ApprovalLifetime::Invocation {
                    self.run_approvals.push(authorization.clone());
                }
                ApprovalOutcome::Authorized(authorization)
            }
            ApprovalDecision::Deny { reason } => ApprovalOutcome::Denied {
                reason: reason.clone(),
            },
        }
    }

    pub fn seed_workspace_grants(&mut self, grants: impl IntoIterator<Item = ApprovalGrant>) {
        self.run_approvals.extend(
            grants
                .into_iter()
                .map(|grant| ApprovalAuthorization::WorkspaceConfigured { grant }),
        );
    }
}

pub async fn request_approval(
    available_lifetimes: Vec<ApprovalLifetime>,
    input: &serde_json::Value,
    subject: ApprovalSubject,
    events: &EventSink,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<ApprovalDecision, AgentExit> {
    let (decision_tx, decision_rx) = oneshot::channel();

    events
        .emit(AgentEvent::ApprovalRequest {
            available_lifetimes,
            input: input.clone(),
            respond_to: decision_tx,
            subject,
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

    fn offered_lifetimes() -> Vec<ApprovalLifetime> {
        vec![ApprovalLifetime::Invocation, ApprovalLifetime::Run]
    }

    fn subject(call_id: &str, tool_name: &str) -> ApprovalSubject {
        ApprovalSubject::tool_call(call_id, tool_name)
    }

    #[test]
    fn check_returns_not_required_for_tools_that_do_not_require_approval() {
        // Arrange
        let gate = ApprovalGate::new();
        let subject = subject("read-1", "read_file");

        // Act
        let result = gate.check(ApprovalRequirement::None, &subject);

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
        let subject = subject("write-1", "write_file");

        // Act
        let result = gate.check(ApprovalRequirement::Required, &subject);

        // Assert
        assert_eq!(result, ApprovalCheck::RequiresDecision);
    }

    #[test]
    fn run_approvals_authorize_only_the_matching_tool() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let approved_subject = subject("write-1", "write_file");
        let run_grant = approved_subject.grant(ApprovalLifetime::Run);
        let decision = ApprovalDecision::Grant(run_grant.clone());
        let outcome = gate.apply_decision(
            &offered_lifetimes(),
            approval_id(),
            &decision,
            &approved_subject,
        );
        let matching_subject = subject("write-2", "write_file");
        let other_subject = subject("edit-1", "edit_file");

        // Act
        let matching = gate.check(ApprovalRequirement::Required, &matching_subject);
        let other = gate.check(ApprovalRequirement::Required, &other_subject);

        // Assert
        assert_eq!(
            outcome,
            ApprovalOutcome::Authorized(ApprovalAuthorization::Granted {
                approval_id: approval_id(),
                grant: run_grant.clone(),
            })
        );
        assert_eq!(
            matching,
            ApprovalCheck::Authorized(ApprovalAuthorization::Granted {
                approval_id: approval_id(),
                grant: run_grant,
            })
        );
        assert_eq!(other, ApprovalCheck::RequiresDecision);
    }

    #[test]
    fn run_capability_approvals_follow_the_exact_resource_across_tool_calls() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let capability = crate::NamedCapability::docker_daemon("unix:///run/user/1000/docker.sock");
        let approved_subject = ApprovalSubject::capability(capability.clone(), "shell-1", "shell");
        let run_grant = approved_subject.grant(ApprovalLifetime::Run);
        let decision = ApprovalDecision::Grant(run_grant.clone());
        let outcome = gate.apply_decision(
            &[ApprovalLifetime::Run],
            approval_id(),
            &decision,
            &approved_subject,
        );
        let same_endpoint = ApprovalSubject::capability(capability, "shell-2", "shell");
        let changed_endpoint = ApprovalSubject::capability(
            crate::NamedCapability::docker_daemon("unix:///var/run/docker.sock"),
            "shell-3",
            "shell",
        );

        // Act
        let reused = gate.check(ApprovalRequirement::Required, &same_endpoint);
        let changed = gate.check(ApprovalRequirement::Required, &changed_endpoint);

        // Assert
        assert_eq!(
            outcome,
            ApprovalOutcome::Authorized(ApprovalAuthorization::Granted {
                approval_id: approval_id(),
                grant: run_grant.clone(),
            })
        );
        assert_eq!(
            reused,
            ApprovalCheck::Authorized(ApprovalAuthorization::Granted {
                approval_id: approval_id(),
                grant: run_grant,
            })
        );
        assert_eq!(changed, ApprovalCheck::RequiresDecision);
    }

    #[test]
    fn configured_workspace_grants_retain_their_source_and_match_exactly() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let configured = ApprovalSubject::capability(
            crate::NamedCapability::docker_daemon("unix:///configured.sock"),
            "seed",
            "shell",
        )
        .grant(ApprovalLifetime::Workspace);
        gate.seed_workspace_grants([configured.clone()]);
        let matching = ApprovalSubject::capability(
            crate::NamedCapability::docker_daemon("unix:///configured.sock"),
            "shell-1",
            "shell",
        );
        let different = ApprovalSubject::capability(
            crate::NamedCapability::docker_daemon("unix:///other.sock"),
            "shell-2",
            "shell",
        );

        // Act
        let authorized = gate.check(ApprovalRequirement::Required, &matching);
        let requires_decision = gate.check(ApprovalRequirement::Required, &different);

        // Assert
        assert_eq!(
            authorized,
            ApprovalCheck::Authorized(ApprovalAuthorization::WorkspaceConfigured {
                grant: configured,
            })
        );
        assert_eq!(requires_decision, ApprovalCheck::RequiresDecision);
    }

    #[test]
    fn applying_decisions_preserves_the_authorizing_approval() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let lifetimes = offered_lifetimes();
        let once_subject = subject("write-1", "write_file");
        let run_subject = subject("edit-1", "edit_file");
        let denied_subject = subject("glob-1", "glob");
        let once_grant = once_subject.grant(ApprovalLifetime::Invocation);
        let run_grant = run_subject.grant(ApprovalLifetime::Run);
        let once_decision = ApprovalDecision::Grant(once_grant.clone());
        let run_decision = ApprovalDecision::Grant(run_grant.clone());
        let deny_decision = ApprovalDecision::Deny {
            reason: "not now".to_string(),
        };

        // Act
        let once = gate.apply_decision(&lifetimes, approval_id(), &once_decision, &once_subject);
        let run = gate.apply_decision(&lifetimes, approval_id(), &run_decision, &run_subject);
        let denied =
            gate.apply_decision(&lifetimes, approval_id(), &deny_decision, &denied_subject);

        // Assert
        assert_eq!(
            once,
            ApprovalOutcome::Authorized(ApprovalAuthorization::Granted {
                approval_id: approval_id(),
                grant: once_grant,
            })
        );
        assert_eq!(
            run,
            ApprovalOutcome::Authorized(ApprovalAuthorization::Granted {
                approval_id: approval_id(),
                grant: run_grant.clone(),
            })
        );
        assert_eq!(
            gate.check(
                ApprovalRequirement::Required,
                &subject("edit-2", "edit_file")
            ),
            ApprovalCheck::Authorized(ApprovalAuthorization::Granted {
                approval_id: approval_id(),
                grant: run_grant,
            })
        );
        assert_eq!(
            denied,
            ApprovalOutcome::Denied {
                reason: "not now".to_string(),
            }
        );
    }

    #[test]
    fn applying_a_grant_rejects_unoffered_lifetimes_and_mismatched_subjects() {
        // Arrange
        let mut gate = ApprovalGate::new();
        let requested = subject("write-1", "write_file");
        let other = subject("edit-1", "edit_file");
        let workspace_decision =
            ApprovalDecision::Grant(requested.grant(ApprovalLifetime::Workspace));
        let other_decision = ApprovalDecision::Grant(other.grant(ApprovalLifetime::Invocation));

        // Act
        let unavailable = gate.apply_decision(
            &offered_lifetimes(),
            approval_id(),
            &workspace_decision,
            &requested,
        );
        let mismatched = gate.apply_decision(
            &offered_lifetimes(),
            approval_id(),
            &other_decision,
            &requested,
        );

        // Assert
        assert!(matches!(
            unavailable,
            ApprovalOutcome::InvalidGrant { reason }
                if reason.contains("lifetime")
        ));
        assert!(matches!(
            mismatched,
            ApprovalOutcome::InvalidGrant { reason }
                if reason.contains("subject")
        ));
    }

    #[tokio::test]
    async fn request_emits_the_call_details_and_returns_the_decision() {
        // Arrange
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        let sink = EventSink::new(events_tx);
        let cancel = CancellationToken::new();
        let payload = json!({ "file": "test.txt", "contents": "test" });
        let offered_lifetimes = offered_lifetimes();
        let subject = subject("write-1", "write_file");

        // Act
        let (request_result, ()) = tokio::join!(
            request_approval(
                offered_lifetimes.clone(),
                &payload,
                subject.clone(),
                &sink,
                &cancel,
            ),
            async {
                let event = events_rx.recv().await.unwrap();

                let AgentEvent::ApprovalRequest {
                    available_lifetimes,
                    input,
                    respond_to,
                    subject: emitted_subject,
                } = event
                else {
                    panic!("expected ApprovalRequest");
                };

                assert_eq!(available_lifetimes, offered_lifetimes);
                assert_eq!(input, payload);
                assert_eq!(emitted_subject, subject);

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
            offered_lifetimes(),
            &json!({ "path": "test.txt" }),
            subject("write-1", "write_file"),
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
            request_approval(
                offered_lifetimes(),
                &payload,
                subject("write-1", "write_file"),
                &sink,
                &cancel,
            ),
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
            offered_lifetimes(),
            &json!({ "path": "test.txt" }),
            subject("write-1", "write_file"),
            &sink,
            &cancel,
        )
        .await;

        // Assert
        assert!(matches!(result, Err(AgentExit::Disconnected)));
    }
}
