use super::{
    ApprovalDecided, ApprovalId, ApprovalRequested, ErrorDetail, JournalApprovalDecision,
    JournalEntry, JournalError, MessageAdded, ProviderRoundCancelled, ProviderRoundCompleted,
    ProviderRoundFailed, ProviderRoundId, ProviderRoundStarted, RunEndReason, RunEnded, RunId,
    SessionId, SessionJournal, ToolAuthorization, ToolCancelled, ToolCompleted, ToolFailed,
    ToolRejected, ToolStarted, TurnAbortOutcome, TurnAborted, TurnCommitOutcome, TurnCommitted,
    TurnId, TurnStarted,
};
use crate::{ApprovalLifetime, ApprovalSubject, Message, ModelTurn, ProviderDescriptor};
use std::path::Path;

pub struct RunJournal {
    journal: SessionJournal,
    model: String,
    provider: ProviderDescriptor,
    run_id: RunId,
}

impl RunJournal {
    pub fn new(
        journal: SessionJournal,
        model: String,
        provider: ProviderDescriptor,
        run_id: RunId,
    ) -> Self {
        Self {
            journal,
            model,
            provider,
            run_id,
        }
    }

    pub async fn abort_turn(
        &mut self,
        turn_id: TurnId,
        outcome: TurnAbortOutcome,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::TurnAborted(TurnAborted { outcome, turn_id }))
            .await?;
        Ok(())
    }

    pub async fn approval_decided(
        &mut self,
        approval_id: ApprovalId,
        decision: JournalApprovalDecision,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ApprovalDecided(ApprovalDecided {
                approval_id,
                decision,
            }))
            .await?;
        Ok(())
    }

    pub async fn approval_requested(
        &mut self,
        approval_id: ApprovalId,
        available_lifetimes: Vec<ApprovalLifetime>,
        subject: ApprovalSubject,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ApprovalRequested(ApprovalRequested {
                approval_id,
                available_lifetimes,
                subject,
                turn_id,
            }))
            .await?;
        Ok(())
    }

    pub async fn commit_turn(
        &mut self,
        turn_id: TurnId,
        outcome: TurnCommitOutcome,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::TurnCommitted(TurnCommitted {
                outcome,
                turn_id,
            }))
            .await?;
        Ok(())
    }

    pub async fn end_run(&mut self, reason: RunEndReason) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::RunEnded(RunEnded {
                reason,
                run_id: self.run_id,
            }))
            .await?;
        Ok(())
    }

    pub async fn message_added(
        &mut self,
        message: &Message,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::MessageAdded(MessageAdded {
                message: message.clone(),
                run_id: self.run_id,
                turn_id,
            }))
            .await?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn inject_flush_failure(&mut self, failure: super::InjectedFlushFailure) {
        self.journal.inject_flush_failure(failure);
    }

    pub fn path(&self) -> &Path {
        self.journal.path()
    }

    pub async fn provider_round_cancelled(
        &mut self,
        latency_ms: u64,
        provider_round_id: ProviderRoundId,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ProviderRoundCancelled(
                ProviderRoundCancelled {
                    latency_ms,
                    provider_round_id,
                    run_id: self.run_id,
                    turn_id,
                },
            ))
            .await?;
        Ok(())
    }

    pub async fn provider_round_completed(
        &mut self,
        latency_ms: u64,
        provider_round_id: ProviderRoundId,
        turn: &ModelTurn,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ProviderRoundCompleted(
                ProviderRoundCompleted {
                    latency_ms,
                    provider_cost: turn.provider_cost.clone(),
                    provider_round_id,
                    request_id: turn.request_id.clone(),
                    run_id: self.run_id,
                    stop_reason: turn.stop_reason.clone(),
                    turn_id,
                    usage: turn.usage.clone(),
                },
            ))
            .await?;
        Ok(())
    }

    pub async fn provider_round_failed(
        &mut self,
        error: ErrorDetail,
        latency_ms: u64,
        provider_round_id: ProviderRoundId,
        request_id: Option<String>,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ProviderRoundFailed(ProviderRoundFailed {
                error,
                latency_ms,
                provider_round_id,
                request_id,
                run_id: self.run_id,
                turn_id,
            }))
            .await?;
        Ok(())
    }

    pub async fn provider_round_started(
        &mut self,
        turn_id: TurnId,
    ) -> Result<ProviderRoundId, JournalError> {
        let provider_round_id = ProviderRoundId::generate();
        self.journal
            .append(JournalEntry::ProviderRoundStarted(ProviderRoundStarted {
                model: self.model.clone(),
                provider: self.provider.clone(),
                provider_round_id,
                run_id: self.run_id,
                turn_id,
            }))
            .await?;
        Ok(provider_round_id)
    }

    pub fn session_id(&self) -> &SessionId {
        self.journal.session_id()
    }

    pub async fn start_turn(&mut self, message: &Message) -> Result<TurnId, JournalError> {
        let turn_id = TurnId::generate();
        self.journal
            .append(JournalEntry::TurnStarted(TurnStarted {
                run_id: self.run_id,
                turn_id,
            }))
            .await?;
        self.message_added(message, turn_id).await?;
        Ok(turn_id)
    }

    pub async fn tool_cancelled(
        &mut self,
        duration_ms: u64,
        tool_call_id: &str,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ToolCancelled(ToolCancelled {
                duration_ms,
                tool_call_id: tool_call_id.to_string(),
                turn_id,
            }))
            .await?;
        Ok(())
    }

    pub async fn tool_completed(
        &mut self,
        duration_ms: u64,
        execution: Option<super::ToolExecutionCompleted>,
        tool_call_id: &str,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ToolCompleted(ToolCompleted {
                duration_ms,
                execution,
                tool_call_id: tool_call_id.to_string(),
                turn_id,
            }))
            .await?;
        Ok(())
    }

    pub async fn tool_failed(
        &mut self,
        duration_ms: u64,
        error_category: &str,
        tool_call_id: &str,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ToolFailed(ToolFailed {
                duration_ms,
                error_category: error_category.to_string(),
                tool_call_id: tool_call_id.to_string(),
                turn_id,
            }))
            .await?;
        Ok(())
    }

    pub async fn tool_rejected(
        &mut self,
        error_category: &str,
        tool_call_id: &str,
        tool_name: &str,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ToolRejected(ToolRejected {
                error_category: error_category.to_string(),
                tool_call_id: tool_call_id.to_string(),
                tool_name: tool_name.to_string(),
                turn_id,
            }))
            .await?;
        Ok(())
    }

    pub async fn tool_started(
        &mut self,
        authorization: ToolAuthorization,
        execution: Option<super::ToolExecutionStarted>,
        tool_call_id: &str,
        tool_name: &str,
        turn_id: TurnId,
    ) -> Result<(), JournalError> {
        self.journal
            .append(JournalEntry::ToolStarted(ToolStarted {
                authorization,
                execution,
                tool_call_id: tool_call_id.to_string(),
                tool_name: tool_name.to_string(),
                turn_id,
            }))
            .await?;
        Ok(())
    }
}
