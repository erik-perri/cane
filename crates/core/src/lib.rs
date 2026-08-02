mod agent;
mod approval;
mod checklist;
pub mod command;
pub mod journal;
mod message;
mod protocol;
mod provider;
mod session;
mod tools;
mod workspace;
mod workspace_consents;

pub use agent::{
    AgentHandle, AgentShellConfig, AgentShellConfigError, AgentStartError, spawn_agent,
    spawn_agent_with_shell,
};
pub use checklist::{Checklist, ChecklistStep, ChecklistStepStatus};
pub use message::{ContentBlock, Message, Role, StopReason, ToolInput, ToolResultData};
pub use protocol::{
    AgentCommand, AgentEvent, ApprovalDecision, ApprovalGrant, ApprovalLifetime, ApprovalMatcher,
    ApprovalScope, ApprovalSubject, CapabilityKind, NamedCapability, ShutdownReason, TurnOutcome,
};
pub use provider::{
    ModelTurn, ModelUsage, ProviderAdapter, ProviderConfig, ProviderDescriptor, ReportedCost,
};
pub use session::SessionConfig;
pub use tools::{ShellIntegration, ToolDefinition};
pub use workspace::Workspace;
pub use workspace_consents::{
    MAX_WORKSPACE_CAPABILITY_CONSENTS, WORKSPACE_CAPABILITY_CONSENTS_DOCUMENT,
    WORKSPACE_CAPABILITY_CONSENTS_SCHEMA, WorkspaceCapabilityConsent,
    WorkspaceCapabilityConsentDocument, WorkspaceCapabilityConsentStore,
    WorkspaceConsentDocumentError,
};
