mod agent;
mod approval;
pub mod command;
pub mod journal;
mod message;
mod protocol;
mod provider;
mod session;
mod tools;
mod workspace;

pub use agent::{
    AgentHandle, AgentShellConfig, AgentShellConfigError, AgentStartError, spawn_agent,
    spawn_agent_with_shell,
};
pub use message::{ContentBlock, Message, Role, StopReason, ToolInput, ToolResultData};
pub use protocol::{
    AgentCommand, AgentEvent, ApprovalDecision, ApprovalGrant, ApprovalLifetime, ApprovalMatcher,
    ApprovalScope, ApprovalSubject, NamedCapability, ShutdownReason, TurnOutcome,
};
pub use provider::{
    ModelTurn, ModelUsage, ProviderAdapter, ProviderConfig, ProviderDescriptor, ReportedCost,
};
pub use session::SessionConfig;
pub use tools::ToolDefinition;
pub use workspace::Workspace;
