use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

mod edit_file;
mod file_discovery;
mod glob;
mod grep;
mod path_display;
mod read_file;
mod shell;
mod update_checklist;
mod write_file;

use crate::Checklist;
use crate::Workspace;
use crate::command::{CommandEnvironmentConfig, CommandExecutor};
use crate::journal::{CapabilityAuthorizationSource, ToolExecutionCompleted, ToolExecutionStarted};
use crate::protocol::{ApprovalLifetime, ApprovalRequirement, EventSink, NamedCapability};
use edit_file::EditFileTool;
use glob::GlobTool;
use grep::GrepTool;
use read_file::ReadFileTool;
pub use shell::ShellIntegration;
pub(crate) use shell::ShellIntegrations;
use shell::ShellTool;
pub(crate) use update_checklist::UpdateChecklistTool;
use write_file::WriteFileTool;

pub(crate) struct ToolSet {
    tool_definitions: Vec<ToolDefinition>,
    tools: Vec<Box<dyn Tool>>,
}

pub(crate) struct ShellToolConfig {
    pub(crate) environment: CommandEnvironmentConfig,
    pub(crate) events: EventSink,
    pub(crate) executor: Arc<dyn CommandExecutor>,
    pub(crate) integrations: ShellIntegrations,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapabilityRequest {
    pub(crate) available_lifetimes: &'static [ApprovalLifetime],
    pub(crate) capability: NamedCapability,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AuthorizedCapability {
    pub(crate) capability: NamedCapability,
    pub(crate) source: CapabilityAuthorizationSource,
}

#[derive(Debug, PartialEq)]
pub(crate) enum ToolExecutionError {
    Cancelled,
    ToolError(String),
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ToolExecutionOutput {
    content: String,
    execution: Option<ToolExecutionCompleted>,
    checklist_update: Option<Checklist>,
}

impl ToolExecutionOutput {
    pub(crate) fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            execution: None,
            checklist_update: None,
        }
    }

    pub(crate) fn completed(content: impl Into<String>, execution: ToolExecutionCompleted) -> Self {
        Self {
            content: content.into(),
            execution: Some(execution),
            checklist_update: None,
        }
    }

    pub(crate) fn checklist_update(checklist: Checklist) -> Self {
        Self {
            content: "Checklist updated.".to_string(),
            execution: None,
            checklist_update: Some(checklist),
        }
    }

    pub(crate) fn into_parts(self) -> (String, Option<ToolExecutionCompleted>, Option<Checklist>) {
        (self.content, self.execution, self.checklist_update)
    }
}

impl From<String> for ToolExecutionError {
    fn from(error: String) -> Self {
        Self::ToolError(error)
    }
}

impl ToolSet {
    pub(crate) fn new(workspace: Arc<Workspace>, shell: Option<ShellToolConfig>) -> Self {
        let mut tools: Vec<Box<dyn Tool>> = vec![
            Box::new(EditFileTool::new(Arc::clone(&workspace))),
            Box::new(GlobTool::new(Arc::clone(&workspace))),
            Box::new(GrepTool::new(Arc::clone(&workspace))),
            Box::new(ReadFileTool::new(Arc::clone(&workspace))),
            Box::new(UpdateChecklistTool),
            Box::new(WriteFileTool::new(Arc::clone(&workspace))),
        ];
        if let Some(shell) = shell {
            tools.push(Box::new(ShellTool::new(
                workspace,
                shell.integrations,
                shell.environment,
                shell.events,
                shell.executor,
            )));
        }

        Self::from_unsorted_tools(tools)
    }

    pub(crate) fn definitions(&self) -> &[ToolDefinition] {
        &self.tool_definitions
    }

    pub(crate) fn locate(&self, name: &str) -> Result<&dyn Tool, String> {
        self.tool_definitions
            .iter()
            .zip(&self.tools)
            .find(|(definition, _)| definition.name == name)
            .map(|(_, tool)| tool.as_ref())
            .ok_or_else(|| format!("unknown tool: `{name}`"))
    }

    #[cfg(test)]
    pub(crate) fn from_tools(tools: Vec<Box<dyn Tool>>) -> Self {
        Self::from_unsorted_tools(tools)
    }

    fn from_unsorted_tools(tools: Vec<Box<dyn Tool>>) -> Self {
        let mut entries: Vec<_> = tools
            .into_iter()
            .map(|tool| (tool.definition(), tool))
            .collect();
        entries.sort_by(|(left, _), (right, _)| left.name.cmp(&right.name));

        for pair in entries.windows(2) {
            assert_ne!(
                pair[0].0.name, pair[1].0.name,
                "duplicate built-in tool name"
            );
        }

        let (tool_definitions, tools) = entries.into_iter().unzip();

        Self {
            tool_definitions,
            tools,
        }
    }
}

/// A tool the model can call.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[async_trait::async_trait]
pub(crate) trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    async fn prepare(&self, input: Value) -> Result<Box<dyn PreparedInvocation>, String>;
}

#[async_trait]
pub(crate) trait PreparedInvocation: Send {
    fn approval_requirement(&self) -> ApprovalRequirement;

    fn capability_request(&self) -> Option<CapabilityRequest> {
        None
    }

    fn authorize_capability(&mut self, _authorization: AuthorizedCapability) -> Result<(), String> {
        Err("prepared invocation did not request a capability".to_string())
    }

    /// Grant lifetimes this invocation may receive.
    ///
    /// Denial is always available and may include a user-provided reason.
    fn available_grant_lifetimes(&self) -> &'static [ApprovalLifetime] {
        &[ApprovalLifetime::Invocation, ApprovalLifetime::Run]
    }

    fn execution_started(&self) -> Option<ToolExecutionStarted> {
        None
    }

    async fn execute(
        self: Box<Self>,
        cancel: CancellationToken,
    ) -> Result<ToolExecutionOutput, ToolExecutionError>;
}

/// Largest file the file tools will load into memory.
const MAX_FILE_SIZE_MIB: u64 = 10;
const MAX_FILE_SIZE_BYTES: u64 = MAX_FILE_SIZE_MIB * 1024 * 1024;

fn invalid_input(tool: &str, reason: impl std::fmt::Display) -> String {
    format!("invalid {tool} input: {reason}")
}

/// Reject resolved paths inside a `.git` directory.
fn reject_git_directory(
    tool: &str,
    workspace: &Workspace,
    resolved_path: &std::path::Path,
) -> Result<(), String> {
    let relative = resolved_path
        .strip_prefix(workspace.root())
        .expect("resolve returns paths inside the workspace root");

    if relative
        .components()
        .any(|component| component.as_os_str() == ".git")
    {
        return Err(invalid_input(
            tool,
            "path must not be inside a `.git` directory",
        ));
    }

    Ok(())
}

fn operation_failed(operation: &str, path: &str, error: impl std::fmt::Display) -> String {
    format!("failed to {operation} `{path}`: {error}")
}

fn background_task_failed(operation: &str, path: &str, error: impl std::fmt::Display) -> String {
    operation_failed(
        operation,
        path,
        format_args!("background task failed: {error}"),
    )
}

#[cfg(test)]
#[async_trait]
trait ToolTestExt: Tool {
    async fn execute(&self, input: Value) -> Result<String, String>;
}

#[cfg(test)]
#[async_trait]
impl<T> ToolTestExt for T
where
    T: Tool + ?Sized,
{
    async fn execute(&self, input: Value) -> Result<String, String> {
        let invocation = self.prepare(input).await?;

        match invocation.execute(CancellationToken::new()).await {
            Ok(output) => Ok(output.into_parts().0),
            Err(ToolExecutionError::ToolError(error)) => Err(error),
            Err(ToolExecutionError::Cancelled) => {
                unreachable!("a fresh test cancellation token cannot be cancelled")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    struct NamedTestTool {
        definition_calls: Arc<AtomicUsize>,
        name: &'static str,
    }

    #[async_trait::async_trait]
    impl Tool for NamedTestTool {
        fn definition(&self) -> ToolDefinition {
            self.definition_calls.fetch_add(1, Ordering::SeqCst);
            ToolDefinition {
                name: self.name.to_string(),
                description: String::new(),
                input_schema: serde_json::json!({ "type": "object" }),
            }
        }

        async fn prepare(&self, _input: Value) -> Result<Box<dyn PreparedInvocation>, String> {
            Err("test tool does not prepare invocations".to_string())
        }
    }

    #[test]
    fn locate_finds_a_registered_tool_by_name() {
        // Arrange
        let dir = tempdir().unwrap();
        let workspace = Workspace::new(dir.path().into()).unwrap();
        let tool_set = ToolSet::new(Arc::new(workspace), None);

        // Act
        let tool = tool_set.locate("read_file").unwrap();

        // Assert
        assert_eq!(tool.definition().name, "read_file");
    }

    #[test]
    fn locate_returns_an_error_for_an_unknown_name() {
        // Arrange
        let dir = tempdir().unwrap();
        let workspace = Workspace::new(dir.path().into()).unwrap();
        let tool_set = ToolSet::new(Arc::new(workspace), None);

        // Act
        let tool = tool_set.locate("what_tool").err().unwrap();

        // Assert
        assert_eq!("unknown tool: `what_tool`", tool);
    }

    #[test]
    fn built_in_definitions_are_alphabetically_sorted_and_include_update_checklist() {
        let dir = tempdir().unwrap();
        let workspace = Workspace::new(dir.path().into()).unwrap();
        let tool_set = ToolSet::new(Arc::new(workspace), None);

        let names: Vec<_> = tool_set
            .definitions()
            .iter()
            .map(|definition| definition.name.as_str())
            .collect();

        assert_eq!(
            names,
            vec![
                "edit_file",
                "glob",
                "grep",
                "read_file",
                "update_checklist",
                "write_file",
            ]
        );
        assert!(tool_set.locate("update_checklist").is_ok());
    }

    #[test]
    fn catalog_construction_builds_each_definition_once_and_keeps_tools_aligned() {
        let definition_calls = Arc::new(AtomicUsize::new(0));
        let tool_set = ToolSet::from_tools(vec![
            Box::new(NamedTestTool {
                definition_calls: Arc::clone(&definition_calls),
                name: "zeta",
            }),
            Box::new(NamedTestTool {
                definition_calls: Arc::clone(&definition_calls),
                name: "alpha",
            }),
        ]);

        assert_eq!(definition_calls.load(Ordering::SeqCst), 2);
        assert_eq!(tool_set.definitions()[0].name, "alpha");
        assert_eq!(tool_set.definitions()[1].name, "zeta");
        assert!(tool_set.locate("alpha").is_ok());
        assert!(tool_set.locate("zeta").is_ok());
        assert_eq!(definition_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    #[should_panic(expected = "duplicate built-in tool name")]
    fn duplicate_tool_names_are_a_programming_error() {
        let definition_calls = Arc::new(AtomicUsize::new(0));

        let _ = ToolSet::from_tools(vec![
            Box::new(NamedTestTool {
                definition_calls: Arc::clone(&definition_calls),
                name: "duplicate",
            }),
            Box::new(NamedTestTool {
                definition_calls,
                name: "duplicate",
            }),
        ]);
    }

    #[test]
    fn operation_errors_follow_the_shared_message_format() {
        // Arrange
        let expected = [
            "invalid read_file input: path must not be empty",
            "failed to read `notes.txt`: permission denied",
            "failed to write `notes.txt`: background task failed: task cancelled",
        ];

        // Act
        let actual = [
            invalid_input("read_file", "path must not be empty"),
            operation_failed("read", "notes.txt", "permission denied"),
            background_task_failed("write", "notes.txt", "task cancelled"),
        ];

        // Assert
        assert_eq!(actual, expected);
    }
}
