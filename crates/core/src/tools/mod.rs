use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

mod edit_file;
mod read_file;
mod write_file;

use crate::Workspace;
use crate::protocol::ApprovalRequirement;
pub use edit_file::EditFileTool;
pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;

pub struct ToolSet {
    tool_definitions: Vec<ToolDefinition>,
    tools: Vec<Box<dyn Tool>>,
}

pub enum ToolExecutionError {
    Cancelled,
    ToolError(String),
}

impl From<String> for ToolExecutionError {
    fn from(error: String) -> Self {
        Self::ToolError(error)
    }
}

impl ToolSet {
    pub fn new(workspace: Arc<Workspace>) -> Self {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(EditFileTool::new(Arc::clone(&workspace))),
            Box::new(ReadFileTool::new(Arc::clone(&workspace))),
            Box::new(WriteFileTool::new(Arc::clone(&workspace))),
        ];

        let tool_definitions = tools
            .iter()
            .map(|tool| tool.definition())
            .collect();

        Self {
            tool_definitions,
            tools,
        }
    }

    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.tool_definitions
    }

    pub fn locate(&self, name: &str) -> Result<&dyn Tool, String> {
        self.tools
            .iter()
            .map(Box::as_ref)
            .find(|tool| tool.definition().name == name)
            .ok_or_else(|| format!("unknown tool: `{name}`"))
    }

    #[cfg(test)]
    pub(crate) fn from_tools(tools: Vec<Box<dyn Tool>>) -> Self {
        let tool_definitions = tools
            .iter()
            .map(|tool| tool.definition())
            .collect();

        Self {
            tool_definitions,
            tools,
        }
    }
}

/// A tool the model can call.
#[derive(Clone, Debug)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    async fn prepare(&self, input: Value) -> Result<Box<dyn PreparedInvocation>, String>;
}

#[async_trait]
pub trait PreparedInvocation: Send {
    fn approval_requirement(&self) -> ApprovalRequirement;

    async fn execute(
        self: Box<Self>,
        cancel: CancellationToken,
    ) -> Result<String, ToolExecutionError>;
}

/// Largest file the file tools will load into memory.
const MAX_FILE_SIZE_MIB: u64 = 10;
const MAX_FILE_SIZE_BYTES: u64 = MAX_FILE_SIZE_MIB * 1024 * 1024;

fn invalid_input(tool: &str, reason: impl std::fmt::Display) -> String {
    format!("invalid {tool} input: {reason}")
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
pub(crate) trait ToolTestExt: Tool {
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
            Ok(output) => Ok(output),
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
    use tempfile::tempdir;

    #[test]
    fn locate_finds_a_registered_tool_by_name() {
        // Arrange
        let dir = tempdir().unwrap();
        let workspace = Workspace::new(dir.path().into()).unwrap();
        let tool_set = ToolSet::new(Arc::new(workspace));

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
        let tool_set = ToolSet::new(Arc::new(workspace));

        // Act
        let tool = tool_set.locate("what_tool").err().unwrap();

        // Assert
        assert_eq!("unknown tool: `what_tool`", tool);
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
