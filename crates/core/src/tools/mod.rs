use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

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
            .collect::<Vec<ToolDefinition>>();

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

    async fn execute(self: Box<Self>) -> Result<String, String>;
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

/// Look up a tool by name and run it. An unknown name is an error tool
/// result, not a panic.
pub async fn dispatch(tools: &[Box<dyn Tool>], name: &str, input: Value) -> Result<String, String> {
    match tools.iter().find(|t| t.definition().name == name) {
        Some(tool) => tool.prepare(input).await?.execute().await,
        None => Err(format!("unknown tool: `{name}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct StubTool;

    #[async_trait::async_trait]
    impl Tool for StubTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "stub".to_string(),
                description: "A stub tool for dispatch tests.".to_string(),
                input_schema: json!({ "type": "object" }),
            }
        }

        async fn prepare(&self, input: Value) -> Result<Box<dyn PreparedInvocation>, String> {
            Ok(Box::new(StubInvocation(input)))
        }
    }

    struct StubInvocation(Value);

    #[async_trait::async_trait]
    impl PreparedInvocation for StubInvocation {
        fn approval_requirement(&self) -> ApprovalRequirement {
            ApprovalRequirement::None
        }

        async fn execute(self: Box<Self>) -> Result<String, String> {
            Ok(format!("stub ran with {}", self.0))
        }
    }

    #[tokio::test]
    async fn dispatch_runs_the_tool_matching_the_name() {
        // Arrange
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(StubTool)];

        // Act
        let result = dispatch(&tools, "stub", json!({ "key": "value" })).await;

        // Assert
        assert_eq!(result, Ok(r#"stub ran with {"key":"value"}"#.to_string()));
    }

    #[tokio::test]
    async fn dispatch_returns_an_error_for_an_unknown_tool_name() {
        // Arrange
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(StubTool)];

        // Act
        let result = dispatch(&tools, "write_file", json!({})).await;

        // Assert
        assert_eq!(result, Err("unknown tool: `write_file`".to_string()));
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
