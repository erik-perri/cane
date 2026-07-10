use serde_json::Value;

mod read_file;

pub use read_file::ReadFileTool;

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
    async fn execute(&self, input: Value) -> Result<String, String>;
}

/// Look up a tool by name and run it. An unknown name is an error tool
/// result, not a panic.
pub async fn dispatch(tools: &[Box<dyn Tool>], name: &str, input: Value) -> Result<String, String> {
    match tools.iter().find(|t| t.definition().name == name) {
        Some(tool) => tool.execute(input).await,
        None => Err(format!("unknown tool: {name}")),
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

        async fn execute(&self, input: Value) -> Result<String, String> {
            Ok(format!("stub ran with {input}"))
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
        assert_eq!(result, Err("unknown tool: write_file".to_string()));
    }
}
