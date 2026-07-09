use serde_json::Value;

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

pub struct FileTool {}

#[async_trait::async_trait]
impl Tool for FileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Reads a file from the local filesystem. Call this whenever the user asks about a file's contents or you need to read a file for context. Returns the file as text.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start from."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return."
                    },
                },
                "required": ["path"]
            })
        }
    }

    async fn execute(&self, input: Value) -> Result<String, String> {
        let input = input.as_object().ok_or("invalid input")?;
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or("missing 'path' field")?;
        let offset = input.get("offset").and_then(Value::as_u64).unwrap_or(1);
        let limit = input.get("limit").and_then(Value::as_u64).unwrap_or(0);

        let path = path.to_owned();
        tokio::task::spawn_blocking(move || read_lines_from_file(&path, offset, limit))
            .await
            .map_err(|e| e.to_string())?
    }
}

/// `offset` is a 1-based starting line (0 is treated as 1); `limit` of 0
/// means no limit.
fn read_lines_from_file(path: &str, offset: u64, limit: u64) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&bytes);

    let lines = text.lines().skip(offset.saturating_sub(1) as usize);
    let selected: Vec<&str> = if limit > 0 {
        lines.take(limit as usize).collect()
    } else {
        lines.collect()
    };

    Ok(selected.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn temp_file_with(contents: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(contents).unwrap();
        file
    }

    fn path_of(file: &NamedTempFile) -> &str {
        file.path().to_str().unwrap()
    }

    #[tokio::test]
    async fn execute_reads_the_whole_file_by_default() {
        // Arrange
        let file = temp_file_with(b"line one\nline two\nline three");

        // Act
        let result = FileTool {}.execute(json!({ "path": path_of(&file) })).await;

        // Assert
        assert_eq!(result, Ok("line one\nline two\nline three".to_string()));
    }

    #[tokio::test]
    async fn execute_applies_line_offset_and_limit() {
        // Arrange
        let file = temp_file_with(b"one\ntwo\nthree\nfour\nfive");

        // Act
        let result = FileTool {}
            .execute(json!({ "path": path_of(&file), "offset": 2, "limit": 3 }))
            .await;

        // Assert
        assert_eq!(result, Ok("two\nthree\nfour".to_string()));
    }

    #[tokio::test]
    async fn execute_treats_offset_as_one_based() {
        // Arrange
        let file = temp_file_with(b"first\nsecond");

        // Act
        let result = FileTool {}
            .execute(json!({ "path": path_of(&file), "offset": 1 }))
            .await;

        // Assert
        assert_eq!(result, Ok("first\nsecond".to_string()));
    }

    #[tokio::test]
    async fn execute_returns_remaining_lines_when_limit_exceeds_file_length() {
        // Arrange
        let file = temp_file_with(b"one\ntwo");

        // Act
        let result = FileTool {}
            .execute(json!({ "path": path_of(&file), "limit": 100 }))
            .await;

        // Assert
        assert_eq!(result, Ok("one\ntwo".to_string()));
    }

    #[tokio::test]
    async fn execute_returns_empty_string_when_offset_is_past_the_last_line() {
        // Arrange
        let file = temp_file_with(b"one\ntwo");

        // Act
        let result = FileTool {}
            .execute(json!({ "path": path_of(&file), "offset": 100 }))
            .await;

        // Assert
        assert_eq!(result, Ok(String::new()));
    }

    #[tokio::test]
    async fn execute_replaces_invalid_utf8_instead_of_failing() {
        // Arrange
        let file = temp_file_with(b"ok \xff\xfe bytes");

        // Act
        let result = FileTool {}.execute(json!({ "path": path_of(&file) })).await;

        // Assert
        assert_eq!(result, Ok("ok \u{FFFD}\u{FFFD} bytes".to_string()));
    }

    #[tokio::test]
    async fn execute_rejects_input_without_a_path() {
        // Act
        let result = FileTool {}.execute(json!({ "offset": 1 })).await;

        // Assert
        assert_eq!(result, Err("missing 'path' field".to_string()));
    }

    #[tokio::test]
    async fn execute_rejects_non_object_input() {
        // Act
        let result = FileTool {}.execute(json!("just a string")).await;

        // Assert
        assert_eq!(result, Err("invalid input".to_string()));
    }

    #[tokio::test]
    async fn execute_reports_an_error_for_a_missing_file() {
        // Act
        let result = FileTool {}
            .execute(json!({ "path": "/definitely/not/a/real/file" }))
            .await;

        // Assert
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn dispatch_runs_the_tool_matching_the_name() {
        // Arrange
        let file = temp_file_with(b"dispatched");
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(FileTool {})];

        // Act
        let result = dispatch(&tools, "read_file", json!({ "path": path_of(&file) })).await;

        // Assert
        assert_eq!(result, Ok("dispatched".to_string()));
    }

    #[tokio::test]
    async fn dispatch_returns_an_error_for_an_unknown_tool_name() {
        // Arrange
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(FileTool {})];

        // Act
        let result = dispatch(&tools, "write_file", json!({})).await;

        // Assert
        assert_eq!(result, Err("unknown tool: write_file".to_string()));
    }
}
