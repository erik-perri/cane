use crate::tools::{Tool, ToolDefinition};
use crate::workspace::Workspace;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

const DEFAULT_READ_FILE_LIMIT: u64 = 2_000;
const MAX_READ_FILE_LIMIT: u64 = 2_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileInput {
    path: String,
    #[serde(default = "default_offset")]
    offset: u64,
    #[serde(default = "default_read_file_limit")]
    limit: u64,
}

fn default_offset() -> u64 {
    1
}

fn default_read_file_limit() -> u64 {
    DEFAULT_READ_FILE_LIMIT
}

pub struct ReadFileTool {
    workspace: Arc<Workspace>,
}

impl ReadFileTool {
    pub fn new(workspace: Arc<Workspace>) -> Self {
        Self { workspace }
    }
}

#[async_trait::async_trait]
impl Tool for ReadFileTool {
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
                        "description": "1-based line number to start from. Defaults to 1.",
                        "minimum": 1
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return. Defaults to 2000.",
                        "minimum": 1,
                        "maximum": MAX_READ_FILE_LIMIT
                    },
                },
                "required": ["path"],
                "additionalProperties": false
            })
        }
    }

    async fn execute(&self, input: Value) -> Result<String, String> {
        let input: ReadFileInput = serde_json::from_value(input)
            .map_err(|error| format!("invalid read_file input: {error}"))?;

        let resolved_path = self
            .workspace
            .resolve(&input.path)
            .map_err(|error| format!("invalid read_file input: {error}"))?;

        if input.offset == 0 {
            return Err("invalid read_file input: offset must be at least 1".to_string());
        }
        if input.limit == 0 || input.limit > MAX_READ_FILE_LIMIT {
            return Err(format!(
                "invalid read_file input: limit must be between 1 and {MAX_READ_FILE_LIMIT}"
            ));
        }

        tokio::task::spawn_blocking(move || {
            read_lines_from_file(&resolved_path, input.offset, input.limit)
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

/// `offset` and `limit` have already been validated by `ReadFileInput`.
fn read_lines_from_file(path: &Path, offset: u64, limit: u64) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&bytes);

    let lines = text.lines().skip(offset.saturating_sub(1) as usize);
    let selected: Vec<&str> = lines.take(limit as usize).collect();

    Ok(selected.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Tool;
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

    fn read_file_tool() -> ReadFileTool {
        ReadFileTool::new(Arc::new(Workspace::new(std::env::temp_dir()).unwrap()))
    }

    #[tokio::test]
    async fn execute_reads_up_to_the_default_line_limit() {
        // Arrange
        let file = temp_file_with(b"line one\nline two\nline three");

        // Act
        let result = read_file_tool()
            .execute(json!({ "path": path_of(&file) }))
            .await;

        // Assert
        assert_eq!(result, Ok("line one\nline two\nline three".to_string()));
    }

    #[tokio::test]
    async fn execute_applies_line_offset_and_limit() {
        // Arrange
        let file = temp_file_with(b"one\ntwo\nthree\nfour\nfive");

        // Act
        let result = read_file_tool()
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
        let result = read_file_tool()
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
        let result = read_file_tool()
            .execute(json!({ "path": path_of(&file), "limit": 100 }))
            .await;

        // Assert
        assert_eq!(result, Ok("one\ntwo".to_string()));
    }

    #[tokio::test]
    async fn execute_applies_the_default_line_limit() {
        // Arrange
        let contents = (0..=DEFAULT_READ_FILE_LIMIT)
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let file = temp_file_with(contents.as_bytes());

        // Act
        let output = read_file_tool()
            .execute(json!({ "path": path_of(&file) }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output.lines().count(), DEFAULT_READ_FILE_LIMIT as usize);
        assert_eq!(output.lines().last(), Some("1999"));
    }

    #[tokio::test]
    async fn execute_returns_empty_string_when_offset_is_past_the_last_line() {
        // Arrange
        let file = temp_file_with(b"one\ntwo");

        // Act
        let result = read_file_tool()
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
        let result = read_file_tool()
            .execute(json!({ "path": path_of(&file) }))
            .await;

        // Assert
        assert_eq!(result, Ok("ok \u{FFFD}\u{FFFD} bytes".to_string()));
    }

    #[tokio::test]
    async fn execute_rejects_input_without_a_path() {
        // Act
        let result = read_file_tool().execute(json!({ "offset": 1 })).await;

        // Assert
        assert!(result.unwrap_err().contains("missing field `path`"));
    }

    #[tokio::test]
    async fn execute_rejects_non_object_input() {
        // Act
        let result = read_file_tool().execute(json!("just a string")).await;

        // Assert
        assert!(result.unwrap_err().starts_with("invalid read_file input:"));
    }

    #[tokio::test]
    async fn execute_rejects_invalid_or_out_of_range_parameters() {
        for input in [
            json!({ "path": "file.txt", "offset": 0 }),
            json!({ "path": "file.txt", "limit": 0 }),
            json!({ "path": "file.txt", "limit": MAX_READ_FILE_LIMIT + 1 }),
            json!({ "path": "file.txt", "offset": "one" }),
            json!({ "path": "file.txt", "unexpected": true }),
        ] {
            let result = read_file_tool().execute(input).await;
            assert!(
                result.is_err(),
                "invalid input should be rejected, got {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn execute_reports_an_error_for_a_missing_file() {
        // Act
        let result = read_file_tool()
            .execute(json!({ "path": "/definitely/not/a/real/file" }))
            .await;

        // Assert
        assert!(result.is_err());
    }
}
