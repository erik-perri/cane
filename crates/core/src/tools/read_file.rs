use crate::protocol::ApprovalRequirement;
use crate::tools::{
    MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MIB, PreparedInvocation, Tool, ToolDefinition,
    ToolExecutionError, background_task_failed, invalid_input, operation_failed,
};
use crate::workspace::Workspace;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

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
            description: format!(
                "Read a text file from the local filesystem. Returns the requested lines as raw text. Files larger than {MAX_FILE_SIZE_MIB} MiB cannot be read."
            ),
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
            }),
        }
    }

    async fn prepare(&self, input: Value) -> Result<Box<dyn PreparedInvocation>, String> {
        let input: ReadFileInput =
            serde_json::from_value(input).map_err(|error| invalid_input("read_file", error))?;

        if input.path.is_empty() {
            return Err(invalid_input("read_file", "path must not be empty"));
        }
        if input.offset == 0 {
            return Err(invalid_input("read_file", "offset must be at least 1"));
        }
        if input.limit == 0 || input.limit > MAX_READ_FILE_LIMIT {
            return Err(invalid_input(
                "read_file",
                format_args!("limit must be between 1 and {MAX_READ_FILE_LIMIT}"),
            ));
        }

        let resolved_path = self.workspace.resolve(&input.path)?;

        Ok(Box::new(PreparedReadFile {
            requested_path: input.path,
            resolved_path,
            offset: input.offset,
            limit: input.limit,
        }))
    }
}

struct PreparedReadFile {
    requested_path: String,
    resolved_path: PathBuf,
    offset: u64,
    limit: u64,
}

#[async_trait::async_trait]
impl PreparedInvocation for PreparedReadFile {
    fn approval_requirement(&self) -> ApprovalRequirement {
        ApprovalRequirement::None
    }

    async fn execute(
        self: Box<Self>,
        cancel: CancellationToken,
    ) -> Result<String, ToolExecutionError> {
        if cancel.is_cancelled() {
            return Err(ToolExecutionError::Cancelled);
        }

        let Self {
            requested_path,
            resolved_path,
            offset,
            limit,
        } = *self;

        let read = tokio::task::spawn_blocking(move || {
            read_lines_from_file(&resolved_path, offset, limit)
        })
        .await
        .map_err(|error| background_task_failed("read", &requested_path, error))?
        .map_err(|error| operation_failed("read", &requested_path, error))?;

        Ok(read)
    }
}

/// `offset` and `limit` have already been validated by `ReadFileInput`.
fn read_lines_from_file(path: &Path, offset: u64, limit: u64) -> std::io::Result<String> {
    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(std::io::Error::other("path is not a file"));
    }

    let size = metadata.len();
    if size > MAX_FILE_SIZE_BYTES {
        return Err(std::io::Error::other(format!(
            "file is {size} bytes, which exceeds the {MAX_FILE_SIZE_BYTES} byte read limit"
        )));
    }

    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);

    let lines = text.lines().skip(offset.saturating_sub(1) as usize);
    let selected: Vec<&str> = lines.take(limit as usize).collect();

    Ok(selected.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{Tool, ToolTestExt};
    use serde_json::json;
    use std::fs;
    use std::io::Write;
    use tempfile::{NamedTempFile, tempdir};

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
    async fn prepared_reads_do_not_require_approval() {
        // Arrange
        let file = temp_file_with(b"content");

        // Act
        let prepared = read_file_tool()
            .prepare(json!({ "path": path_of(&file) }))
            .await
            .unwrap();

        // Assert
        assert_eq!(prepared.approval_requirement(), ApprovalRequirement::None);
    }

    #[tokio::test]
    async fn execute_reads_an_entire_file_shorter_than_the_default_limit() {
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
    async fn execute_rejects_a_file_over_the_size_cap() {
        // Arrange
        let file = temp_file_with(b"");
        file.as_file().set_len(MAX_FILE_SIZE_BYTES + 1).unwrap();

        // Act
        let error = read_file_tool()
            .execute(json!({ "path": path_of(&file) }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(
            error,
            format!(
                "failed to read `{}`: file is {} bytes, which exceeds the {MAX_FILE_SIZE_BYTES} byte read limit",
                path_of(&file),
                MAX_FILE_SIZE_BYTES + 1
            )
        );
    }

    #[tokio::test]
    async fn execute_rejects_missing_fields_wrong_types_unknown_fields_and_non_objects() {
        // Arrange
        let cases = [
            (
                json!({ "offset": 1 }),
                "invalid read_file input: missing field `path`",
            ),
            (
                json!({ "path": 7 }),
                "invalid read_file input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "path": "file.txt", "offset": "one" }),
                "invalid read_file input: invalid type: string \"one\", expected u64",
            ),
            (
                json!({ "path": "file.txt", "limit": "many" }),
                "invalid read_file input: invalid type: string \"many\", expected u64",
            ),
            (
                json!({ "path": "file.txt", "unexpected": true }),
                "invalid read_file input: unknown field `unexpected`, expected one of `path`, `offset`, `limit`",
            ),
            (
                json!("just a string"),
                "invalid read_file input: invalid type: string \"just a string\", expected struct ReadFileInput",
            ),
            (
                json!(null),
                "invalid read_file input: invalid type: null, expected struct ReadFileInput",
            ),
        ];
        let tool = read_file_tool();

        for (input, expected) in cases {
            // Act
            let error = tool.execute(input).await.unwrap_err();

            // Assert
            assert_eq!(error, expected);
        }
    }

    #[tokio::test]
    async fn execute_rejects_empty_and_out_of_range_parameters() {
        // Arrange
        let cases = [
            (
                json!({ "path": "" }),
                "invalid read_file input: path must not be empty".to_string(),
            ),
            (
                json!({ "path": "file.txt", "offset": 0 }),
                "invalid read_file input: offset must be at least 1".to_string(),
            ),
            (
                json!({ "path": "file.txt", "limit": 0 }),
                format!(
                    "invalid read_file input: limit must be between 1 and {MAX_READ_FILE_LIMIT}"
                ),
            ),
            (
                json!({ "path": "file.txt", "limit": MAX_READ_FILE_LIMIT + 1 }),
                format!(
                    "invalid read_file input: limit must be between 1 and {MAX_READ_FILE_LIMIT}"
                ),
            ),
        ];
        let tool = read_file_tool();

        for (input, expected) in cases {
            // Act
            let error = tool.execute(input).await.unwrap_err();

            // Assert
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn definition_describes_strict_read_file_input() {
        // Arrange
        let tool = read_file_tool();

        // Act
        let definition = tool.definition();

        // Assert
        assert_eq!(definition.name, "read_file");
        assert_eq!(definition.input_schema["type"], "object");
        assert_eq!(definition.input_schema["required"], json!(["path"]));
        assert_eq!(definition.input_schema["additionalProperties"], false);
        assert_eq!(
            definition.input_schema["properties"]["path"]["type"],
            "string"
        );
        assert_eq!(
            definition.input_schema["properties"]["offset"]["type"],
            "integer"
        );
        assert_eq!(
            definition.input_schema["properties"]["offset"]["minimum"],
            1
        );
        assert_eq!(
            definition.input_schema["properties"]["limit"]["type"],
            "integer"
        );
        assert_eq!(definition.input_schema["properties"]["limit"]["minimum"], 1);
        assert_eq!(
            definition.input_schema["properties"]["limit"]["maximum"],
            MAX_READ_FILE_LIMIT
        );
    }

    #[tokio::test]
    async fn execute_reads_a_file_exactly_at_the_size_cap() {
        // Arrange
        let tool = read_file_tool();
        let target = temp_file_with(b"");
        let file = fs::File::create(&target).unwrap();
        file.set_len(MAX_FILE_SIZE_BYTES).unwrap();
        drop(file);

        // Act
        let result = tool
            .execute(json!({ "path": target.path() }))
            .await
            .unwrap();

        // Assert
        assert_eq!(MAX_FILE_SIZE_BYTES as usize, result.len());
    }

    #[tokio::test]
    async fn execute_errors_when_the_target_is_a_directory() {
        // Arrange
        let tool = read_file_tool();
        let dir = tempdir().unwrap();

        // Act
        let error = tool
            .execute(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(
            format!(
                "failed to read `{}`: path is not a file",
                dir.path().display(),
            ),
            error,
        );
    }

    #[tokio::test]
    async fn execute_reports_outside_paths_as_access_denied() {
        // Arrange
        let tool = read_file_tool();
        let outside = tool.workspace.root().parent().unwrap().join("outside.txt");
        let expected = format!(
            "access denied: path `{}` is outside workspace root `{}`",
            outside.display(),
            tool.workspace.root().display()
        );

        // Act
        let error = tool
            .execute(json!({ "path": path_of_path(&outside) }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, expected);
    }

    #[tokio::test]
    async fn execute_adds_operation_and_path_context_to_filesystem_errors() {
        // Arrange
        let parent = tempdir().unwrap();
        let missing = parent.path().join("missing.txt");

        // Act
        let error = read_file_tool()
            .execute(json!({ "path": path_of_path(&missing) }))
            .await
            .unwrap_err();

        // Assert
        assert!(
            error.starts_with(&format!("failed to read `{}`: ", missing.display())),
            "unexpected error: {error}"
        );
    }

    fn path_of_path(path: &Path) -> &str {
        path.to_str().unwrap()
    }
}
