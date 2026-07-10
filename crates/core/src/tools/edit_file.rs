use crate::Workspace;
use crate::tools::{
    MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MIB, Tool, ToolDefinition, background_task_failed,
    invalid_input, operation_failed,
};
use serde::Deserialize;
use serde_json::Value;
use std::fs::Permissions;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EditFileInput {
    path: String,
    old_str: String,
    new_str: String,
    expected_occurrences: Option<usize>,
}

pub struct EditFileTool {
    workspace: Arc<Workspace>,
}

impl EditFileTool {
    pub fn new(workspace: Arc<Workspace>) -> Self {
        Self { workspace }
    }
}

#[async_trait::async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_string(),
            description: format!(
                "Replace an exact string in a UTF-8 text file in the workspace. By default, the string must occur exactly once. Files larger than {MAX_FILE_SIZE_MIB} MiB cannot be edited."
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit."
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact text to replace."
                    },
                    "new_str": {
                        "type": "string",
                        "description": "New string to insert in place of old_str."
                    },
                    "expected_occurrences": {
                        "type": "integer",
                        "description": "Required number of matches. Defaults to 1.",
                        "minimum": 1
                    },
                },
                "required": ["path", "old_str", "new_str"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, input: Value) -> Result<String, String> {
        let input: EditFileInput =
            serde_json::from_value(input).map_err(|error| invalid_input("edit_file", error))?;

        if input.path.is_empty() {
            return Err(invalid_input("edit_file", "path must not be empty"));
        }
        if input.old_str.is_empty() {
            return Err(invalid_input("edit_file", "old_str must not be empty"));
        }
        if input.old_str == input.new_str {
            return Err(invalid_input(
                "edit_file",
                "old_str and new_str must differ",
            ));
        }
        if input.expected_occurrences == Some(0) {
            return Err(invalid_input(
                "edit_file",
                "expected_occurrences must be at least 1",
            ));
        }

        let resolved_path = self.workspace.resolve(&input.path)?;

        if resolved_path == self.workspace.root() {
            return Err(invalid_input(
                "edit_file",
                "path must name a file inside the workspace",
            ));
        }

        let requested_path = input.path;
        let expected_occurrences = input.expected_occurrences.unwrap_or(1);
        let explicit_occurrence_count = input.expected_occurrences.is_some();

        let result = tokio::task::spawn_blocking(move || {
            replace_in_file(
                &resolved_path,
                &input.old_str,
                &input.new_str,
                expected_occurrences,
                explicit_occurrence_count,
            )
        })
        .await
        .map_err(|error| background_task_failed("edit", &requested_path, error))?;

        match result {
            Ok(replaced) => Ok(format!(
                "edited `{requested_path}`; {replaced} {} replaced",
                occurrence_label(replaced)
            )),
            Err(error) => Err(format_edit_error(&requested_path, error)),
        }
    }
}

#[derive(Debug)]
enum EditFileError {
    Read(io::Error),
    TooLarge { size: u64 },
    InvalidUtf8,
    NoMatch,
    NonUnique { found: usize },
    UnexpectedOccurrences { expected: usize, found: usize },
    Write(io::Error),
}

fn replace_in_file(
    path: &Path,
    old_str: &str,
    new_str: &str,
    expected_occurrences: usize,
    explicit_occurrence_count: bool,
) -> Result<usize, EditFileError> {
    let metadata = std::fs::metadata(path).map_err(EditFileError::Read)?;
    if metadata.len() > MAX_FILE_SIZE_BYTES {
        return Err(EditFileError::TooLarge {
            size: metadata.len(),
        });
    }

    let bytes = std::fs::read(path).map_err(EditFileError::Read)?;
    let contents = String::from_utf8(bytes).map_err(|_| EditFileError::InvalidUtf8)?;
    let found = contents.matches(old_str).count();

    if found == 0 {
        return Err(EditFileError::NoMatch);
    }
    if found != expected_occurrences {
        return Err(if explicit_occurrence_count {
            EditFileError::UnexpectedOccurrences {
                expected: expected_occurrences,
                found,
            }
        } else {
            EditFileError::NonUnique { found }
        });
    }

    let edited = contents.replace(old_str, new_str);

    write_atomically(path, &edited, metadata.permissions()).map_err(EditFileError::Write)?;

    Ok(found)
}

/// Write via a temp file and rename so a crash mid-write cannot leave a
/// truncated file. The temp file must live in the target's directory: rename
/// is only atomic within one filesystem.
fn write_atomically(path: &Path, contents: &str, permissions: Permissions) -> io::Result<()> {
    // If it's a symlink, follow it to its actual target so we don't blow away the link
    let target_path = if path.is_symlink() {
        std::fs::canonicalize(path)?
    } else {
        path.to_path_buf()
    };

    let directory = target_path
        .parent()
        .ok_or_else(|| io::Error::other("path has no parent directory"))?;

    let mut temp = tempfile::NamedTempFile::new_in(directory)?;
    temp.write_all(contents.as_bytes())?;
    temp.as_file().set_permissions(permissions)?;
    temp.as_file().sync_all()?;

    // Persist to the absolute target path
    temp.persist(&target_path).map_err(|e| e.error)?;

    Ok(())
}

fn format_edit_error(path: &str, error: EditFileError) -> String {
    match error {
        EditFileError::Read(error) => operation_failed("read", path, error),
        EditFileError::TooLarge { size } => format!(
            "file `{path}` is {size} bytes, which exceeds the {MAX_FILE_SIZE_BYTES} byte edit limit"
        ),
        EditFileError::InvalidUtf8 => format!("file `{path}` is not valid UTF-8"),
        EditFileError::NoMatch => format!("old_str not found in `{path}`"),
        EditFileError::NonUnique { found } => format!(
            "old_str matches {found} times in `{path}`; include more surrounding context to make it unique"
        ),
        EditFileError::UnexpectedOccurrences { expected, found } => {
            format!("old_str matches {found} times in `{path}`; expected {expected}")
        }
        EditFileError::Write(error) => operation_failed("write", path, error),
    }
}

fn occurrence_label(count: usize) -> &'static str {
    if count == 1 {
        "occurrence"
    } else {
        "occurrences"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::path::Path;
    use tempfile::{TempDir, tempdir};

    fn edit_file_tool() -> (TempDir, EditFileTool) {
        let root = tempdir().unwrap();
        let workspace = Arc::new(Workspace::new(root.path().to_path_buf()).unwrap());
        let tool = EditFileTool::new(workspace);
        (root, tool)
    }

    fn path_str(path: &Path) -> &str {
        path.to_str().unwrap()
    }

    #[test]
    fn definition_describes_strict_edit_file_input() {
        // Arrange
        let (_root, tool) = edit_file_tool();

        // Act
        let definition = tool.definition();

        // Assert
        assert_eq!(definition.name, "edit_file");
        assert_eq!(definition.input_schema["type"], "object");
        assert_eq!(
            definition.input_schema["required"],
            json!(["path", "old_str", "new_str"])
        );
        assert_eq!(definition.input_schema["additionalProperties"], false);
        for property in ["path", "old_str", "new_str"] {
            assert_eq!(
                definition.input_schema["properties"][property]["type"],
                "string"
            );
        }
        assert_eq!(
            definition.input_schema["properties"]["expected_occurrences"]["type"],
            "integer"
        );
        assert_eq!(
            definition.input_schema["properties"]["expected_occurrences"]["minimum"],
            json!(1)
        );
    }

    #[tokio::test]
    async fn execute_replaces_one_exact_match() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("message.txt");
        fs::write(&target, "hello old world").unwrap();

        // Act
        let output = tool
            .execute(json!({
                "path": "message.txt",
                "old_str": "old",
                "new_str": "new"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "hello new world");
        assert_eq!(output, "edited `message.txt`; 1 occurrence replaced");
    }

    #[tokio::test]
    async fn execute_accepts_an_absolute_path_inside_the_workspace() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("absolute.txt");
        fs::write(&target, "before").unwrap();

        // Act
        let output = tool
            .execute(json!({
                "path": path_str(&target),
                "old_str": "before",
                "new_str": "after"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(&target).unwrap(), "after");
        assert_eq!(
            output,
            format!("edited `{}`; 1 occurrence replaced", target.display())
        );
    }

    #[tokio::test]
    async fn execute_preserves_every_byte_outside_the_match() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("source.txt");
        let original = b"prefix\r\nold value\r\nsuffix\n";
        fs::write(&target, original).unwrap();

        // Act
        let output = tool
            .execute(json!({
                "path": "source.txt",
                "old_str": "old value",
                "new_str": "new value"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(
            fs::read(target).unwrap(),
            b"prefix\r\nnew value\r\nsuffix\n"
        );
        assert_eq!(output, "edited `source.txt`; 1 occurrence replaced");
    }

    #[tokio::test]
    async fn execute_can_delete_the_matched_text() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("delete.txt");
        fs::write(&target, "keep DELETE keep").unwrap();

        // Act
        let output = tool
            .execute(json!({
                "path": "delete.txt",
                "old_str": " DELETE",
                "new_str": ""
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "keep keep");
        assert_eq!(output, "edited `delete.txt`; 1 occurrence replaced");
    }

    #[tokio::test]
    async fn execute_handles_multibyte_utf8_without_corrupting_it() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("unicode.txt");
        fs::write(&target, "before 🦀 after").unwrap();

        // Act
        let output = tool
            .execute(json!({
                "path": "unicode.txt",
                "old_str": "🦀",
                "new_str": "🐸"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "before 🐸 after");
        assert_eq!(output, "edited `unicode.txt`; 1 occurrence replaced");
    }

    #[tokio::test]
    async fn execute_does_not_apply_a_new_match_created_by_the_replacement() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("single-pass.txt");
        fs::write(&target, "a").unwrap();

        // Act
        let output = tool
            .execute(json!({
                "path": "single-pass.txt",
                "old_str": "a",
                "new_str": "aa"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "aa");
        assert_eq!(output, "edited `single-pass.txt`; 1 occurrence replaced");
    }

    #[tokio::test]
    async fn execute_returns_the_required_no_match_error_and_leaves_the_file_untouched() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("target.txt");
        fs::write(&target, "original").unwrap();

        // Act
        let error = tool
            .execute(json!({
                "path": "target.txt",
                "old_str": "missing",
                "new_str": "replacement"
            }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, "old_str not found in `target.txt`");
        assert_eq!(fs::read_to_string(target).unwrap(), "original");
    }

    #[tokio::test]
    async fn execute_returns_the_required_multi_match_error_and_leaves_the_file_untouched() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("target.txt");
        fs::write(&target, "old and old and old").unwrap();

        // Act
        let error = tool
            .execute(json!({
                "path": "target.txt",
                "old_str": "old",
                "new_str": "new"
            }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(
            error,
            "old_str matches 3 times in `target.txt`; include more surrounding context to make it unique"
        );
        assert_eq!(fs::read_to_string(target).unwrap(), "old and old and old");
    }

    #[tokio::test]
    async fn execute_replaces_the_requested_number_of_occurrences() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("target.txt");
        fs::write(&target, "old and old").unwrap();

        // Act
        let output = tool
            .execute(json!({
                "path": "target.txt",
                "old_str": "old",
                "new_str": "new",
                "expected_occurrences": 2
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "new and new");
        assert_eq!(output, "edited `target.txt`; 2 occurrences replaced");
    }

    #[tokio::test]
    async fn execute_rejects_an_expected_occurrence_mismatch_without_writing() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("target.txt");
        fs::write(&target, "old and old").unwrap();

        // Act
        let error = tool
            .execute(json!({
                "path": "target.txt",
                "old_str": "old",
                "new_str": "new",
                "expected_occurrences": 3
            }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, "old_str matches 2 times in `target.txt`; expected 3");
        assert_eq!(fs::read_to_string(target).unwrap(), "old and old");
    }

    #[tokio::test]
    async fn execute_rejects_zero_expected_occurrences_without_writing() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("target.txt");
        fs::write(&target, "old").unwrap();

        // Act
        let error = tool
            .execute(json!({
                "path": "target.txt",
                "old_str": "old",
                "new_str": "new",
                "expected_occurrences": 0
            }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(
            error,
            "invalid edit_file input: expected_occurrences must be at least 1"
        );
        assert_eq!(fs::read_to_string(target).unwrap(), "old");
    }

    #[tokio::test]
    async fn execute_rejects_an_empty_old_str_without_writing() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("target.txt");
        fs::write(&target, "original").unwrap();

        // Act
        let error = tool
            .execute(json!({
                "path": "target.txt",
                "old_str": "",
                "new_str": "new"
            }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, "invalid edit_file input: old_str must not be empty");
        assert_eq!(fs::read_to_string(target).unwrap(), "original");
    }

    #[tokio::test]
    async fn execute_rejects_identical_old_and_new_strings_without_writing() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("target.txt");
        fs::write(&target, "same").unwrap();

        // Act
        let error = tool
            .execute(json!({
                "path": "target.txt",
                "old_str": "same",
                "new_str": "same"
            }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(
            error,
            "invalid edit_file input: old_str and new_str must differ"
        );
        assert_eq!(fs::read_to_string(target).unwrap(), "same");
    }

    #[tokio::test]
    async fn execute_rejects_an_empty_path_without_writing() {
        // Arrange
        let (root, tool) = edit_file_tool();

        // Act
        let error = tool
            .execute(json!({ "path": "", "old_str": "old", "new_str": "new" }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, "invalid edit_file input: path must not be empty");
        assert!(fs::read_dir(root.path()).unwrap().next().is_none());
    }

    #[tokio::test]
    async fn execute_rejects_the_workspace_root_as_an_edit_target() {
        let (root, tool) = edit_file_tool();

        for path in [".".to_string(), path_str(root.path()).to_string()] {
            let error = tool
                .execute(json!({ "path": path, "old_str": "old", "new_str": "new" }))
                .await
                .unwrap_err();

            assert_eq!(
                error,
                "invalid edit_file input: path must name a file inside the workspace"
            );
        }
    }

    #[tokio::test]
    async fn execute_rejects_invalid_utf8_and_preserves_the_original_bytes() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("binary.dat");
        let original = b"before \xff after";
        fs::write(&target, original).unwrap();

        // Act
        let error = tool
            .execute(json!({
                "path": "binary.dat",
                "old_str": "before",
                "new_str": "changed"
            }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, "file `binary.dat` is not valid UTF-8");
        assert_eq!(fs::read(target).unwrap(), original);
    }

    #[tokio::test]
    async fn execute_rejects_missing_required_fields() {
        // Arrange
        let (_root, tool) = edit_file_tool();

        for (input, expected) in [
            (
                json!({ "old_str": "old", "new_str": "new" }),
                "invalid edit_file input: missing field `path`",
            ),
            (
                json!({ "path": "target.txt", "new_str": "new" }),
                "invalid edit_file input: missing field `old_str`",
            ),
            (
                json!({ "path": "target.txt", "old_str": "old" }),
                "invalid edit_file input: missing field `new_str`",
            ),
        ] {
            // Act
            let error = tool.execute(input).await.unwrap_err();

            // Assert
            assert_eq!(error, expected);
        }
    }

    #[tokio::test]
    async fn execute_rejects_unknown_fields_wrong_types_and_non_objects() {
        // Arrange
        let (_root, tool) = edit_file_tool();

        for (input, expected) in [
            (
                json!({ "path": "target.txt", "old_str": "old", "new_str": "new", "extra": true }),
                "invalid edit_file input: unknown field `extra`, expected one of `path`, `old_str`, `new_str`, `expected_occurrences`",
            ),
            (
                json!({ "path": 7, "old_str": "old", "new_str": "new" }),
                "invalid edit_file input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "path": "target.txt", "old_str": 7, "new_str": "new" }),
                "invalid edit_file input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "path": "target.txt", "old_str": "old", "new_str": 7 }),
                "invalid edit_file input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "path": "target.txt", "old_str": "old", "new_str": "new", "expected_occurrences": "two" }),
                "invalid edit_file input: invalid type: string \"two\", expected usize",
            ),
            (
                json!("not an object"),
                "invalid edit_file input: invalid type: string \"not an object\", expected struct EditFileInput",
            ),
            (
                json!(null),
                "invalid edit_file input: invalid type: null, expected struct EditFileInput",
            ),
        ] {
            // Act
            let error = tool.execute(input).await.unwrap_err();

            // Assert
            assert_eq!(error, expected);
        }
    }

    #[tokio::test]
    async fn execute_rejects_a_file_over_the_size_cap_without_modifying_it() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("huge.txt");
        let file = fs::File::create(&target).unwrap();
        file.set_len(MAX_FILE_SIZE_BYTES + 1).unwrap();
        drop(file);

        // Act
        let error = tool
            .execute(json!({
                "path": "huge.txt",
                "old_str": "old",
                "new_str": "new"
            }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(
            error,
            format!(
                "file `huge.txt` is {} bytes, which exceeds the {MAX_FILE_SIZE_BYTES} byte edit limit",
                MAX_FILE_SIZE_BYTES + 1
            )
        );
        assert_eq!(
            fs::metadata(&target).unwrap().len(),
            MAX_FILE_SIZE_BYTES + 1
        );
    }

    #[tokio::test]
    async fn execute_edits_a_file_exactly_at_the_size_cap() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("exact.txt");
        let padding = "a".repeat(MAX_FILE_SIZE_BYTES as usize - 3);
        fs::write(&target, format!("{padding}old")).unwrap();

        // Act
        let output = tool
            .execute(json!({
                "path": "exact.txt",
                "old_str": "old",
                "new_str": "new"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, "edited `exact.txt`; 1 occurrence replaced");
        assert!(fs::read_to_string(target).unwrap().ends_with("new"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn execute_preserves_the_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("script.sh");
        fs::write(&target, "echo old").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();

        // Act
        tool.execute(json!({
            "path": "script.sh",
            "old_str": "old",
            "new_str": "new"
        }))
        .await
        .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(&target).unwrap(), "echo new");
        assert_eq!(
            fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }

    #[tokio::test]
    async fn execute_reports_a_missing_file_without_creating_it() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("missing.txt");

        // Act
        let error = tool
            .execute(json!({
                "path": "missing.txt",
                "old_str": "old",
                "new_str": "new"
            }))
            .await
            .unwrap_err();

        // Assert
        assert!(
            error.starts_with("failed to read `missing.txt`: "),
            "unexpected error: {error}"
        );
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn execute_adds_path_context_when_path_resolution_fails() {
        let (root, tool) = edit_file_tool();
        let parent_file = root.path().join("Cargo.toml");
        fs::write(&parent_file, "original").unwrap();

        let error = tool
            .execute(json!({
                "path": "Cargo.toml/nested.txt",
                "old_str": "old",
                "new_str": "new"
            }))
            .await
            .unwrap_err();

        assert!(
            error.starts_with("failed to resolve `Cargo.toml/nested.txt`: "),
            "unexpected error: {error}"
        );
        assert_eq!(fs::read_to_string(parent_file).unwrap(), "original");
    }

    #[tokio::test]
    async fn execute_rejects_an_absolute_outside_path_and_leaves_it_untouched() {
        // Arrange
        let (_root, tool) = edit_file_tool();
        let outside = tempdir().unwrap();
        let target = outside.path().join("important.txt");
        fs::write(&target, "original").unwrap();
        let expected_error = format!(
            "access denied: path `{}` is outside workspace root `{}`",
            target.display(),
            tool.workspace.root().display()
        );

        // Act
        let error = tool
            .execute(json!({
                "path": path_str(&target),
                "old_str": "original",
                "new_str": "changed"
            }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, expected_error);
        assert_eq!(fs::read_to_string(target).unwrap(), "original");
    }

    #[tokio::test]
    async fn execute_errors_when_the_target_is_a_directory() {
        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("directory");
        fs::create_dir(&target).unwrap();

        // Act
        let error = tool
            .execute(json!({
                "path": "directory",
                "old_str": "old",
                "new_str": "new"
            }))
            .await
            .unwrap_err();

        // Assert
        assert!(
            error.starts_with("failed to read `directory`: "),
            "unexpected error: {error}"
        );
        assert!(target.is_dir());
    }

    #[test]
    fn format_edit_error_covers_read_and_write_failures() {
        assert_eq!(
            format_edit_error(
                "notes.txt",
                EditFileError::Read(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
            ),
            "failed to read `notes.txt`: denied"
        );
        assert_eq!(
            format_edit_error(
                "notes.txt",
                EditFileError::Write(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
            ),
            "failed to write `notes.txt`: denied"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn execute_edits_an_inside_symlink_target_without_replacing_the_link() {
        use std::os::unix::fs::symlink;

        // Arrange
        let (root, tool) = edit_file_tool();
        let target = root.path().join("target.txt");
        let link = root.path().join("link.txt");
        fs::write(&target, "old").unwrap();
        symlink(&target, &link).unwrap();

        // Act
        let output = tool
            .execute(json!({ "path": "link.txt", "old_str": "old", "new_str": "new" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "new");
        assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
        assert_eq!(output, "edited `link.txt`; 1 occurrence replaced");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn execute_rejects_an_outside_symlink_and_leaves_its_target_untouched() {
        use std::os::unix::fs::symlink;

        // Arrange
        let (root, tool) = edit_file_tool();
        let outside = tempdir().unwrap();
        let target = outside.path().join("target.txt");
        fs::write(&target, "old").unwrap();
        symlink(&target, root.path().join("link.txt")).unwrap();
        let expected_error = format!(
            "access denied: path `link.txt` is outside workspace root `{}`",
            tool.workspace.root().display()
        );

        // Act
        let error = tool
            .execute(json!({ "path": "link.txt", "old_str": "old", "new_str": "new" }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, expected_error);
        assert_eq!(fs::read_to_string(target).unwrap(), "old");
    }
}
