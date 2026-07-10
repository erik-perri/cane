use crate::Workspace;
use crate::tools::{Tool, ToolDefinition, background_task_failed, invalid_input, operation_failed};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteFileInput {
    path: String,
    content: String,
}

pub struct WriteFileTool {
    workspace: Arc<Workspace>,
}

impl WriteFileTool {
    pub fn new(workspace: Arc<Workspace>) -> Self {
        Self { workspace }
    }
}

#[async_trait::async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description:
                "Create or overwrite a file in the workspace, creating missing parent directories."
                    .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file."
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, input: Value) -> Result<String, String> {
        let input: WriteFileInput =
            serde_json::from_value(input).map_err(|error| invalid_input("write_file", error))?;

        if input.path.is_empty() {
            return Err(invalid_input("write_file", "path must not be empty"));
        }

        let resolved_path = self.workspace.resolve(&input.path)?;

        if resolved_path == self.workspace.root() {
            return Err(invalid_input(
                "write_file",
                "path must name a file inside the workspace",
            ));
        }

        let existed = resolved_path.exists();
        let requested_path = input.path;

        let written =
            tokio::task::spawn_blocking(move || write_to_file(&resolved_path, input.content))
                .await
                .map_err(|error| background_task_failed("write", &requested_path, error))?
                .map_err(|error| operation_failed("write", &requested_path, error))?;

        let message = if existed {
            format!("updated `{requested_path}`; {written} bytes written")
        } else {
            format!("created `{requested_path}`; {written} bytes written")
        };

        Ok(message)
    }
}

fn write_to_file(path: &Path, content: String) -> std::io::Result<usize> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(path, &content)?;

    Ok(content.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::path::Path;
    use tempfile::{TempDir, tempdir};

    fn write_file_tool() -> (TempDir, WriteFileTool) {
        let root = tempdir().unwrap();
        let workspace = Arc::new(Workspace::new(root.path().to_path_buf()).unwrap());
        let tool = WriteFileTool::new(workspace);
        (root, tool)
    }

    fn path_str(path: &Path) -> &str {
        path.to_str().unwrap()
    }

    #[test]
    fn definition_describes_strict_write_file_input() {
        // Arrange
        let (_root, tool) = write_file_tool();

        // Act
        let definition = tool.definition();

        // Assert
        assert_eq!(definition.name, "write_file");
        assert_eq!(definition.input_schema["type"], "object");
        assert_eq!(
            definition.input_schema["required"],
            json!(["path", "content"])
        );
        assert_eq!(definition.input_schema["additionalProperties"], false);
        assert_eq!(
            definition.input_schema["properties"]["path"]["type"],
            "string"
        );
        assert_eq!(
            definition.input_schema["properties"]["content"]["type"],
            "string"
        );
    }

    #[tokio::test]
    async fn execute_creates_a_file_relative_to_the_workspace() {
        // Arrange
        let (root, tool) = write_file_tool();
        let target = root.path().join("notes.txt");

        // Act
        let output = tool
            .execute(json!({ "path": "notes.txt", "content": "hello" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "hello");
        assert_eq!(output, "created `notes.txt`; 5 bytes written");
    }

    #[tokio::test]
    async fn execute_accepts_an_absolute_path_inside_the_workspace() {
        // Arrange
        let (root, tool) = write_file_tool();
        let target = root.path().join("absolute.txt");

        // Act
        let output = tool
            .execute(json!({ "path": path_str(&target), "content": "absolute" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(&target).unwrap(), "absolute");
        assert_eq!(
            output,
            format!("created `{}`; 8 bytes written", target.display())
        );
    }

    #[tokio::test]
    async fn execute_creates_all_missing_parent_directories() {
        // Arrange
        let (root, tool) = write_file_tool();
        let target = root.path().join("generated/nested/output.txt");

        // Act
        let output = tool
            .execute(json!({
                "path": "generated/nested/output.txt",
                "content": "generated"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "generated");
        assert_eq!(
            output,
            "created `generated/nested/output.txt`; 9 bytes written"
        );
    }

    #[tokio::test]
    async fn execute_overwrites_and_truncates_an_existing_file() {
        // Arrange
        let (root, tool) = write_file_tool();
        let target = root.path().join("existing.txt");
        fs::write(&target, "a much longer original value").unwrap();

        // Act
        let output = tool
            .execute(json!({ "path": "existing.txt", "content": "short" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "short");
        assert_eq!(output, "updated `existing.txt`; 5 bytes written");
    }

    #[tokio::test]
    async fn execute_can_create_an_empty_file() {
        // Arrange
        let (root, tool) = write_file_tool();
        let target = root.path().join("empty.txt");

        // Act
        let output = tool
            .execute(json!({ "path": "empty.txt", "content": "" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read(target).unwrap(), b"");
        assert_eq!(output, "created `empty.txt`; 0 bytes written");
    }

    #[tokio::test]
    async fn execute_reports_utf8_bytes_not_character_count() {
        // Arrange
        let (root, tool) = write_file_tool();
        let content = "🦀";

        // Act
        let output = tool
            .execute(json!({ "path": "unicode.txt", "content": content }))
            .await
            .unwrap();

        // Assert
        assert_eq!(
            fs::read(root.path().join("unicode.txt")).unwrap(),
            content.as_bytes()
        );
        assert_eq!(output, "created `unicode.txt`; 4 bytes written");
    }

    #[tokio::test]
    async fn execute_rejects_an_empty_path_without_writing() {
        // Arrange
        let (root, tool) = write_file_tool();

        // Act
        let error = tool
            .execute(json!({ "path": "", "content": "hello" }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, "invalid write_file input: path must not be empty");
        assert!(fs::read_dir(root.path()).unwrap().next().is_none());
    }

    #[tokio::test]
    async fn execute_rejects_the_workspace_root_as_a_write_target() {
        // Arrange
        let (root, tool) = write_file_tool();
        let absolute_root = path_str(root.path()).to_string();

        for path in [".".to_string(), absolute_root] {
            // Act
            let error = tool
                .execute(json!({ "path": path, "content": "hello" }))
                .await
                .unwrap_err();

            // Assert
            assert_eq!(
                error,
                "invalid write_file input: path must name a file inside the workspace"
            );
        }
        assert!(fs::read_dir(root.path()).unwrap().next().is_none());
    }

    #[tokio::test]
    async fn execute_rejects_missing_required_fields() {
        // Arrange
        let (_root, tool) = write_file_tool();

        for (input, expected) in [
            (
                json!({ "content": "hello" }),
                "invalid write_file input: missing field `path`",
            ),
            (
                json!({ "path": "file.txt" }),
                "invalid write_file input: missing field `content`",
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
        let (_root, tool) = write_file_tool();

        for (input, expected) in [
            (
                json!({ "path": "file.txt", "content": "hello", "extra": true }),
                "invalid write_file input: unknown field `extra`, expected `path` or `content`",
            ),
            (
                json!({ "path": 7, "content": "hello" }),
                "invalid write_file input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "path": "file.txt", "content": 7 }),
                "invalid write_file input: invalid type: integer `7`, expected a string",
            ),
            (
                json!("not an object"),
                "invalid write_file input: invalid type: string \"not an object\", expected struct WriteFileInput",
            ),
            (
                json!(null),
                "invalid write_file input: invalid type: null, expected struct WriteFileInput",
            ),
        ] {
            // Act
            let error = tool.execute(input).await.unwrap_err();

            // Assert
            assert_eq!(error, expected);
        }
    }

    #[tokio::test]
    async fn execute_rejects_an_absolute_outside_path_and_leaves_it_untouched() {
        // Arrange
        let (_root, tool) = write_file_tool();
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
            .execute(json!({ "path": path_str(&target), "content": "changed" }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, expected_error);
        assert_eq!(fs::read_to_string(target).unwrap(), "original");
    }

    #[tokio::test]
    async fn execute_rejects_a_parent_escape_without_creating_the_file() {
        // Arrange
        let (root, tool) = write_file_tool();
        let escaped_name = format!(
            "{}-escaped.txt",
            root.path().file_name().unwrap().to_string_lossy()
        );
        let outside = root.path().parent().unwrap().join(&escaped_name);
        let candidate = format!("../{escaped_name}");
        let _ = fs::remove_file(&outside);
        let expected_error = format!(
            "access denied: path `{candidate}` is outside workspace root `{}`",
            tool.workspace.root().display()
        );

        // Act
        let error = tool
            .execute(json!({ "path": candidate, "content": "escaped" }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, expected_error);
        assert!(!outside.exists());
    }

    #[tokio::test]
    async fn execute_errors_when_the_target_is_a_directory() {
        // Arrange
        let (root, tool) = write_file_tool();
        let target = root.path().join("directory");
        fs::create_dir(&target).unwrap();

        // Act
        let error = tool
            .execute(json!({ "path": "directory", "content": "hello" }))
            .await
            .unwrap_err();

        // Assert
        assert!(
            error.starts_with("failed to write `directory`: "),
            "unexpected error: {error}"
        );
        assert!(target.is_dir());
    }

    #[tokio::test]
    async fn execute_errors_when_a_parent_component_is_a_file() {
        // Arrange
        let (root, tool) = write_file_tool();
        let parent_file = root.path().join("Cargo.toml");
        fs::write(&parent_file, "original").unwrap();

        // Act
        let error = tool
            .execute(json!({ "path": "Cargo.toml/nested.txt", "content": "hello" }))
            .await
            .unwrap_err();

        // Assert
        assert!(
            error.starts_with("failed to resolve `Cargo.toml/nested.txt`: "),
            "unexpected error: {error}"
        );
        assert_eq!(fs::read_to_string(parent_file).unwrap(), "original");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn execute_follows_an_inside_symlink_without_replacing_the_link() {
        use std::os::unix::fs::symlink;

        // Arrange
        let (root, tool) = write_file_tool();
        let target = root.path().join("target.txt");
        let link = root.path().join("link.txt");
        fs::write(&target, "original").unwrap();
        symlink(&target, &link).unwrap();

        // Act
        let output = tool
            .execute(json!({ "path": "link.txt", "content": "changed" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(fs::read_to_string(target).unwrap(), "changed");
        assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
        assert_eq!(output, "updated `link.txt`; 7 bytes written");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn execute_rejects_an_outside_symlink_and_leaves_its_target_untouched() {
        use std::os::unix::fs::symlink;

        // Arrange
        let (root, tool) = write_file_tool();
        let outside = tempdir().unwrap();
        let target = outside.path().join("target.txt");
        fs::write(&target, "original").unwrap();
        symlink(&target, root.path().join("link.txt")).unwrap();
        let expected_error = format!(
            "access denied: path `link.txt` is outside workspace root `{}`",
            tool.workspace.root().display()
        );

        // Act
        let error = tool
            .execute(json!({ "path": "link.txt", "content": "changed" }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(error, expected_error);
        assert_eq!(fs::read_to_string(target).unwrap(), "original");
    }
}
