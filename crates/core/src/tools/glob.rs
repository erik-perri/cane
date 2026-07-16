use crate::Workspace;
use crate::protocol::ApprovalRequirement;
use crate::tools::{PreparedInvocation, Tool, ToolDefinition, ToolExecutionError, invalid_input};
use globset::{GlobBuilder, GlobMatcher};
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GlobInput {
    path: Option<String>,
    pattern: String,
}

/// Most matches a single glob call will return; the newest are kept.
const MAX_GLOB_MATCHES: usize = 250;

pub struct GlobTool {
    workspace: Arc<Workspace>,
}

#[async_trait::async_trait]
impl Tool for GlobTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob".to_string(),
            description: format!(
                "Find files in the workspace by glob pattern. Returns workspace-relative \
                 file paths, one per line, most recently modified first; pass them directly \
                 to read_file or edit_file. Only files are listed, never directories or \
                 symlinks. Files ignored by .gitignore are excluded, .git is skipped, and \
                 hidden files are included. At most {MAX_GLOB_MATCHES} paths are returned; \
                 a truncated result says so explicitly — narrow the pattern and search \
                 again. Finding no matches is a normal result, not an error. File names \
                 that are not valid UTF-8 are displayed with replacement characters and \
                 may not open by that displayed name."
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to search, relative to the workspace \
                            root (or an absolute path inside the workspace). Omit to \
                            search the entire workspace.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern matched against paths relative to the \
                            search directory. Always use `/` as the separator, including on \
                            Windows. Matching is case-sensitive. `*` does not cross `/`: \
                            `*.rs` matches only direct children, `**/*.rs` matches \
                            recursively. Also supported: `?` (one character), `[abc]` \
                            (character class), `{a,b}` (alternation), `\\` (escape). \
                            Examples: `**/*.rs`, `src/**/*.test.ts`, `crates/*/Cargo.toml`.",
                    },
                },
                "required": ["pattern"],
                "additionalProperties": false,
            }),
        }
    }

    async fn prepare(&self, input: Value) -> Result<Box<dyn PreparedInvocation>, String> {
        let tool = self.prepare_glob(input)?;

        Ok(Box::new(tool))
    }
}

impl GlobTool {
    pub fn new(workspace: Arc<Workspace>) -> Self {
        Self { workspace }
    }

    fn prepare_glob(&self, input: Value) -> Result<PreparedGlob, String> {
        let input: GlobInput =
            serde_json::from_value(input).map_err(|error| invalid_input("glob", error))?;

        let (requested_path, resolved_path) = match input.path {
            Some(path) if path.is_empty() => {
                return Err(invalid_input("glob", "`path` must not be empty"));
            }
            Some(path) => {
                let resolved = self.workspace.resolve(&path)?;
                (path, resolved)
            }
            None => (".".to_owned(), self.workspace.root().to_path_buf()),
        };

        if input.pattern.is_empty() {
            return Err(invalid_input("glob", "`pattern` must not be empty"));
        }

        let matcher = GlobBuilder::new(&input.pattern)
            .backslash_escape(true)
            .case_insensitive(false)
            .literal_separator(true)
            .build()
            .map_err(|error| invalid_input("glob", error))?
            .compile_matcher();

        Ok(PreparedGlob {
            pattern: input.pattern,
            matcher,
            requested_path,
            resolved_path,
            workspace_path: self.workspace.root().to_path_buf(),
        })
    }
}

#[derive(Debug)]
struct PreparedGlob {
    pattern: String,
    matcher: GlobMatcher,
    requested_path: String,
    resolved_path: PathBuf,
    workspace_path: PathBuf,
}

#[async_trait::async_trait]
impl PreparedInvocation for PreparedGlob {
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
            pattern,
            matcher,
            requested_path,
            resolved_path,
            workspace_path,
        } = *self;

        todo!()
    }
}

#[derive(Debug)]
enum GlobError {
    //
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::{TempDir, tempdir};

    fn glob_tool() -> (TempDir, GlobTool) {
        let root = tempdir().unwrap();
        let workspace = Arc::new(Workspace::new(root.path().to_path_buf()).unwrap());
        let tool = GlobTool::new(workspace);
        (root, tool)
    }

    #[test]
    fn no_path_uses_workspace_root() {
        // Arrange
        let (root, tool) = glob_tool();

        // Act
        let result = tool.prepare_glob(json!({ "pattern": "*" })).unwrap();

        // Assert
        assert_eq!(result.resolved_path, root.path());
        assert_eq!(result.requested_path, ".");
    }

    #[test]
    fn validation_failures_produce_errors() {
        // Arrange
        let cases = [
            (
                json!({ "pattern": "" }),
                "invalid glob input: `pattern` must not be empty",
            ),
            (
                json!({ "pattern": "*", "path": "" }),
                "invalid glob input: `path` must not be empty",
            ),
            (
                json!({ "path": "." }),
                "invalid glob input: missing field `pattern`",
            ),
            (
                json!({ "pattern": 7 }),
                "invalid glob input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "pattern": "*", "path": 7 }),
                "invalid glob input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "pattern": "*", "extra": true }),
                "invalid glob input: unknown field `extra`, expected `path` or `pattern`",
            ),
            (
                json!("not an object"),
                "invalid glob input: invalid type: string \"not an object\", expected struct GlobInput",
            ),
            (
                json!(null),
                "invalid glob input: invalid type: null, expected struct GlobInput",
            ),
        ];
        let (_root, tool) = glob_tool();

        for (input, expected_error) in cases {
            // Act
            let error = tool.prepare_glob(input).unwrap_err();

            // Assert
            assert_eq!(error, expected_error);
        }
    }

    #[test]
    fn invalid_pattern_is_rejected() {
        // Arrange
        let cases = [
            (
                "[a-z",
                "invalid glob input: error parsing glob '[a-z': unclosed character class; missing ']'",
            ),
            (
                "*.{rs,txt",
                "invalid glob input: error parsing glob '*.{rs,txt': unclosed alternate group; missing '}' (maybe escape '{' with '[{]'?)",
            ),
            (
                "foo[",
                "invalid glob input: error parsing glob 'foo[': unclosed character class; missing ']'",
            ),
        ];
        let (_root, tool) = glob_tool();

        for (pattern, expected_error) in cases {
            // Act
            let result = tool
                .prepare_glob(json!({ "pattern": pattern }))
                .unwrap_err();

            // Assert
            assert_eq!(result, expected_error);
        }
    }

    #[tokio::test]
    async fn prepared_glob_invocations_require_no_approval() {
        // Arrange
        let (_root, tool) = glob_tool();

        // Act
        let prepared = tool.prepare(json!({ "pattern": "*" })).await.unwrap();

        // Assert
        assert_eq!(prepared.approval_requirement(), ApprovalRequirement::None);
    }

    #[tokio::test]
    async fn preparing_does_not_touch_directory_contents() {
        // Arrange
        let (root, tool) = glob_tool();
        let root_path = root.path().to_path_buf();
        std::fs::remove_dir(&root_path).unwrap();

        // Act
        let result = tool.prepare(json!({ "pattern": "*" })).await;

        // Assert
        assert!(result.is_ok());
        assert!(!root_path.exists());
    }
}
