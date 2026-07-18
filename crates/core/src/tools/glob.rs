use crate::Workspace;
use crate::protocol::ApprovalRequirement;
use crate::tools::{
    PreparedInvocation, Tool, ToolDefinition, ToolExecutionError, background_task_failed,
    invalid_input, operation_failed,
};
use globset::{GlobBuilder, GlobMatcher};
use ignore::{DirEntry, WalkBuilder};
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GlobInput {
    path: Option<String>,
    pattern: String,
}

/// The maximum number of nodes to visit when searching for files.
const MAX_GLOB_VISITED_NODES: usize = 100000;
/// Most matches a single glob call will return; the newest are kept.
const MAX_GLOB_MATCHES: usize = 250;
/// The maximum size (in bytes) allowed to return for glob lists.
const MAX_GLOB_OUTPUT_BYTES: usize = 32 * 1024;

#[derive(Clone, Copy, Debug)]
struct GlobLimits {
    matches: usize,
    output_bytes: usize,
    visited_nodes: usize,
}

pub struct GlobTool {
    workspace: Arc<Workspace>,
    limits: GlobLimits,
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
                 a truncated result says so explicitly; narrow the pattern and search \
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
        Self {
            workspace,
            limits: GlobLimits {
                matches: MAX_GLOB_MATCHES,
                output_bytes: MAX_GLOB_OUTPUT_BYTES,
                visited_nodes: MAX_GLOB_VISITED_NODES,
            },
        }
    }

    fn prepare_glob(&self, input: Value) -> Result<PreparedGlob, String> {
        let input: GlobInput =
            serde_json::from_value(input).map_err(|error| invalid_input("glob", error))?;

        if input.pattern.is_empty() {
            return Err(invalid_input("glob", "`pattern` must not be empty"));
        }

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

        let matcher = GlobBuilder::new(&input.pattern)
            .backslash_escape(true)
            .case_insensitive(false)
            .literal_separator(true)
            .build()
            .map_err(|error| invalid_input("glob", error))?
            .compile_matcher();

        Ok(PreparedGlob {
            limits: self.limits,
            matcher,
            requested_path,
            resolved_path,
            workspace_path: self.workspace.root().to_path_buf(),
        })
    }
}

#[derive(Debug)]
struct PreparedGlob {
    limits: GlobLimits,
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
            limits,
            matcher,
            requested_path,
            resolved_path,
            workspace_path,
        } = *self;

        let result = tokio::task::spawn_blocking(move || {
            glob_files(
                cancel.clone(),
                &matcher,
                &resolved_path,
                &workspace_path,
                limits.visited_nodes,
                limits.matches,
            )
        })
        .await
        .map_err(|error| background_task_failed("glob", &requested_path, error))?;

        let result = match result {
            Ok(paths) => paths,
            Err(GlobError::Cancelled) => {
                return Err(ToolExecutionError::Cancelled);
            }
            Err(error) => {
                return Err(operation_failed("glob", &requested_path, error).into());
            }
        };

        Ok(format_result_output(result, limits.output_bytes))
    }
}

#[derive(Debug, Error)]
enum GlobError {
    #[error("glob was cancelled")]
    Cancelled,

    #[error("failed to get metadata for root path `{path}`: {source}")]
    RootMetadata {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("root path is not a directory: {0}")]
    RootNotDirectory(PathBuf),

    #[error(
        "glob traversal exceeded the {limit}-entry limit; \
       choose a narrower search path"
    )]
    TooManyFiles { limit: usize },

    #[error("failed while traversing files: {0}")]
    Traversal(#[from] ignore::Error),

    #[error("walked path `{path}` was outside expected root `{root}`")]
    UnexpectedPath { path: PathBuf, root: PathBuf },
}

#[derive(Debug)]
struct GlobMatch {
    modified: Option<SystemTime>,
    output_path: String,
}

#[derive(Debug, PartialEq)]
enum GlobResult {
    Full(Vec<String>),
    Truncated {
        paths: Vec<String>,
        total_matches: usize,
    },
}

fn glob_files(
    cancel: CancellationToken,
    matcher: &GlobMatcher,
    resolved_path: &Path,
    workspace_path: &Path,
    max_visited_nodes: usize,
    max_matches: usize,
) -> Result<GlobResult, GlobError> {
    let metadata = std::fs::metadata(resolved_path).map_err(|source| GlobError::RootMetadata {
        path: resolved_path.to_path_buf(),
        source,
    })?;

    if !metadata.is_dir() {
        return Err(GlobError::RootNotDirectory(resolved_path.to_path_buf()));
    }

    let mut builder = WalkBuilder::new(resolved_path);

    builder
        .hidden(false)
        .parents(false)
        .ignore(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .add_custom_ignore_filename(".gitignore")
        .follow_links(false)
        .filter_entry(|entry| entry.file_name() != ".git");

    let mut matches = collect_matches(
        &cancel,
        matcher,
        resolved_path,
        workspace_path,
        max_visited_nodes,
        builder.build(),
    )?;

    if cancel.is_cancelled() {
        return Err(GlobError::Cancelled);
    }

    // Sort the list by mtime
    matches.sort_by(|left, right| {
        right
            .modified
            .cmp(&left.modified)
            .then_with(|| left.output_path.cmp(&right.output_path))
    });

    if cancel.is_cancelled() {
        return Err(GlobError::Cancelled);
    }

    let found_paths = matches.len();
    let returned_paths: Vec<_> = matches
        .into_iter()
        .take(max_matches)
        .map(|m| m.output_path)
        .collect();

    if found_paths > returned_paths.len() {
        return Ok(GlobResult::Truncated {
            paths: returned_paths,
            total_matches: found_paths,
        });
    }

    Ok(GlobResult::Full(returned_paths))
}

/// Walk `entries`, collecting every regular file whose root-relative path
/// matches. The first walker error fails the whole call so an incomplete
/// listing is never returned as if it were complete.
fn collect_matches(
    cancel: &CancellationToken,
    matcher: &GlobMatcher,
    resolved_path: &Path,
    workspace_path: &Path,
    max_visited_nodes: usize,
    entries: impl Iterator<Item = Result<DirEntry, ignore::Error>>,
) -> Result<Vec<GlobMatch>, GlobError> {
    let mut matches = Vec::new();
    let mut visited_nodes = 0;

    for entry_result in entries {
        if cancel.is_cancelled() {
            return Err(GlobError::Cancelled);
        }

        let entry = entry_result?;

        // We don't filter out the root so WalkBuilder can load its .gitignore.
        // The root itself is not a visited search entry or a possible match.
        if entry.depth() == 0 {
            continue;
        }

        visited_nodes += 1;
        if visited_nodes > max_visited_nodes {
            return Err(GlobError::TooManyFiles {
                limit: max_visited_nodes,
            });
        }

        // Directories and symlinks are excluded.
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }

        // Match relative to the requested search root, not against an absolute path.
        let matching_path =
            entry
                .path()
                .strip_prefix(resolved_path)
                .map_err(|_| GlobError::UnexpectedPath {
                    path: entry.path().to_path_buf(),
                    root: resolved_path.to_path_buf(),
                })?;

        if matcher.is_match(matching_path) {
            // Results should be relative to the workspace.
            let result_path = entry.path().strip_prefix(workspace_path).map_err(|_| {
                GlobError::UnexpectedPath {
                    path: entry.path().to_path_buf(),
                    root: workspace_path.to_path_buf(),
                }
            })?;

            matches.push(GlobMatch {
                modified: entry
                    .metadata()
                    .ok()
                    .and_then(|metadata| metadata.modified().ok()),
                output_path: normalize_workspace_path(result_path),
            });
        }
    }

    Ok(matches)
}

fn format_result_output(result: GlobResult, max_bytes: usize) -> String {
    // Normalize both variants into the same representation.
    let (paths, total_matches) = match result {
        GlobResult::Full(paths) => {
            let total_matches = paths.len();
            (paths, total_matches)
        }
        GlobResult::Truncated {
            paths,
            total_matches,
        } => (paths, total_matches),
    };

    if total_matches == 0 {
        return "no files matched".to_string();
    }

    let available_paths = paths.len();
    let mut shown_paths = available_paths;

    loop {
        let match_truncated = available_paths < total_matches;
        let size_truncated = shown_paths < available_paths;

        // Taking a prefix preserves the newest-first order from glob_files.
        let mut output = paths[..shown_paths].join("\n");

        if match_truncated || size_truncated {
            let notice = if size_truncated {
                format!(
                    "[truncated: showing {shown_paths} most recently modified \
                       of {total_matches} matches; output limited to \
                       {max_bytes} bytes; narrow the pattern or search path]"
                )
            } else {
                format!(
                    "[truncated: showing {shown_paths} most recently modified \
                       of {total_matches} matches; narrow the pattern or search path]"
                )
            };

            if !output.is_empty() {
                output.push('\n');
            }

            output.push_str(&notice);
        }

        if output.len() <= max_bytes {
            return output;
        }

        if shown_paths == 0 {
            return format!(
                "[truncated: {total_matches} matches omitted because output \
                   exceeded the {max_bytes}-byte limit]"
            );
        }

        shown_paths -= 1;
    }
}

fn normalize_workspace_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolTestExt;
    use serde_json::json;
    use std::fs;
    use std::time::Duration;
    use tempfile::{TempDir, tempdir};

    fn glob_tool() -> (TempDir, GlobTool) {
        let root = tempdir().unwrap();
        let workspace = Arc::new(Workspace::new(root.path().to_path_buf()).unwrap());
        let tool = GlobTool::new(workspace);
        (root, tool)
    }

    fn generous_limits() -> GlobLimits {
        GlobLimits {
            matches: usize::MAX,
            output_bytes: usize::MAX,
            visited_nodes: usize::MAX,
        }
    }

    fn run_glob(
        tool: &GlobTool,
        input: Value,
        cancel: CancellationToken,
        limits: GlobLimits,
    ) -> Result<GlobResult, GlobError> {
        let prepared = tool.prepare_glob(input).unwrap();

        glob_files(
            cancel,
            &prepared.matcher,
            &prepared.resolved_path,
            &prepared.workspace_path,
            limits.visited_nodes,
            limits.matches,
        )
    }

    fn set_mtime(path: &Path, seconds_after_epoch: u64) {
        let file = fs::File::options().write(true).open(path).unwrap();
        file.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(seconds_after_epoch))
            .unwrap();
    }

    #[test]
    fn no_path_uses_workspace_root() {
        // Arrange
        let (_root, tool) = glob_tool();

        // Act
        let result = tool.prepare_glob(json!({ "pattern": "*" })).unwrap();

        // Assert
        assert_eq!(result.resolved_path, tool.workspace.root());
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
        fs::remove_dir(&root_path).unwrap();

        // Act
        let result = tool.prepare(json!({ "pattern": "*" })).await;

        // Assert
        assert!(result.is_ok());
        assert!(!root_path.exists());
    }

    #[test]
    fn output_reports_no_matches() {
        // Arrange
        let result = GlobResult::Full(vec![]);

        // Act
        let output = format_result_output(result, 1024);

        // Assert
        assert_eq!(output, "no files matched");
    }

    #[test]
    fn output_formats_paths_one_per_line() {
        // Arrange
        let result = GlobResult::Full(vec!["./src/lib.rs".to_string(), "./Cargo.toml".to_string()]);

        // Act
        let output = format_result_output(result, 1024);

        // Assert
        assert_eq!(output, "./src/lib.rs\n./Cargo.toml");
    }

    #[test]
    fn output_reports_match_truncation() {
        // Arrange
        let result = GlobResult::Truncated {
            paths: vec!["src/lib.rs".to_string(), "src/main.rs".to_string()],
            total_matches: 20,
        };

        // Act
        let output = format_result_output(result, 1024);

        // Assert
        assert!(output.contains("src/lib.rs\nsrc/main.rs"));
        assert!(output.contains("showing 2 most recently modified of 20 matches"));
    }

    #[test]
    fn output_reports_size_truncation() {
        // Arrange
        let paths: Vec<_> = (0..5)
            .map(|index| format!("src/file-{index}-{}.rs", "x".repeat(80)))
            .collect();

        let max_bytes = 256;

        let result = GlobResult::Full(paths);

        // Act
        let output = format_result_output(result, max_bytes);

        // Assert
        assert!(output.contains("[truncated: showing 1 most recently modified of 5 matches; output limited to 256 bytes; narrow the pattern or search path]"));
        assert!(output.len() <= max_bytes);
    }

    #[test]
    fn definition_describes_strict_glob_input() {
        // Arrange
        let (_root, tool) = glob_tool();

        // Act
        let definition = tool.definition();

        // Assert
        assert_eq!(definition.name, "glob");
        assert_eq!(definition.input_schema["type"], "object");
        assert_eq!(definition.input_schema["required"], json!(["pattern"]));
        assert_eq!(definition.input_schema["additionalProperties"], false);
        assert_eq!(
            definition.input_schema["properties"]["pattern"]["type"],
            "string"
        );
        assert_eq!(
            definition.input_schema["properties"]["path"]["type"],
            "string"
        );
    }

    #[test]
    fn preparation_rejects_a_path_outside_the_workspace() {
        // Arrange
        let (_root, tool) = glob_tool();
        let outside = tempdir().unwrap();

        // Act
        let error = tool
            .prepare_glob(json!({
                "pattern": "*",
                "path": outside.path().to_string_lossy(),
            }))
            .unwrap_err();

        // Assert
        assert!(error.contains("outside workspace root"), "{error}");
    }

    #[test]
    fn glob_rejects_a_missing_search_root() {
        // Arrange
        let (_root, tool) = glob_tool();

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "*", "path": "missing" }),
            CancellationToken::new(),
            generous_limits(),
        );

        // Assert
        assert!(matches!(
            result,
            Err(GlobError::RootMetadata { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[test]
    fn glob_rejects_a_search_root_that_is_a_file() {
        // Arrange
        let (root, tool) = glob_tool();
        fs::write(root.path().join("root.txt"), "content").unwrap();

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "*", "path": "root.txt" }),
            CancellationToken::new(),
            generous_limits(),
        );

        // Assert
        assert!(
            matches!(result, Err(GlobError::RootNotDirectory(path)) if path.ends_with("root.txt"))
        );
    }

    #[test]
    fn traversal_limit_is_an_error_with_the_injected_limit() {
        // Arrange
        let (root, tool) = glob_tool();
        fs::write(root.path().join("one.rs"), "one").unwrap();
        fs::write(root.path().join("two.rs"), "two").unwrap();
        let mut limits = generous_limits();
        limits.visited_nodes = 1;

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "*" }),
            CancellationToken::new(),
            limits,
        );

        // Assert
        assert!(matches!(result, Err(GlobError::TooManyFiles { limit: 1 })));
    }

    #[test]
    fn match_limit_returns_the_total_and_only_the_requested_number_of_paths() {
        // Arrange
        let (root, tool) = glob_tool();
        for name in ["one.rs", "two.rs", "three.rs"] {
            fs::write(root.path().join(name), name).unwrap();
        }
        let mut limits = generous_limits();
        limits.matches = 2;

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "*" }),
            CancellationToken::new(),
            limits,
        )
        .unwrap();

        // Assert
        assert!(matches!(
            result,
            GlobResult::Truncated {
                paths,
                total_matches: 3,
            } if paths.len() == 2
        ));
    }

    #[tokio::test]
    async fn pre_cancelled_execution_returns_cancellation() {
        // Arrange
        let (_root, tool) = glob_tool();
        let prepared = tool.prepare_glob(json!({ "pattern": "*" })).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();

        // Act
        let result = Box::new(prepared).execute(cancel).await;

        // Assert
        assert_eq!(result, Err(ToolExecutionError::Cancelled));
    }

    #[tokio::test]
    async fn execute_returns_a_workspace_relative_match() {
        // Arrange
        let (root, tool) = glob_tool();
        fs::write(root.path().join("main.rs"), "fn main() {}").unwrap();

        // Act
        let output = tool.execute(json!({ "pattern": "*.rs" })).await.unwrap();

        // Assert
        assert_eq!(output, "main.rs");
    }

    #[test]
    fn gitignore_rules_start_at_the_search_root() {
        // Arrange
        let (root, tool) = glob_tool();
        let search_root = root.path().join("search");
        fs::create_dir(&search_root).unwrap();
        fs::write(root.path().join(".gitignore"), "parent-ignored.txt\n").unwrap();
        fs::write(search_root.join(".gitignore"), "local-ignored.txt\n").unwrap();
        fs::write(search_root.join("parent-ignored.txt"), "visible").unwrap();
        fs::write(search_root.join("local-ignored.txt"), "ignored").unwrap();
        fs::write(search_root.join("visible.txt"), "visible").unwrap();

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "*.txt", "path": "search" }),
            CancellationToken::new(),
            generous_limits(),
        )
        .unwrap();

        // Assert
        let GlobResult::Full(paths) = result else {
            panic!("generous limits should not truncate this result");
        };
        assert!(paths.contains(&"search/parent-ignored.txt".to_string()));
        assert!(paths.contains(&"search/visible.txt".to_string()));
        assert!(!paths.contains(&"search/local-ignored.txt".to_string()));
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn direct_and_recursive_patterns_have_distinct_scope() {
        // Arrange
        let (root, tool) = glob_tool();
        let nested_root = root.path().join("nested");
        fs::create_dir(&nested_root).unwrap();

        let direct_path = root.path().join("direct.rs");
        let nested_path = nested_root.join("nested.rs");

        fs::write(&direct_path, "direct").unwrap();
        fs::write(&nested_path, "nested").unwrap();

        set_mtime(&direct_path, 1);
        set_mtime(&nested_path, 1);

        // Act
        let direct_result = run_glob(
            &tool,
            json!({ "pattern": "*.rs" }),
            CancellationToken::new(),
            generous_limits(),
        )
        .unwrap();
        let recursive_result = run_glob(
            &tool,
            json!({ "pattern": "**/*.rs" }),
            CancellationToken::new(),
            generous_limits(),
        )
        .unwrap();

        // Assert
        assert_eq!(
            direct_result,
            GlobResult::Full(vec!["direct.rs".to_string()],)
        );

        assert_eq!(
            recursive_result,
            GlobResult::Full(vec![
                "direct.rs".to_string(),
                "nested/nested.rs".to_string()
            ],)
        );
    }

    #[test]
    fn question_classes_alternation_escaping_and_case_sensitivity_match_the_contract() {
        // Arrange
        let (root, tool) = glob_tool();
        let files = vec![
            "a.txt",
            "b.txt",
            "c.txt",
            "file.txt",
            "{weird}.txt",
            "nested/a.txt",
        ];

        fs::create_dir(root.path().join("nested")).unwrap();

        for file in &files {
            fs::write(root.path().join(file), "exists").unwrap();
            set_mtime(&root.path().join(file), 1);
        }

        let cases = [
            (
                json!({ "pattern": "fi?e.txt" }),
                vec!["file.txt".to_string()],
            ),
            (json!({ "pattern": "*.TXT" }), vec![]),
            (json!({ "pattern": "nested?a.txt" }), vec![]),
            (
                json!({ "pattern": "\\{*\\}.txt" }),
                vec!["{weird}.txt".to_string()],
            ),
            (
                json!({ "pattern": "[ab].txt" }),
                vec!["a.txt".to_string(), "b.txt".to_string()],
            ),
            (
                json!({ "pattern": "{a,c}.txt" }),
                vec!["a.txt".to_string(), "c.txt".to_string()],
            ),
        ];

        // Act
        for (pattern, expected) in cases {
            let actual =
                run_glob(&tool, pattern, CancellationToken::new(), generous_limits()).unwrap();

            assert_eq!(actual, GlobResult::Full(expected));
        }
    }

    #[test]
    fn gitignore_is_honored_without_a_git_directory() {
        // Arrange
        let (root, tool) = glob_tool();
        fs::write(root.path().join(".gitignore"), "parent-ignored.txt\n").unwrap();
        fs::write(root.path().join("parent-ignored.txt"), "visible").unwrap();

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "*.txt" }),
            CancellationToken::new(),
            generous_limits(),
        )
        .unwrap();

        // Assert
        assert_eq!(result, GlobResult::Full(vec![]));
    }

    #[test]
    fn hidden_paths_are_visible_but_dot_git_is_pruned() {
        // Arrange
        let (root, tool) = glob_tool();

        fs::create_dir_all(root.path().join(".github/workflows")).unwrap();
        fs::create_dir_all(root.path().join(".git/objects")).unwrap();

        fs::write(root.path().join(".github/workflows/ci.yml"), "ci").unwrap();
        fs::write(root.path().join(".git/objects/example"), "example").unwrap();

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "**/*" }),
            CancellationToken::new(),
            generous_limits(),
        )
        .unwrap();

        // Assert
        assert_eq!(
            result,
            GlobResult::Full(vec![".github/workflows/ci.yml".to_string(),])
        );
    }

    #[test]
    fn matches_are_ordered_by_mtime_then_portable_lexical_path() {
        // Arrange
        let (root, tool) = glob_tool();

        let path = root.path().join("a.txt");
        fs::write(&path, "a").unwrap();
        set_mtime(&path, 1);

        let path = root.path().join("b.txt");
        fs::write(&path, "b").unwrap();
        set_mtime(&path, 2);

        let path = root.path().join("c.txt");
        fs::write(&path, "c").unwrap();
        set_mtime(&path, 3);

        let path = root.path().join("d.txt");
        fs::write(&path, "d").unwrap();
        set_mtime(&path, 3);

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "*.txt" }),
            CancellationToken::new(),
            generous_limits(),
        )
        .unwrap();

        // Assert
        assert_eq!(
            result,
            GlobResult::Full(vec![
                "c.txt".to_string(),
                "d.txt".to_string(),
                "b.txt".to_string(),
                "a.txt".to_string()
            ])
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_skipped_and_non_utf8_names_are_formatted_lossily() {
        use std::os::unix::ffi::OsStrExt;

        // Arrange
        let (root, tool) = glob_tool();

        let exists_path = root.path().join("exists.txt");
        let bad_utf8_bytes = b"why_\xFF.txt";
        let bad_utf8_name = std::ffi::OsStr::from_bytes(bad_utf8_bytes);
        let non_utf8_path = root.path().join(bad_utf8_name);

        fs::write(&exists_path, "exists").unwrap();
        set_mtime(&exists_path, 3);
        fs::write(&non_utf8_path, "exists").unwrap();
        set_mtime(&non_utf8_path, 3);
        std::os::unix::fs::symlink(exists_path, root.path().join("linked.txt")).unwrap();

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "*.txt" }),
            CancellationToken::new(),
            generous_limits(),
        )
        .unwrap();

        // Assert
        assert_eq!(
            result,
            GlobResult::Full(vec![
                "exists.txt".to_string(),
                "why_\u{FFFD}.txt".to_string()
            ])
        );
    }

    #[cfg(unix)]
    #[test]
    fn matching_is_performed_against_non_lossy_version() {
        use std::os::unix::ffi::OsStrExt;

        // Arrange
        let (root, tool) = glob_tool();

        let bad_utf8_bytes = b"why_\xFF.txt";
        let bad_utf8_name = std::ffi::OsStr::from_bytes(bad_utf8_bytes);
        let non_utf8_path = root.path().join(bad_utf8_name);

        fs::write(&non_utf8_path, "exists").unwrap();
        set_mtime(&non_utf8_path, 3);

        // Act
        let result = run_glob(
            &tool,
            json!({ "pattern": "why_\u{FFFD}.txt" }),
            CancellationToken::new(),
            generous_limits(),
        )
        .unwrap();

        // Assert
        assert_eq!(result, GlobResult::Full(vec![]));
    }

    #[test]
    fn unreadable_directories_fail_instead_of_returning_partial_results() {
        // Arrange
        let (root, tool) = glob_tool();
        fs::write(root.path().join("a.txt"), "a").unwrap();
        fs::write(root.path().join("b.txt"), "b").unwrap();
        let prepared = tool.prepare_glob(json!({ "pattern": "*.txt" })).unwrap();

        // Both matches are visited before the error, as when a walker fails on
        // an unreadable directory after listing readable siblings.
        let unreadable = ignore::Error::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "permission denied",
        ));
        let entries = WalkBuilder::new(&prepared.resolved_path)
            .build()
            .chain(std::iter::once(Err(unreadable)));

        // Act
        let result = collect_matches(
            &CancellationToken::new(),
            &prepared.matcher,
            &prepared.resolved_path,
            &prepared.workspace_path,
            usize::MAX,
            entries,
        );

        // Assert
        assert!(matches!(result, Err(GlobError::Traversal(_))), "{result:?}");
    }

    #[test]
    fn cancellation_during_a_walk_stops_promptly() {
        // Arrange
        let (root, tool) = glob_tool();
        fs::write(root.path().join("a.txt"), "a").unwrap();
        fs::write(root.path().join("b.txt"), "b").unwrap();
        let prepared = tool.prepare_glob(json!({ "pattern": "*.txt" })).unwrap();

        let mut walked = 0;
        let cancel = CancellationToken::new();
        let entries = WalkBuilder::new(&prepared.resolved_path)
            .build()
            .inspect(|_entry| {
                walked += 1;
                cancel.cancel();
            });

        // Act
        let result = collect_matches(
            &cancel,
            &prepared.matcher,
            &prepared.resolved_path,
            &prepared.workspace_path,
            usize::MAX,
            entries,
        );

        // Assert
        assert_eq!(walked, 1);
        assert!(matches!(result, Err(GlobError::Cancelled)), "{result:?}");
    }
}
