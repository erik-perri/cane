use crate::Workspace;
use crate::protocol::ApprovalRequirement;
use crate::tools::file_discovery::{FileDiscoveryError, FileSelector, LocatedFile, locate_files};
use crate::tools::path_display::{compare_located_files, render_workspace_path};
use crate::tools::{
    MAX_FILE_SIZE_MIB, PreparedInvocation, Tool, ToolDefinition, ToolExecutionError,
    ToolExecutionOutput, background_task_failed, invalid_input, operation_failed,
};
use grep_regex::{RegexMatcher, RegexMatcherBuilder};
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

mod output;
mod scan;

use output::{append_recursive_include_hint, format_results};
use scan::{GrepCandidate, ScanOptions, scan_candidates};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GrepInput {
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default)]
    context: usize,
    include: Option<String>,
    #[serde(default)]
    multiline: bool,
    #[serde(default)]
    output_mode: OutputMode,
    path: Option<String>,
    pattern: String,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum OutputMode {
    #[default]
    Content,
    FilesWithMatches,
}

/// Most matching paths a single grep call will return; the newest are kept.
const MAX_GREP_PATHS: usize = 250;
/// Most matched source lines returned from one file.
const MAX_GREP_MATCHED_LINES_PER_FILE: usize = 50;
/// Most matched source lines returned across one content search.
const MAX_GREP_MATCHED_LINES: usize = 200;
/// The maximum size (in bytes) allowed to return for grep lists.
const MAX_GREP_OUTPUT_BYTES: usize = 32 * 1024;
/// Most context lines allowed before and after a match.
const MAX_GREP_CONTEXT: usize = 10;
/// Most decoded source-text bytes rendered for one source line.
const MAX_GREP_SOURCE_LINE_BYTES: usize = 2 * 1024;

const NO_MATCHES: &str = "No matches found.";
const RECURSIVE_INCLUDE_HINT: &str = "[hint: include patterns without `/` match only direct children; prefix with `**/` to match recursively]";
const BINARY_FILE_SKIPPED: &str = "binary file skipped";
const OMITTED_SOURCE_MARKER: &str = "[...]";
const LOSSY_SOURCE_WARNING: &str =
    "[warning: invalid UTF-8 in returned source lines was replaced with U+FFFD]";
const EXCERPT_WARNING: &str =
    "[warning: [...] marks source text omitted by grep and is not source content]";
const LOSSY_PATH_WARNING: &str = "[warning: displayed paths containing replacement characters may not round-trip through \
     another file tool]";
const ESCAPED_PATH_WARNING: &str = "[warning: JSON-quoted paths must be decoded before reuse]";

#[derive(Clone, Copy, Debug)]
struct GrepLimits {
    matched_lines: usize,
    matched_lines_per_file: usize,
    output_bytes: usize,
    paths: usize,
    source_line_bytes: usize,
}

pub(super) struct GrepTool {
    limits: GrepLimits,
    workspace: Arc<Workspace>,
}

#[derive(Debug)]
struct PreparedGrep {
    context: usize,
    include_has_no_separator: bool,
    limits: GrepLimits,
    matcher: RegexMatcher,
    multiline: bool,
    output_mode: OutputMode,
    requested_path: String,
    search_root: PathBuf,
    selector: FileSelector,
    workspace_root: PathBuf,
}

#[async_trait::async_trait]
impl Tool for GrepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".to_string(),
            description: format!(
                "Search file contents in the workspace using a Rust regular expression. Returns \
                 complete matching source lines with workspace-relative paths and 1-based line \
                 numbers, or only the paths of files with matches. Directory searches exclude \
                 files ignored by .gitignore, skip .git and symlinks, include hidden files, and \
                 silently skip binary files. Files larger than {MAX_FILE_SIZE_MIB} MiB are not \
                 searched. Include globs are relative to the search directory: patterns without \
                 `/` match only direct children; prefix with `**/` to match recursively. Results \
                 are grouped by file, most recently modified first. Content output returns at \
                 most {MAX_GREP_MATCHED_LINES} matched lines; file-list output returns at most \
                 {MAX_GREP_PATHS} paths. A truncated result says so explicitly; narrow path, \
                 include, or pattern and search again. Finding no matches is a normal result, \
                 not an error."
            ),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or directory to search, relative to the workspace \
                            root (or an absolute path inside the workspace). Omit to search the \
                            entire workspace.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Rust regular expression matched against file contents. \
                            Matching is case-sensitive and line-oriented by default. Inline \
                            flags such as `(?i)` are supported; `/pattern/i` syntax is not.",
                    },
                    "include": {
                        "type": "string",
                        "description": "Optional glob pattern restricting which files are \
                            searched, matched against paths relative to the search directory. \
                            Uses the same case-sensitive syntax as glob: always use `/` as the \
                            separator, and `*` does not cross `/`. `*.rs` matches only direct \
                            children; `**/*.rs` matches recursively. Omit to search all files.",
                    },
                    "output_mode": {
                        "type": "string",
                        "description": "Return complete matching source lines grouped by file \
                            (`content`) or only workspace-relative paths of files containing a \
                            match (`files_with_matches`). Defaults to `content`.",
                        "enum": ["content", "files_with_matches"],
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Match without distinguishing uppercase and lowercase \
                            letters. Defaults to false.",
                    },
                    "multiline": {
                        "type": "boolean",
                        "description": "Allow matches to cross line boundaries. `.` still does \
                            not match a newline unless the pattern enables dot-all mode with \
                            `(?s)`. Defaults to false.",
                    },
                    "context": {
                        "type": "integer",
                        "description": "Number of source lines to include before and after each \
                            match. Must be zero with `files_with_matches`. Defaults to 0.",
                        "minimum": 0,
                        "maximum": MAX_GREP_CONTEXT,
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false,
            }),
        }
    }

    async fn prepare(&self, input: Value) -> Result<Box<dyn PreparedInvocation>, String> {
        let tool = self.prepare_grep(input)?;

        Ok(Box::new(tool))
    }
}

impl GrepTool {
    pub(super) fn new(workspace: Arc<Workspace>) -> Self {
        Self {
            limits: GrepLimits {
                matched_lines: MAX_GREP_MATCHED_LINES,
                matched_lines_per_file: MAX_GREP_MATCHED_LINES_PER_FILE,
                output_bytes: MAX_GREP_OUTPUT_BYTES,
                paths: MAX_GREP_PATHS,
                source_line_bytes: MAX_GREP_SOURCE_LINE_BYTES,
            },
            workspace,
        }
    }

    fn prepare_grep(&self, input: Value) -> Result<PreparedGrep, String> {
        let input: GrepInput =
            serde_json::from_value(input).map_err(|error| invalid_input("grep", error))?;

        if input.pattern.is_empty() {
            return Err(invalid_input("grep", "`pattern` must not be empty"));
        }

        if input.context > MAX_GREP_CONTEXT {
            return Err(invalid_input(
                "grep",
                format_args!("`context` must be at most {MAX_GREP_CONTEXT}"),
            ));
        }

        if input.output_mode == OutputMode::FilesWithMatches && input.context != 0 {
            return Err(invalid_input(
                "grep",
                "`context` must be zero with `files_with_matches`",
            ));
        }

        let (requested_path, resolved_path) = match input.path {
            Some(path) if path.is_empty() => {
                return Err(invalid_input("grep", "`path` must not be empty"));
            }
            Some(path) => {
                let resolved = self.workspace.resolve(&path)?;
                (path, resolved)
            }
            None => (".".to_owned(), self.workspace.root().to_path_buf()),
        };

        if resolved_path
            .components()
            .any(|component| component.as_os_str() == ".git")
        {
            return Err(invalid_input(
                "grep",
                "path must not be inside a `.git` directory",
            ));
        }

        let (selector, include_has_no_separator) = match input.include {
            Some(pattern) if pattern.is_empty() => {
                return Err(invalid_input("grep", "`include` must not be empty"));
            }
            Some(pattern) => (
                FileSelector::glob(&pattern).map_err(|error| invalid_input("grep", error))?,
                !pattern.contains('/'),
            ),
            None => (FileSelector::all(), false),
        };

        let matcher = compile_matcher(&input.pattern, input.case_insensitive, input.multiline)
            .map_err(|error| {
                let mut detail = error.to_string();

                if !input.multiline && detail == r#"the literal "\n" is not allowed in a regex"# {
                    detail.push_str("; set `multiline: true` to allow matches across lines");
                }

                invalid_input("grep", detail)
            })?;

        Ok(PreparedGrep {
            context: input.context,
            include_has_no_separator,
            limits: self.limits,
            matcher,
            multiline: input.multiline,
            output_mode: input.output_mode,
            requested_path,
            search_root: resolved_path,
            selector,
            workspace_root: self.workspace.root().to_path_buf(),
        })
    }
}

fn compile_matcher(
    pattern: &str,
    case_insensitive: bool,
    multiline: bool,
) -> Result<RegexMatcher, grep_regex::Error> {
    let mut builder = RegexMatcherBuilder::new();

    builder
        // Give ^ and $ normal grep line-anchor behavior.
        .multi_line(true)
        .case_insensitive(case_insensitive)
        .dot_matches_new_line(false)
        // Line mode rejects patterns that explicitly require crossing \n.
        .line_terminator(if multiline { None } else { Some(b'\n') });

    builder.build(pattern)
}

struct CandidateSet {
    explicit_file: bool,
    values: Vec<GrepCandidate>,
}

async fn discover_candidates(
    cancel: &CancellationToken,
    requested_path: &str,
    search_root: PathBuf,
    selector: FileSelector,
    workspace_root: &Path,
) -> Result<CandidateSet, ToolExecutionError> {
    let metadata_path = search_root.clone();
    let metadata_requested_path = requested_path.to_string();
    let search_root_metadata = tokio::task::spawn_blocking(move || fs::metadata(metadata_path))
        .await
        .map_err(|error| background_task_failed("grep", &metadata_requested_path, error))?
        .map_err(|error| operation_failed("grep", requested_path, error))?;
    let explicit_file = search_root_metadata.is_file();

    let files = match search_root_metadata.file_type() {
        file_type if file_type.is_file() => {
            let file_name = search_root.file_name().ok_or_else(|| {
                operation_failed("grep", requested_path, "resolved file path has no basename")
            })?;

            if selector.matches(Path::new(file_name)) {
                vec![LocatedFile::from_metadata(
                    search_root,
                    search_root_metadata,
                )]
            } else {
                Vec::new()
            }
        }
        file_type if file_type.is_dir() => {
            match locate_files(
                cancel.clone(),
                workspace_root.to_path_buf(),
                search_root,
                selector,
            )
            .await
            {
                Ok(files) => files,
                Err(FileDiscoveryError::Cancelled) => {
                    return Err(ToolExecutionError::Cancelled);
                }
                Err(error) => {
                    return Err(operation_failed("grep", requested_path, error).into());
                }
            }
        }
        _ => {
            return Err(operation_failed(
                "grep",
                requested_path,
                "explicit target is neither a regular file nor directory",
            )
            .into());
        }
    };

    if cancel.is_cancelled() {
        return Err(ToolExecutionError::Cancelled);
    }

    let mut values = files
        .into_iter()
        .map(|file| -> Result<GrepCandidate, ToolExecutionError> {
            let rendered_path =
                render_workspace_path(workspace_root, &file.path).map_err(|error| {
                    ToolExecutionError::ToolError(operation_failed("grep", requested_path, error))
                })?;

            Ok(GrepCandidate {
                file,
                rendered_path,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    values.sort_by(|left, right| {
        compare_located_files(
            &left.file,
            &left.rendered_path,
            &right.file,
            &right.rendered_path,
        )
    });

    Ok(CandidateSet {
        explicit_file,
        values,
    })
}

#[async_trait::async_trait]
impl PreparedInvocation for PreparedGrep {
    fn approval_requirement(&self) -> ApprovalRequirement {
        ApprovalRequirement::None
    }

    async fn execute(
        self: Box<Self>,
        cancel: CancellationToken,
    ) -> Result<ToolExecutionOutput, ToolExecutionError> {
        if cancel.is_cancelled() {
            return Err(ToolExecutionError::Cancelled);
        }

        let Self {
            context,
            include_has_no_separator,
            limits,
            matcher,
            multiline,
            output_mode,
            requested_path,
            search_root,
            selector,
            workspace_root,
        } = *self;

        let candidates = discover_candidates(
            &cancel,
            &requested_path,
            search_root,
            selector,
            &workspace_root,
        )
        .await?;
        let show_recursive_include_hint =
            include_has_no_separator && !candidates.explicit_file && candidates.values.is_empty();

        let scan_cancel = cancel.clone();
        let scan_requested_path = requested_path.clone();
        let scan = tokio::task::spawn_blocking(move || {
            scan_candidates(
                candidates.values,
                ScanOptions {
                    cancel: scan_cancel,
                    context,
                    explicit_file: candidates.explicit_file,
                    limits,
                    matcher,
                    multiline,
                    output_mode,
                    requested_path: scan_requested_path,
                },
            )
        })
        .await
        .map_err(|error| background_task_failed("grep", &requested_path, error))??;

        if cancel.is_cancelled() {
            return Err(ToolExecutionError::Cancelled);
        }

        let mut formatted_results = format_results(
            scan.results,
            output_mode,
            limits,
            scan.warnings,
            scan.truncation,
        );
        if show_recursive_include_hint {
            formatted_results =
                append_recursive_include_hint(formatted_results, limits.output_bytes);
        }

        if cancel.is_cancelled() {
            return Err(ToolExecutionError::Cancelled);
        }

        Ok(ToolExecutionOutput::text(formatted_results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::MAX_FILE_SIZE_BYTES;
    use crate::tools::ToolTestExt;
    use serde_json::json;
    use std::time::{Duration, SystemTime};
    use tempfile::{TempDir, tempdir};

    fn grep_tool() -> (TempDir, GrepTool) {
        let root = tempdir().unwrap();
        let workspace = Arc::new(Workspace::new(root.path().to_path_buf()).unwrap());
        let tool = GrepTool::new(workspace);
        (root, tool)
    }

    fn set_mtime(path: &Path, seconds_after_epoch: u64) {
        let file = fs::File::options().write(true).open(path).unwrap();
        file.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(seconds_after_epoch))
            .unwrap();
    }

    #[test]
    fn definition_describes_the_strict_grep_contract() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let definition = tool.definition();

        // Assert
        assert_eq!(definition.name, "grep");
        assert!(
            definition
                .description
                .contains("prefix with `**/` to match recursively")
        );
        assert_eq!(definition.input_schema["type"], "object");
        assert_eq!(definition.input_schema["required"], json!(["pattern"]));
        assert_eq!(definition.input_schema["additionalProperties"], false);
        assert_eq!(
            definition.input_schema["properties"]
                .as_object()
                .unwrap()
                .len(),
            7
        );
        for property in ["path", "pattern", "include"] {
            assert_eq!(
                definition.input_schema["properties"][property]["type"],
                "string"
            );
        }
        for property in ["case_insensitive", "multiline"] {
            assert_eq!(
                definition.input_schema["properties"][property]["type"],
                "boolean"
            );
        }
        assert_eq!(
            definition.input_schema["properties"]["output_mode"]["type"],
            "string"
        );
        assert_eq!(
            definition.input_schema["properties"]["context"]["type"],
            "integer"
        );
        assert_eq!(
            definition.input_schema["properties"]["output_mode"]["enum"],
            json!(["content", "files_with_matches"])
        );
        assert_eq!(
            definition.input_schema["properties"]["context"]["minimum"],
            0
        );
        assert_eq!(
            definition.input_schema["properties"]["context"]["maximum"],
            10
        );
        assert!(
            definition.input_schema["properties"]["include"]["description"]
                .as_str()
                .unwrap()
                .contains("`*.rs` matches only direct children; `**/*.rs` matches recursively")
        );
    }

    #[test]
    fn omitted_options_use_the_documented_defaults() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let prepared = tool.prepare_grep(json!({ "pattern": "needle" })).unwrap();

        // Assert
        assert_eq!(prepared.requested_path, ".");
        assert_eq!(prepared.search_root, tool.workspace.root());
        assert_eq!(prepared.workspace_root, tool.workspace.root());
        assert_eq!(prepared.context, 0);
        assert!(!prepared.multiline);
        assert_eq!(prepared.output_mode, OutputMode::Content);
        assert!(prepared.selector.matches(Path::new("any/path.txt")));
        assert_eq!(prepared.limits.matched_lines, MAX_GREP_MATCHED_LINES);
        assert_eq!(
            prepared.limits.matched_lines_per_file,
            MAX_GREP_MATCHED_LINES_PER_FILE
        );
        assert_eq!(prepared.limits.output_bytes, MAX_GREP_OUTPUT_BYTES);
        assert_eq!(prepared.limits.paths, MAX_GREP_PATHS);
        assert_eq!(
            prepared.limits.source_line_bytes,
            MAX_GREP_SOURCE_LINE_BYTES
        );
    }

    #[test]
    fn serde_rejects_missing_wrong_and_unknown_input() {
        // Arrange
        let (_root, tool) = grep_tool();
        let cases = [
            (json!({}), "invalid grep input: missing field `pattern`"),
            (
                json!({ "pattern": 7 }),
                "invalid grep input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "pattern": "needle", "case_insensitive": "yes" }),
                "invalid grep input: invalid type: string \"yes\", expected a boolean",
            ),
            (
                json!({ "pattern": "needle", "path": 7 }),
                "invalid grep input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "pattern": "needle", "include": 7 }),
                "invalid grep input: invalid type: integer `7`, expected a string",
            ),
            (
                json!({ "pattern": "needle", "multiline": "yes" }),
                "invalid grep input: invalid type: string \"yes\", expected a boolean",
            ),
            (
                json!({ "pattern": "needle", "context": -1 }),
                "invalid grep input: invalid value: integer `-1`, expected usize",
            ),
            (
                json!({ "pattern": "needle", "context": 1.5 }),
                "invalid grep input: invalid type: floating point `1.5`, expected usize",
            ),
            (
                json!({ "pattern": "needle", "output_mode": "paths" }),
                "invalid grep input: unknown variant `paths`, expected `content` or \
                 `files_with_matches`",
            ),
            (
                json!({ "pattern": "needle", "unexpected": true }),
                "invalid grep input: unknown field `unexpected`, expected one of \
                 `case_insensitive`, `context`, `include`, `multiline`, `output_mode`, `path`, \
                 `pattern`",
            ),
            (
                json!("not an object"),
                "invalid grep input: invalid type: string \"not an object\", expected struct \
                 GrepInput",
            ),
            (
                json!(null),
                "invalid grep input: invalid type: null, expected struct GrepInput",
            ),
        ];

        for (input, expected_error) in cases {
            // Act
            let error = tool.prepare_grep(input).unwrap_err();

            // Assert
            assert_eq!(error, expected_error);
        }
    }

    #[test]
    fn empty_pattern_path_and_include_are_rejected() {
        // Arrange
        let (_root, tool) = grep_tool();
        let cases = [
            (
                json!({ "pattern": "" }),
                "invalid grep input: `pattern` must not be empty",
            ),
            (
                json!({ "pattern": "needle", "path": "" }),
                "invalid grep input: `path` must not be empty",
            ),
            (
                json!({ "pattern": "needle", "include": "" }),
                "invalid grep input: `include` must not be empty",
            ),
        ];

        for (input, expected_error) in cases {
            // Act
            let error = tool.prepare_grep(input).unwrap_err();

            // Assert
            assert_eq!(error, expected_error);
        }
    }

    #[test]
    fn malformed_regex_and_include_are_rejected_during_preparation() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let regex_error = tool.prepare_grep(json!({ "pattern": "(" })).unwrap_err();
        let include_error = tool
            .prepare_grep(json!({ "pattern": "needle", "include": "[" }))
            .unwrap_err();

        // Assert
        assert_eq!(
            regex_error,
            "invalid grep input: regex parse error:\n    (?:()\n    ^\nerror: unclosed group"
        );
        assert_eq!(
            include_error,
            "invalid grep input: error parsing glob '[': unclosed character class; missing ']'"
        );
    }

    #[test]
    fn include_uses_the_same_path_relative_glob_rules_as_glob() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let direct = tool
            .prepare_grep(json!({ "pattern": "needle", "include": "*.rs" }))
            .unwrap();
        let recursive = tool
            .prepare_grep(json!({ "pattern": "needle", "include": "**/*.rs" }))
            .unwrap();

        // Assert
        assert!(direct.selector.matches(Path::new("main.rs")));
        assert!(!direct.selector.matches(Path::new("src/main.rs")));
        assert!(recursive.selector.matches(Path::new("main.rs")));
        assert!(recursive.selector.matches(Path::new("src/main.rs")));
        assert!(!recursive.selector.matches(Path::new("src/main.txt")));
    }

    #[test]
    fn line_mode_rejects_newline_patterns_and_multiline_accepts_them() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let error = tool
            .prepare_grep(json!({ "pattern": "first\\nsecond" }))
            .unwrap_err();
        let prepared = tool
            .prepare_grep(json!({
                "pattern": "first\\nsecond",
                "multiline": true
            }))
            .unwrap();

        // Assert
        assert_eq!(
            error,
            "invalid grep input: the literal \"\\n\" is not allowed in a regex; set \
             `multiline: true` to allow matches across lines"
        );
        assert!(prepared.multiline);
    }

    #[test]
    fn zero_width_and_inline_flag_patterns_are_accepted() {
        // Arrange
        let (_root, tool) = grep_tool();

        for pattern in ["^", "$", r"\b", "(?i)needle"] {
            // Act
            let result = tool.prepare_grep(json!({ "pattern": pattern }));

            // Assert
            assert!(
                result.is_ok(),
                "expected {pattern:?} to compile: {result:?}"
            );
        }
    }

    #[test]
    fn context_at_the_schema_maximum_is_accepted() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let prepared = tool
            .prepare_grep(json!({ "pattern": "needle", "context": 10 }))
            .unwrap();

        // Assert
        assert_eq!(prepared.context, 10);
    }

    #[test]
    fn preparation_resolves_paths_but_does_not_read_the_target() {
        // Arrange
        let (root, tool) = grep_tool();
        root.close().unwrap();

        // Act
        let prepared = tool.prepare_grep(json!({ "pattern": "needle" })).unwrap();

        // Assert
        assert_eq!(prepared.requested_path, ".");
    }

    #[test]
    fn paths_outside_the_workspace_and_inside_dot_git_are_rejected() {
        // Arrange
        let (root, tool) = grep_tool();
        let outside = tempdir().unwrap();
        fs::create_dir_all(root.path().join(".git/objects")).unwrap();

        // Act
        let outside_error = tool
            .prepare_grep(json!({
                "pattern": "needle",
                "path": outside.path().to_string_lossy()
            }))
            .unwrap_err();
        let git_error = tool
            .prepare_grep(json!({ "pattern": "needle", "path": ".git/objects" }))
            .unwrap_err();

        // Assert
        assert_eq!(
            outside_error,
            format!(
                "access denied: path `{}` is outside workspace root `{}`",
                outside.path().display(),
                tool.workspace.root().display()
            )
        );
        assert_eq!(
            git_error,
            "invalid grep input: path must not be inside a `.git` directory"
        );
    }

    #[tokio::test]
    async fn grep_never_requires_approval() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let prepared = tool.prepare(json!({ "pattern": "needle" })).await.unwrap();

        // Assert
        assert_eq!(prepared.approval_requirement(), ApprovalRequirement::None);
    }

    #[tokio::test]
    async fn pre_cancelled_execution_returns_cancellation() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(root.path().join("main.rs"), "needle\n").unwrap();
        let prepared = tool
            .prepare(json!({ "pattern": "needle", "path": "main.rs" }))
            .await
            .unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();

        // Act
        let result = prepared.execute(cancel).await;

        // Assert
        assert_eq!(result.unwrap_err(), ToolExecutionError::Cancelled);
    }

    #[test]
    fn context_must_not_exceed_the_schema_maximum() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let result = tool.prepare_grep(json!({ "pattern": "needle", "context": 11 }));

        // Assert
        let Err(error) = result else {
            panic!("expected context above 10 to be rejected during preparation");
        };
        assert_eq!(error, "invalid grep input: `context` must be at most 10");
    }

    #[test]
    fn files_with_matches_rejects_nonzero_context() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let result = tool.prepare_grep(json!({
            "pattern": "needle",
            "output_mode": "files_with_matches",
            "context": 1
        }));

        // Assert
        let Err(error) = result else {
            panic!("expected files_with_matches with context to be rejected during preparation");
        };
        assert_eq!(
            error,
            "invalid grep input: `context` must be zero with `files_with_matches`"
        );
    }

    #[tokio::test]
    async fn exact_file_content_output_uses_the_documented_line_grammar() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(root.path().join("main.rs"), "before\nneedle\nafter\n").unwrap();

        // Act
        let output = tool
            .execute(json!({ "pattern": "needle", "path": "main.rs" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, "main.rs:\n  2: needle");
    }

    #[tokio::test]
    async fn no_match_is_a_success_with_the_exact_sentinel() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(root.path().join("main.rs"), "nothing here\n").unwrap();

        // Act
        let output = tool
            .execute(json!({ "pattern": "needle", "path": "main.rs" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, "No matches found.");
    }

    #[tokio::test]
    async fn content_context_uses_markers_and_separates_disjoint_ranges() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(
            root.path().join("main.rs"),
            "zero\nneedle\ntwo\nthree\nfour\nneedle\nsix\n",
        )
        .unwrap();

        // Act
        let output = tool
            .execute(json!({
                "pattern": "needle",
                "path": "main.rs",
                "context": 1
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(
            output,
            "main.rs:\n  1- zero\n  2: needle\n  3- two\n  --\n  5- four\n  6: needle\n  7- six"
        );
    }

    #[tokio::test]
    async fn files_with_matches_is_newest_first_and_honors_the_injected_path_limit() {
        // Arrange
        let (root, mut tool) = grep_tool();
        let old = root.path().join("old.rs");
        let new = root.path().join("new.rs");
        fs::write(&old, "needle\n").unwrap();
        fs::write(&new, "needle\n").unwrap();
        set_mtime(&old, 1);
        set_mtime(&new, 2);
        tool.limits = GrepLimits {
            paths: 1,
            output_bytes: usize::MAX,
            ..tool.limits
        };

        // Act
        let output = tool
            .execute(json!({
                "pattern": "needle",
                "output_mode": "files_with_matches"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(
            output,
            "new.rs\n[truncated: showing 1 most recently modified matching path; more matches \
             may exist; narrow path, include, or pattern and search again]"
        );
    }

    #[tokio::test]
    async fn explicit_file_include_matches_only_the_basename() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::create_dir(root.path().join("src")).unwrap();
        fs::write(root.path().join("src/main.rs"), "needle\n").unwrap();

        // Act
        let basename_output = tool
            .execute(json!({
                "pattern": "needle",
                "path": "src/main.rs",
                "include": "*.rs"
            }))
            .await
            .unwrap();
        let relative_path_output = tool
            .execute(json!({
                "pattern": "needle",
                "path": "src/main.rs",
                "include": "src/*.rs"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(basename_output, "src/main.rs:\n  1: needle");
        assert_eq!(relative_path_output, "No matches found.");
    }

    #[tokio::test]
    async fn basename_include_with_no_selected_directory_files_suggests_recursion() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::create_dir(root.path().join("src")).unwrap();
        fs::write(root.path().join("src/main.rs"), "needle\n").unwrap();

        // Act
        let output = tool
            .execute(json!({ "pattern": "needle", "include": "*.rs" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, format!("{NO_MATCHES}\n{RECURSIVE_INCLUDE_HINT}"));
    }

    #[tokio::test]
    async fn basename_include_with_selected_files_and_no_content_match_omits_the_hint() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(root.path().join("main.rs"), "haystack\n").unwrap();

        // Act
        let output = tool
            .execute(json!({ "pattern": "needle", "include": "*.rs" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, NO_MATCHES);
    }

    #[tokio::test]
    async fn basename_include_with_an_explicit_file_target_omits_the_hint() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(root.path().join("main.rs"), "needle\n").unwrap();

        // Act
        let output = tool
            .execute(json!({
                "pattern": "needle",
                "path": "main.rs",
                "include": "*.txt"
            }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, NO_MATCHES);
    }

    #[tokio::test]
    async fn path_include_with_no_selected_directory_files_omits_the_hint() {
        // Arrange
        let (_root, tool) = grep_tool();

        // Act
        let output = tool
            .execute(json!({ "pattern": "needle", "include": "src/*.rs" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, NO_MATCHES);
    }

    #[tokio::test]
    async fn directory_search_applies_include_ignore_and_newest_first_order() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::create_dir(root.path().join("src")).unwrap();
        fs::write(root.path().join(".gitignore"), "ignored.rs\n").unwrap();
        fs::write(root.path().join("old.rs"), "needle\n").unwrap();
        fs::write(root.path().join("src/new.rs"), "needle\n").unwrap();
        fs::write(root.path().join("ignored.rs"), "needle\n").unwrap();
        fs::write(root.path().join("notes.txt"), "needle\n").unwrap();
        set_mtime(&root.path().join("old.rs"), 1);
        set_mtime(&root.path().join("src/new.rs"), 2);

        // Act
        let output = tool
            .execute(json!({ "pattern": "needle", "include": "**/*.rs" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, "src/new.rs:\n  1: needle\n\nold.rs:\n  1: needle");
    }

    #[tokio::test]
    async fn an_explicit_ignored_file_is_still_searched() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(root.path().join(".gitignore"), "ignored.rs\n").unwrap();
        fs::write(root.path().join("ignored.rs"), "needle\n").unwrap();

        // Act
        let output = tool
            .execute(json!({ "pattern": "needle", "path": "ignored.rs" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, "ignored.rs:\n  1: needle");
    }

    #[tokio::test]
    async fn a_missing_explicit_target_is_a_path_specific_failure() {
        // Arrange
        let (root, tool) = grep_tool();

        // Act
        let error = tool
            .execute(json!({ "pattern": "needle", "path": "missing.rs" }))
            .await
            .unwrap_err();

        // Assert
        let io_error = fs::metadata(root.path().join("missing.rs")).unwrap_err();
        assert_eq!(error, format!("failed to grep `missing.rs`: {io_error}"));
    }

    #[tokio::test]
    async fn an_explicit_binary_file_has_a_distinct_successful_result() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(root.path().join("binary.dat"), b"needle\n\0binary\n").unwrap();

        // Act
        let output = tool
            .execute(json!({ "pattern": "needle", "path": "binary.dat" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(output, "binary file skipped");
    }

    #[tokio::test]
    async fn malformed_utf8_content_is_lossy_with_one_result_warning() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(
            root.path().join("broken.txt"),
            b"needle \xFF\nneedle \xFE\n",
        )
        .unwrap();

        // Act
        let output = tool
            .execute(json!({ "pattern": "needle", "path": "broken.txt" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(
            output,
            "broken.txt:\n  1: needle �\n  2: needle �\n\n[warning: invalid UTF-8 in \
             returned source lines was replaced with U+FFFD]"
        );
    }

    #[tokio::test]
    async fn an_oversized_explicit_file_returns_a_path_specific_size_error() {
        // Arrange
        let (root, tool) = grep_tool();
        let path = root.path().join("large.txt");
        fs::File::create(&path)
            .unwrap()
            .set_len(MAX_FILE_SIZE_BYTES + 1)
            .unwrap();

        // Act
        let error = tool
            .execute(json!({ "pattern": "needle", "path": "large.txt" }))
            .await
            .unwrap_err();

        // Assert
        assert_eq!(
            error,
            format!(
                "failed to grep `large.txt`: file is {} bytes, which exceeds the \
                 {MAX_FILE_SIZE_BYTES} byte grep limit",
                MAX_FILE_SIZE_BYTES + 1
            )
        );
    }

    #[tokio::test]
    async fn recursive_search_skips_oversized_files_and_reports_one_warning() {
        // Arrange
        let (root, tool) = grep_tool();
        fs::write(root.path().join("small.txt"), "needle\n").unwrap();
        fs::File::create(root.path().join("large.txt"))
            .unwrap()
            .set_len(MAX_FILE_SIZE_BYTES + 1)
            .unwrap();

        // Act
        let output = tool.execute(json!({ "pattern": "needle" })).await.unwrap();

        // Assert
        assert_eq!(
            output,
            "small.txt:\n  1: needle\n\n[warning: skipped 1 oversized file]"
        );
    }

    #[test]
    fn grep_is_registered_in_the_tool_set() {
        // Arrange
        let root = tempdir().unwrap();
        let workspace = Workspace::new(root.path().to_path_buf()).unwrap();
        let tools = crate::tools::ToolSet::new(Arc::new(workspace), None);

        // Act
        let definition = tools.locate("grep").unwrap().definition();

        // Assert
        assert_eq!(definition.name, "grep");
    }
}
