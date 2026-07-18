use globset::{GlobBuilder, GlobMatcher};
use ignore::{DirEntry, WalkBuilder};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

const MAX_VISITED_ENTRIES: usize = 100_000;

#[derive(Debug)]
pub(super) struct FileSelector {
    matcher: Option<GlobMatcher>,
}

impl FileSelector {
    pub(super) fn all() -> Self {
        Self { matcher: None }
    }

    pub(super) fn glob(pattern: &str) -> Result<Self, globset::Error> {
        let matcher = GlobBuilder::new(pattern)
            .backslash_escape(true)
            .case_insensitive(false)
            .literal_separator(true)
            .build()?
            .compile_matcher();

        Ok(Self {
            matcher: Some(matcher),
        })
    }

    pub(super) fn matches(&self, target_relative_path: &Path) -> bool {
        self.matcher
            .as_ref()
            .is_none_or(|matcher| matcher.is_match(target_relative_path))
    }
}

#[derive(Debug)]
pub(super) struct LocatedMetadata {
    pub(super) size: u64,
    pub(super) modified: Option<SystemTime>,
}

#[derive(Debug)]
pub(super) struct LocatedFile {
    pub(super) path: PathBuf,
    pub(super) metadata: Option<LocatedMetadata>,
}

impl LocatedFile {
    pub(super) fn from_metadata(path: PathBuf, metadata: std::fs::Metadata) -> Self {
        Self {
            path,
            metadata: Some(LocatedMetadata {
                size: metadata.len(),
                modified: metadata.modified().ok(),
            }),
        }
    }
}

#[derive(Debug, Error)]
pub(super) enum FileDiscoveryError {
    #[error("background discovery task failed: {0}")]
    BlockingTask(#[from] tokio::task::JoinError),

    #[error("file discovery was cancelled")]
    Cancelled,

    #[error("failed to load ignore file `{path}`: {source}")]
    Ignore {
        path: PathBuf,
        source: ignore::Error,
    },

    #[error("invalid discovery scope: workspace `{workspace_root}`, search root `{search_root}`")]
    InvalidScope {
        workspace_root: PathBuf,
        search_root: PathBuf,
    },

    #[error("search root must not be inside `.git`: {0}")]
    RootInGit(PathBuf),

    #[error("failed to get metadata for search root `{path}`: {source}")]
    RootMetadata {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("search root is not a directory: {0}")]
    RootNotDirectory(PathBuf),

    #[error("file discovery exceeded the {limit}-entry limit; choose a narrower search path")]
    TooManyEntries { limit: usize },

    #[error("failed while traversing files: {0}")]
    Traversal(#[from] ignore::Error),

    #[error("walked path `{path}` was outside expected root `{root}`")]
    UnexpectedPath { path: PathBuf, root: PathBuf },
}

pub(super) async fn locate_files(
    cancel: CancellationToken,
    workspace_root: PathBuf,
    search_root: PathBuf,
    selector: FileSelector,
) -> Result<Vec<LocatedFile>, FileDiscoveryError> {
    if cancel.is_cancelled() {
        return Err(FileDiscoveryError::Cancelled);
    }

    tokio::task::spawn_blocking(move || {
        locate_files_sync(&cancel, &workspace_root, &search_root, &selector)
    })
    .await?
}

pub(super) fn locate_files_sync(
    cancel: &CancellationToken,
    workspace_root: &Path,
    search_root: &Path,
    selector: &FileSelector,
) -> Result<Vec<LocatedFile>, FileDiscoveryError> {
    locate_files_with_limit(
        cancel,
        workspace_root,
        search_root,
        selector,
        MAX_VISITED_ENTRIES,
    )
}

fn locate_files_with_limit(
    cancel: &CancellationToken,
    workspace_root: &Path,
    search_root: &Path,
    selector: &FileSelector,
    max_visited_entries: usize,
) -> Result<Vec<LocatedFile>, FileDiscoveryError> {
    if cancel.is_cancelled() {
        return Err(FileDiscoveryError::Cancelled);
    }

    if !workspace_root.is_absolute()
        || !search_root.is_absolute()
        || !search_root.starts_with(workspace_root)
    {
        return Err(FileDiscoveryError::InvalidScope {
            workspace_root: workspace_root.to_path_buf(),
            search_root: search_root.to_path_buf(),
        });
    }

    let relative_search_root =
        search_root
            .strip_prefix(workspace_root)
            .map_err(|_| FileDiscoveryError::InvalidScope {
                workspace_root: workspace_root.to_path_buf(),
                search_root: search_root.to_path_buf(),
            })?;

    if relative_search_root
        .components()
        .any(|component| component.as_os_str() == ".git")
    {
        return Err(FileDiscoveryError::RootInGit(search_root.to_path_buf()));
    }

    let metadata = std::fs::symlink_metadata(search_root).map_err(|source| {
        FileDiscoveryError::RootMetadata {
            path: search_root.to_path_buf(),
            source,
        }
    })?;

    if !metadata.is_dir() {
        return Err(FileDiscoveryError::RootNotDirectory(
            search_root.to_path_buf(),
        ));
    }

    let mut builder = WalkBuilder::new(search_root);
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

    attach_ancestor_gitignores(cancel, workspace_root, search_root, &mut builder)?;

    collect_files(
        cancel,
        workspace_root,
        search_root,
        selector,
        max_visited_entries,
        builder.build(),
    )
}

fn attach_ancestor_gitignores(
    cancel: &CancellationToken,
    workspace_root: &Path,
    search_root: &Path,
    builder: &mut WalkBuilder,
) -> Result<(), FileDiscoveryError> {
    if search_root.eq(workspace_root) {
        return Ok(());
    }

    let mut ancestors = search_root
        .ancestors()
        .skip(1) // exclude search_root itself
        .take_while(|ancestor| ancestor.starts_with(workspace_root))
        .collect::<Vec<_>>();

    ancestors.reverse();

    for ancestor in ancestors {
        if cancel.is_cancelled() {
            return Err(FileDiscoveryError::Cancelled);
        }

        let ignore_path = ancestor.join(".gitignore");

        match std::fs::symlink_metadata(&ignore_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                continue;
            }
            Err(source) => {
                return Err(FileDiscoveryError::Ignore {
                    path: ignore_path,
                    source: source.into(),
                });
            }
            Ok(_) => {}
        }

        builder.current_dir(ancestor);

        if let Some(source) = builder.add_ignore(&ignore_path) {
            return Err(FileDiscoveryError::Ignore {
                path: ignore_path,
                source,
            });
        }
    }

    if cancel.is_cancelled() {
        return Err(FileDiscoveryError::Cancelled);
    }

    Ok(())
}

fn collect_files(
    cancel: &CancellationToken,
    workspace_root: &Path,
    search_root: &Path,
    selector: &FileSelector,
    max_visited_entries: usize,
    mut entries: impl Iterator<Item = Result<DirEntry, ignore::Error>>,
) -> Result<Vec<LocatedFile>, FileDiscoveryError> {
    let mut files = Vec::new();
    let mut visited_entries = 0;

    loop {
        if cancel.is_cancelled() {
            return Err(FileDiscoveryError::Cancelled);
        }

        let next = entries.next();

        if cancel.is_cancelled() {
            return Err(FileDiscoveryError::Cancelled);
        }

        let Some(entry_result) = next else {
            break;
        };
        let entry = entry_result?;

        // Keep the depth-zero entry so WalkBuilder can load the root's .gitignore,
        // but do not charge or match it.
        if entry.depth() == 0 {
            continue;
        }

        visited_entries += 1;
        if visited_entries > max_visited_entries {
            return Err(FileDiscoveryError::TooManyEntries {
                limit: max_visited_entries,
            });
        }

        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }

        let target_relative_path = entry.path().strip_prefix(search_root).map_err(|_| {
            FileDiscoveryError::UnexpectedPath {
                path: entry.path().to_path_buf(),
                root: search_root.to_path_buf(),
            }
        })?;

        if !selector.matches(target_relative_path) {
            continue;
        }

        entry.path().strip_prefix(workspace_root).map_err(|_| {
            FileDiscoveryError::UnexpectedPath {
                path: entry.path().to_path_buf(),
                root: workspace_root.to_path_buf(),
            }
        })?;

        let metadata = entry.metadata().ok().map(|metadata| LocatedMetadata {
            size: metadata.len(),
            modified: metadata.modified().ok(),
        });

        files.push(LocatedFile {
            path: entry.path().to_path_buf(),
            metadata,
        });
    }

    if cancel.is_cancelled() {
        return Err(FileDiscoveryError::Cancelled);
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::{TempDir, tempdir};

    fn workspace() -> (TempDir, PathBuf) {
        let root = tempdir().unwrap();
        let canonical_root = dunce::canonicalize(root.path()).unwrap();
        (root, canonical_root)
    }

    fn locate_all(
        workspace_root: &Path,
        search_root: &Path,
    ) -> Result<Vec<LocatedFile>, FileDiscoveryError> {
        locate_files_with_limit(
            &CancellationToken::new(),
            workspace_root,
            search_root,
            &FileSelector::all(),
            usize::MAX,
        )
    }

    #[test]
    fn injected_visited_entry_limit_fails_without_partial_results() {
        // Arrange
        let (_root, workspace_root) = workspace();
        fs::write(workspace_root.join("one.rs"), "one").unwrap();
        fs::write(workspace_root.join("two.rs"), "two").unwrap();

        // Act
        let result = locate_files_with_limit(
            &CancellationToken::new(),
            &workspace_root,
            &workspace_root,
            &FileSelector::all(),
            1,
        );

        // Assert
        assert!(matches!(
            result,
            Err(FileDiscoveryError::TooManyEntries { limit: 1 })
        ));
    }

    #[test]
    fn nearer_ancestor_gitignore_overrides_an_outer_rule() {
        // Arrange
        let (_root, workspace_root) = workspace();
        let package_root = workspace_root.join("packages");
        let search_root = package_root.join("app");
        fs::create_dir_all(&search_root).unwrap();
        fs::write(workspace_root.join(".gitignore"), "restored.txt\n").unwrap();
        fs::write(package_root.join(".gitignore"), "!restored.txt\n").unwrap();
        fs::write(search_root.join("restored.txt"), "visible").unwrap();

        // Act
        let files = locate_all(&workspace_root, &search_root).unwrap();

        // Assert
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, search_root.join("restored.txt"));
    }

    #[test]
    fn malformed_ancestor_gitignore_fails_with_its_path() {
        // Arrange
        let (_root, workspace_root) = workspace();
        let search_root = workspace_root.join("nested");
        fs::create_dir(&search_root).unwrap();
        let ignore_path = workspace_root.join(".gitignore");
        fs::write(&ignore_path, "*.{rs,txt\n").unwrap();
        fs::write(search_root.join("visible.txt"), "visible").unwrap();

        // Act
        let result = locate_all(&workspace_root, &search_root);

        // Assert
        assert!(
            matches!(
                result,
                Err(FileDiscoveryError::Ignore { ref path, .. }) if path == &ignore_path
            ),
            "{result:?}"
        );
    }

    #[test]
    fn walker_errors_fail_instead_of_returning_partial_results() {
        // Arrange
        let (_root, workspace_root) = workspace();
        fs::write(workspace_root.join("a.txt"), "a").unwrap();
        fs::write(workspace_root.join("b.txt"), "b").unwrap();

        // Both files are available before the synthetic error, modeling a
        // failure after readable siblings have already been visited.
        let unreadable = ignore::Error::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "permission denied",
        ));
        let entries = WalkBuilder::new(&workspace_root)
            .build()
            .chain(std::iter::once(Err(unreadable)));

        // Act
        let result = collect_files(
            &CancellationToken::new(),
            &workspace_root,
            &workspace_root,
            &FileSelector::glob("*.txt").unwrap(),
            usize::MAX,
            entries,
        );

        // Assert
        assert!(
            matches!(result, Err(FileDiscoveryError::Traversal(_))),
            "{result:?}"
        );
    }

    #[test]
    fn cancellation_during_a_walk_stops_promptly() {
        // Arrange
        let (_root, workspace_root) = workspace();
        fs::write(workspace_root.join("a.txt"), "a").unwrap();
        fs::write(workspace_root.join("b.txt"), "b").unwrap();

        let mut walked = 0;
        let cancel = CancellationToken::new();
        let entries = WalkBuilder::new(&workspace_root).build().inspect(|_entry| {
            walked += 1;
            cancel.cancel();
        });

        // Act
        let result = collect_files(
            &cancel,
            &workspace_root,
            &workspace_root,
            &FileSelector::glob("*.txt").unwrap(),
            usize::MAX,
            entries,
        );

        // Assert
        assert_eq!(walked, 1);
        assert!(
            matches!(result, Err(FileDiscoveryError::Cancelled)),
            "{result:?}"
        );
    }

    #[test]
    fn nested_search_inherits_workspace_gitignore_rules() {
        // Arrange
        let (_root, workspace_root) = workspace();
        let search_root = workspace_root.join("nested");
        fs::create_dir(&search_root).unwrap();
        fs::write(workspace_root.join(".gitignore"), "parent-ignored.txt\n").unwrap();
        fs::write(search_root.join(".gitignore"), "local-ignored.txt\n").unwrap();
        fs::write(search_root.join("parent-ignored.txt"), "ignored").unwrap();
        fs::write(search_root.join("local-ignored.txt"), "ignored").unwrap();
        fs::write(search_root.join("visible.txt"), "visible").unwrap();

        // Act
        let files = locate_all(&workspace_root, &search_root).unwrap();
        let mut names = files
            .into_iter()
            .map(|file| file.path.file_name().unwrap().to_owned())
            .collect::<Vec<_>>();
        names.sort();

        // Assert
        assert_eq!(names, [".gitignore", "visible.txt"]);
    }

    #[test]
    fn discovery_rejects_a_search_root_inside_dot_git() {
        // Arrange
        let (_root, workspace_root) = workspace();
        let search_root = workspace_root.join(".git").join("objects");
        fs::create_dir_all(&search_root).unwrap();
        fs::write(search_root.join("object"), "contents").unwrap();

        // Act
        let result = locate_all(&workspace_root, &search_root);

        // Assert
        assert!(
            matches!(result, Err(FileDiscoveryError::RootInGit(ref path)) if path == &search_root),
            "{result:?}"
        );
    }
}
