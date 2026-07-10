use std::io;
use std::path::{Component, Path, PathBuf};

pub struct Workspace {
    canonical_root: PathBuf,
}

impl Workspace {
    pub fn new(root: PathBuf) -> io::Result<Self> {
        let canonical_root = dunce::canonicalize(root)?;

        Ok(Self { canonical_root })
    }

    /// Resolve a tool-supplied path (absolute or relative-to-root) and
    /// error if it escapes the root.
    pub fn resolve(&self, candidate: &str) -> Result<PathBuf, String> {
        if candidate.is_empty() {
            return Err("path must not be empty".to_string());
        }

        let candidate = Path::new(candidate);
        let absolute_candidate = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.canonical_root.join(candidate)
        };
        let normalized_candidate = lexical_normalize(&absolute_candidate);
        let canonical_user = canonicalize_with_missing_tail(&normalized_candidate)
            .map_err(|err| format!("failed to canonicalize path: {err}"))?;

        if canonical_user.starts_with(&self.canonical_root) {
            Ok(canonical_user)
        } else {
            Err(format!(
                "you do not have access to path {}",
                candidate.display()
            ))
        }
    }
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }

    normalized
}

fn canonicalize_with_missing_tail(path: &Path) -> io::Result<PathBuf> {
    let mut existing_ancestor = path.to_path_buf();
    let mut missing_components = Vec::new();

    let canonical_ancestor = loop {
        match dunce::canonicalize(&existing_ancestor) {
            Ok(canonical) => {
                if !missing_components.is_empty() && !canonical.is_dir() {
                    return Err(io::Error::new(
                        io::ErrorKind::NotADirectory,
                        format!("{} is not a directory", existing_ancestor.display()),
                    ));
                }

                break canonical;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if std::fs::symlink_metadata(&existing_ancestor)
                    .is_ok_and(|metadata| metadata.file_type().is_symlink())
                {
                    return Err(error);
                }

                let component = existing_ancestor.file_name().ok_or(error)?.to_os_string();
                missing_components.push(component);
                existing_ancestor.pop();
            }
            Err(error) => return Err(error),
        }
    };

    Ok(missing_components
        .into_iter()
        .rev()
        .fold(canonical_ancestor, |path, component| path.join(component)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    fn workspace() -> (TempDir, Workspace) {
        let root = TempDir::new().unwrap();
        let workspace = Workspace::new(root.path().to_path_buf()).unwrap();
        (root, workspace)
    }

    fn path_str(path: &Path) -> &str {
        path.to_str().unwrap()
    }

    #[test]
    fn new_canonicalizes_the_workspace_root() {
        // Arrange
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("root");
        fs::create_dir(&root).unwrap();
        let non_canonical_root = root.join(".");

        // Act
        let workspace = Workspace::new(non_canonical_root).unwrap();

        // Assert
        assert_eq!(workspace.canonical_root, dunce::canonicalize(root).unwrap());
    }

    #[test]
    fn new_returns_an_io_error_when_the_workspace_root_does_not_exist() {
        // Arrange
        let parent = TempDir::new().unwrap();
        let missing_root = parent.path().join("missing");

        // Act
        let error = Workspace::new(missing_root).err().unwrap();

        // Assert
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn resolve_accepts_a_relative_path_inside_the_workspace() {
        // Arrange
        let (root, workspace) = workspace();
        let file = root.path().join("src").join("lib.rs");
        fs::create_dir(root.path().join("src")).unwrap();
        fs::write(&file, "mock source").unwrap();

        // Act
        let resolved = workspace.resolve("src/lib.rs");

        // Assert
        assert_eq!(resolved, Ok(dunce::canonicalize(file).unwrap()));
    }

    #[test]
    fn resolve_accepts_an_absolute_path_inside_the_workspace() {
        // Arrange
        let (root, workspace) = workspace();
        let file = root.path().join("Cargo.toml");
        fs::write(&file, "[package]").unwrap();

        // Act
        let resolved = workspace.resolve(path_str(&file));

        // Assert
        assert_eq!(resolved, Ok(dunce::canonicalize(file).unwrap()));
    }

    #[test]
    fn resolve_accepts_the_workspace_root_itself() {
        // Arrange
        let (root, workspace) = workspace();

        // Act
        let relative = workspace.resolve(".");
        let absolute = workspace.resolve(path_str(root.path()));

        // Assert
        let canonical_root = dunce::canonicalize(root.path()).unwrap();
        assert_eq!(relative, Ok(canonical_root.clone()));
        assert_eq!(absolute, Ok(canonical_root));
    }

    #[test]
    fn resolve_normalizes_parent_components_that_remain_inside_the_workspace() {
        // Arrange
        let (root, workspace) = workspace();
        fs::create_dir(root.path().join("src")).unwrap();
        let file = root.path().join("lib.rs");
        fs::write(&file, "mock source").unwrap();

        // Act
        let resolved = workspace.resolve("src/../lib.rs");

        // Assert
        assert_eq!(resolved, Ok(dunce::canonicalize(file).unwrap()));
    }

    #[test]
    fn resolve_rejects_a_relative_parent_escape() {
        // Arrange
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("root");
        fs::create_dir(&root).unwrap();
        let outside = parent.path().join("outside.txt");
        fs::write(&outside, "secret").unwrap();
        let workspace = Workspace::new(root).unwrap();

        // Act
        let error = workspace.resolve("../outside.txt").unwrap_err();

        // Assert
        assert_eq!(error, "you do not have access to path ../outside.txt");
    }

    #[test]
    fn resolve_rejects_an_absolute_path_outside_the_workspace() {
        // Arrange
        let workspace_root = TempDir::new().unwrap();
        let outside_root = TempDir::new().unwrap();
        let outside = outside_root.path().join("outside.txt");
        fs::write(&outside, "secret").unwrap();
        let workspace = Workspace::new(workspace_root.path().to_path_buf()).unwrap();

        // Act
        let error = workspace.resolve(path_str(&outside)).unwrap_err();

        // Assert
        assert_eq!(
            error,
            format!("you do not have access to path {}", outside.display())
        );
    }

    #[test]
    fn resolve_does_not_confuse_a_sibling_with_a_shared_string_prefix() {
        // Arrange
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("project");
        let sibling = parent.path().join("project-secrets");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&sibling).unwrap();
        let outside = sibling.join("secret.txt");
        fs::write(&outside, "secret").unwrap();
        let workspace = Workspace::new(root).unwrap();

        // Act
        let resolved = workspace.resolve(path_str(&outside));

        // Assert
        assert!(resolved.is_err());
    }

    #[test]
    fn resolve_accepts_a_file_that_does_not_exist_in_an_existing_subdirectory() {
        // Arrange
        let (root, workspace) = workspace();
        let source_dir = root.path().join("src");
        fs::create_dir(&source_dir).unwrap();

        // Act
        let resolved = workspace.resolve("src/new.rs");

        // Assert
        assert_eq!(
            resolved,
            Ok(dunce::canonicalize(source_dir).unwrap().join("new.rs"))
        );
    }

    #[test]
    fn resolve_accepts_a_path_with_multiple_nonexistent_components() {
        // Arrange
        let (root, workspace) = workspace();

        // Act
        let resolved = workspace.resolve("generated/nested/file.rs");

        // Assert
        assert_eq!(
            resolved,
            Ok(dunce::canonicalize(root.path())
                .unwrap()
                .join("generated/nested/file.rs"))
        );
    }

    #[test]
    fn resolve_rejects_a_nonexistent_path_beneath_an_outside_directory() {
        // Arrange
        let workspace_root = TempDir::new().unwrap();
        let outside_root = TempDir::new().unwrap();
        let candidate = outside_root.path().join("new.txt");
        let workspace = Workspace::new(workspace_root.path().to_path_buf()).unwrap();

        // Act
        let resolved = workspace.resolve(path_str(&candidate));

        // Assert
        assert!(resolved.is_err());
    }

    #[test]
    fn resolve_rejects_an_empty_path() {
        // Arrange
        let (_root, workspace) = workspace();

        // Act
        let error = workspace.resolve("").unwrap_err();

        // Assert
        assert_eq!(error, "path must not be empty");
    }

    #[test]
    fn resolve_rejects_a_parent_escape_through_a_nonexistent_directory() {
        // Arrange
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("root");
        fs::create_dir(&root).unwrap();
        let outside = parent.path().join("outside.txt");
        fs::write(&outside, "secret").unwrap();
        let workspace = Workspace::new(root).unwrap();

        // Act
        let resolved = workspace.resolve("missing/../../outside.txt");

        // Assert
        assert!(resolved.is_err());
    }

    #[test]
    fn resolve_rejects_an_absolute_path_with_a_parent_escape() {
        // Arrange
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("root");
        fs::create_dir(&root).unwrap();
        let outside = parent.path().join("outside.txt");
        fs::write(&outside, "secret").unwrap();
        let workspace = Workspace::new(root.clone()).unwrap();
        let candidate = root.join("..").join("outside.txt");

        // Act
        let resolved = workspace.resolve(path_str(&candidate));

        // Assert
        assert!(resolved.is_err());
    }

    #[test]
    fn resolve_errors_when_a_middle_component_is_a_file() {
        // Arrange
        let (root, workspace) = workspace();
        fs::write(root.path().join("Cargo.toml"), "[package]").unwrap();

        // Act
        let error = workspace.resolve("Cargo.toml/nested.txt").unwrap_err();

        // Assert
        assert!(error.starts_with("failed to canonicalize path:"));
    }

    #[cfg(unix)]
    #[test]
    fn resolve_accepts_a_symlink_that_resolves_inside_the_workspace() {
        use std::os::unix::fs::symlink;

        // Arrange
        let (root, workspace) = workspace();
        let target = root.path().join("target.txt");
        let link = root.path().join("link.txt");
        fs::write(&target, "mock source").unwrap();
        symlink(&target, &link).unwrap();

        // Act
        let resolved = workspace.resolve("link.txt");

        // Assert
        assert_eq!(resolved, Ok(dunce::canonicalize(target).unwrap()));
    }

    #[cfg(unix)]
    #[test]
    fn resolve_accepts_a_nonexistent_path_beneath_an_inside_symlink() {
        use std::os::unix::fs::symlink;

        // Arrange
        let (root, workspace) = workspace();
        let source_dir = root.path().join("src");
        fs::create_dir(&source_dir).unwrap();
        symlink(&source_dir, root.path().join("srclink")).unwrap();

        // Act
        let resolved = workspace.resolve("srclink/new.rs");

        // Assert
        assert_eq!(
            resolved,
            Ok(dunce::canonicalize(source_dir).unwrap().join("new.rs"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_rejects_a_symlink_that_resolves_outside_the_workspace() {
        use std::os::unix::fs::symlink;

        // Arrange
        let parent = TempDir::new().unwrap();
        let root = parent.path().join("root");
        fs::create_dir(&root).unwrap();
        let outside = parent.path().join("outside.txt");
        fs::write(&outside, "secret").unwrap();
        symlink(&outside, root.join("link.txt")).unwrap();
        let workspace = Workspace::new(root).unwrap();

        // Act
        let resolved = workspace.resolve("link.txt");

        // Assert
        assert!(resolved.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn resolve_rejects_a_nonexistent_path_beneath_an_outside_symlink() {
        use std::os::unix::fs::symlink;

        // Arrange
        let workspace_root = TempDir::new().unwrap();
        let outside_root = TempDir::new().unwrap();
        symlink(
            outside_root.path(),
            workspace_root.path().join("outside-link"),
        )
        .unwrap();
        let workspace = Workspace::new(workspace_root.path().to_path_buf()).unwrap();

        // Act
        let resolved = workspace.resolve("outside-link/new.txt");

        // Assert
        assert!(resolved.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn resolve_rejects_a_symlink_to_a_nonexistent_outside_path() {
        use std::os::unix::fs::symlink;

        // Arrange
        let workspace_root = TempDir::new().unwrap();
        let outside_root = TempDir::new().unwrap();
        let outside_target = outside_root.path().join("new.txt");
        symlink(
            &outside_target,
            workspace_root.path().join("outside-link.txt"),
        )
        .unwrap();
        let workspace = Workspace::new(workspace_root.path().to_path_buf()).unwrap();

        // Act
        let resolved = workspace.resolve("outside-link.txt");

        // Assert
        assert!(resolved.is_err());
    }

    #[cfg(windows)]
    #[test]
    fn resolve_accepts_mixed_separators_in_a_relative_path() {
        // Arrange
        let (root, workspace) = workspace();
        let source_dir = root.path().join("crates/core/src");
        fs::create_dir_all(&source_dir).unwrap();
        let file = source_dir.join("lib.rs");
        fs::write(&file, "mock source").unwrap();

        // Act
        let resolved = workspace.resolve(r"crates\core/src/lib.rs");

        // Assert
        assert_eq!(resolved, Ok(dunce::canonicalize(file).unwrap()));
    }
}
