use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SandboxFilesystemAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SandboxPathPurpose {
    GitMetadata,
    PassThrough,
    Runtime,
    Toolchain,
    Workspace,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SandboxPathGrant {
    pub access: SandboxFilesystemAccess,
    pub path: PathBuf,
    pub purpose: SandboxPathPurpose,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandSandboxPolicy {
    executable_path: Vec<PathBuf>,
    excluded_path_entries: Vec<PathBuf>,
    grants: Vec<SandboxPathGrant>,
    private_home: PathBuf,
    private_temp: PathBuf,
    workspace: PathBuf,
}

impl CommandSandboxPolicy {
    pub fn executable_path(&self) -> &[PathBuf] {
        &self.executable_path
    }

    pub fn excluded_path_entries(&self) -> &[PathBuf] {
        &self.excluded_path_entries
    }

    pub fn grants(&self) -> &[SandboxPathGrant] {
        &self.grants
    }

    pub fn private_home(&self) -> &Path {
        &self.private_home
    }

    pub fn private_temp(&self) -> &Path {
        &self.private_temp
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandSandboxPolicyConfig {
    pub git_metadata_roots: Vec<PathBuf>,
    pub inherited_path: Vec<PathBuf>,
    pub pass_through_roots: Vec<PathBuf>,
    pub private_home: PathBuf,
    pub private_temp: PathBuf,
    pub runtime_roots: Vec<PathBuf>,
    pub toolchain_roots: Vec<PathBuf>,
    pub workspace: PathBuf,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CommandSandboxPolicyError {
    #[error("{name} path `{path}` must be absolute")]
    RelativePath { name: &'static str, path: PathBuf },

    #[error("{name} path `{path}` must not expose the host root")]
    RootExposure { name: &'static str, path: PathBuf },

    #[error("{name} path `{path}` is not available: {detail}")]
    UnavailablePath {
        detail: String,
        name: &'static str,
        path: PathBuf,
    },

    #[error("private home and temporary directories must be distinct")]
    SharedPrivateDirectory,

    #[error(
        "{purpose:?} grant `{grant}` would hide private sandbox directory `{private_directory}`"
    )]
    PrivateDirectoryShadowed {
        grant: PathBuf,
        private_directory: PathBuf,
        purpose: SandboxPathPurpose,
    },
}

pub fn build_command_sandbox_policy(
    config: CommandSandboxPolicyConfig,
) -> Result<CommandSandboxPolicy, CommandSandboxPolicyError> {
    validate_directory("workspace", &config.workspace)?;
    validate_private_path("private home", &config.private_home)?;
    validate_private_path("private temporary directory", &config.private_temp)?;

    if config.private_home == config.private_temp {
        return Err(CommandSandboxPolicyError::SharedPrivateDirectory);
    }

    let mut grants = Vec::new();
    let mut granted_paths = HashSet::new();

    add_grants(
        &mut grants,
        &mut granted_paths,
        config.runtime_roots,
        SandboxFilesystemAccess::ReadOnly,
        SandboxPathPurpose::Runtime,
        "runtime",
        true,
    )?;
    add_grants(
        &mut grants,
        &mut granted_paths,
        config.toolchain_roots,
        SandboxFilesystemAccess::ReadOnly,
        SandboxPathPurpose::Toolchain,
        "toolchain",
        false,
    )?;
    add_grants(
        &mut grants,
        &mut granted_paths,
        config.pass_through_roots,
        SandboxFilesystemAccess::ReadOnly,
        SandboxPathPurpose::PassThrough,
        "pass-through",
        false,
    )?;

    grants.push(SandboxPathGrant {
        access: SandboxFilesystemAccess::ReadWrite,
        path: config.workspace.clone(),
        purpose: SandboxPathPurpose::Workspace,
    });

    add_grants(
        &mut grants,
        &mut granted_paths,
        config.git_metadata_roots,
        SandboxFilesystemAccess::ReadOnly,
        SandboxPathPurpose::GitMetadata,
        "Git metadata",
        false,
    )?;

    for grant in &grants {
        for private_directory in [&config.private_home, &config.private_temp] {
            if private_directory.starts_with(&grant.path) {
                return Err(CommandSandboxPolicyError::PrivateDirectoryShadowed {
                    grant: grant.path.clone(),
                    private_directory: private_directory.clone(),
                    purpose: grant.purpose,
                });
            }
        }
    }

    let (executable_path, excluded_path_entries) =
        filter_executable_path(config.inherited_path, &grants);

    if executable_path.is_empty() {
        return Err(CommandSandboxPolicyError::UnavailablePath {
            detail: "no inherited PATH entry is beneath a declared filesystem grant".to_string(),
            name: "executable PATH",
            path: PathBuf::from("<empty>"),
        });
    }

    Ok(CommandSandboxPolicy {
        executable_path,
        excluded_path_entries,
        grants,
        private_home: config.private_home,
        private_temp: config.private_temp,
        workspace: config.workspace,
    })
}

fn add_grants(
    grants: &mut Vec<SandboxPathGrant>,
    granted_paths: &mut HashSet<PathBuf>,
    paths: Vec<PathBuf>,
    access: SandboxFilesystemAccess,
    purpose: SandboxPathPurpose,
    name: &'static str,
    allow_symlink: bool,
) -> Result<(), CommandSandboxPolicyError> {
    for path in paths {
        validate_available_path(name, &path)?;
        if !allow_symlink
            && fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err(CommandSandboxPolicyError::UnavailablePath {
                detail: "symbolic-link grants are not permitted".to_string(),
                name,
                path,
            });
        }

        if granted_paths.insert(path.clone()) {
            grants.push(SandboxPathGrant {
                access,
                path,
                purpose,
            });
        }
    }

    Ok(())
}

fn filter_executable_path(
    inherited_path: Vec<PathBuf>,
    grants: &[SandboxPathGrant],
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut included = Vec::new();
    let mut excluded = Vec::new();
    let mut seen = HashSet::new();

    for entry in inherited_path {
        let visible = entry.is_absolute()
            && entry.is_dir()
            && fs::canonicalize(&entry).is_ok_and(|canonical_entry| {
                grants.iter().any(|grant| {
                    fs::canonicalize(&grant.path)
                        .is_ok_and(|canonical_grant| canonical_entry.starts_with(canonical_grant))
                })
            });

        if visible && seen.insert(entry.clone()) {
            included.push(entry);
        } else {
            excluded.push(entry);
        }
    }

    (included, excluded)
}

fn validate_available_path(
    name: &'static str,
    path: &Path,
) -> Result<(), CommandSandboxPolicyError> {
    validate_absolute_non_root(name, path)?;
    fs::metadata(path).map_err(|error| CommandSandboxPolicyError::UnavailablePath {
        detail: error.to_string(),
        name,
        path: path.to_path_buf(),
    })?;
    Ok(())
}

fn validate_directory(name: &'static str, path: &Path) -> Result<(), CommandSandboxPolicyError> {
    validate_available_path(name, path)?;

    if !path.is_dir() {
        return Err(CommandSandboxPolicyError::UnavailablePath {
            detail: "expected a directory".to_string(),
            name,
            path: path.to_path_buf(),
        });
    }

    Ok(())
}

fn validate_private_path(name: &'static str, path: &Path) -> Result<(), CommandSandboxPolicyError> {
    validate_absolute_non_root(name, path)
}

fn validate_absolute_non_root(
    name: &'static str,
    path: &Path,
) -> Result<(), CommandSandboxPolicyError> {
    if !path.is_absolute() {
        return Err(CommandSandboxPolicyError::RelativePath {
            name,
            path: path.to_path_buf(),
        });
    }

    if path.parent().is_none() {
        return Err(CommandSandboxPolicyError::RootExposure {
            name,
            path: path.to_path_buf(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    struct Fixture {
        _root: TempDir,
        config: CommandSandboxPolicyConfig,
        excluded_bin: PathBuf,
        runtime_bin: PathBuf,
        workspace: PathBuf,
    }

    fn fixture() -> Fixture {
        let root = TempDir::new().unwrap();
        let workspace = root.path().join("workspace");
        let runtime = root.path().join("runtime");
        let runtime_bin = runtime.join("bin");
        let excluded_bin = root.path().join("ambient").join("bin");
        let git_metadata = workspace.join(".git");
        fs::create_dir_all(&runtime_bin).unwrap();
        fs::create_dir_all(&excluded_bin).unwrap();
        fs::create_dir_all(&git_metadata).unwrap();

        let config = CommandSandboxPolicyConfig {
            git_metadata_roots: vec![git_metadata],
            inherited_path: vec![excluded_bin.clone(), runtime_bin.clone()],
            pass_through_roots: Vec::new(),
            private_home: root.path().join("private-home"),
            private_temp: root.path().join("private-temp"),
            runtime_roots: vec![runtime],
            toolchain_roots: Vec::new(),
            workspace: workspace.clone(),
        };

        Fixture {
            _root: root,
            config,
            excluded_bin,
            runtime_bin,
            workspace,
        }
    }

    #[test]
    fn policy_keeps_only_path_entries_beneath_declared_roots() {
        // Arrange
        let fixture = fixture();

        // Act
        let policy = build_command_sandbox_policy(fixture.config).unwrap();

        // Assert
        assert_eq!(policy.executable_path(), [fixture.runtime_bin]);
        assert_eq!(policy.excluded_path_entries(), [fixture.excluded_bin]);
    }

    #[test]
    fn policy_orders_workspace_before_read_only_git_overlays() {
        // Arrange
        let fixture = fixture();
        let git_metadata = fixture.workspace.join(".git");

        // Act
        let policy = build_command_sandbox_policy(fixture.config).unwrap();
        let workspace_index = policy
            .grants()
            .iter()
            .position(|grant| grant.purpose == SandboxPathPurpose::Workspace)
            .unwrap();
        let git_index = policy
            .grants()
            .iter()
            .position(|grant| grant.purpose == SandboxPathPurpose::GitMetadata)
            .unwrap();

        // Assert
        assert!(workspace_index < git_index);
        assert_eq!(
            policy.grants()[git_index],
            SandboxPathGrant {
                access: SandboxFilesystemAccess::ReadOnly,
                path: git_metadata,
                purpose: SandboxPathPurpose::GitMetadata,
            }
        );
    }

    #[test]
    fn policy_rejects_missing_roots_and_host_root_exposure() {
        // Arrange
        let fixture = fixture();
        let mut missing = fixture.config.clone();
        missing.runtime_roots = vec![fixture.workspace.join("missing")];
        let mut host_root = fixture.config;
        let filesystem_root = fixture
            .workspace
            .ancestors()
            .last()
            .expect("an absolute workspace has a filesystem root")
            .to_path_buf();
        host_root.pass_through_roots = vec![filesystem_root.clone()];

        // Act
        let missing_error = build_command_sandbox_policy(missing).unwrap_err();
        let root_error = build_command_sandbox_policy(host_root).unwrap_err();

        // Assert
        assert!(matches!(
            missing_error,
            CommandSandboxPolicyError::UnavailablePath {
                name: "runtime",
                ..
            }
        ));
        assert_eq!(
            root_error,
            CommandSandboxPolicyError::RootExposure {
                name: "pass-through",
                path: filesystem_root,
            }
        );
    }

    #[test]
    fn policy_fails_closed_when_no_visible_path_entry_remains() {
        // Arrange
        let fixture = fixture();
        let mut config = fixture.config;
        config.inherited_path = vec![fixture.excluded_bin];

        // Act
        let error = build_command_sandbox_policy(config).unwrap_err();

        // Assert
        assert!(matches!(
            error,
            CommandSandboxPolicyError::UnavailablePath {
                name: "executable PATH",
                ..
            }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn policy_rejects_a_pass_through_symlink_into_an_undeclared_tree() {
        use std::os::unix::fs::symlink;

        // Arrange
        let fixture = fixture();
        let outside = fixture.workspace.parent().unwrap().join("outside");
        let link = fixture.workspace.parent().unwrap().join("linked-tools");
        fs::create_dir_all(&outside).unwrap();
        symlink(outside, &link).unwrap();
        let mut config = fixture.config;
        config.pass_through_roots = vec![link];

        // Act
        let error = build_command_sandbox_policy(config).unwrap_err();

        // Assert
        assert!(matches!(
            error,
            CommandSandboxPolicyError::UnavailablePath {
                detail,
                name: "pass-through",
                ..
            } if detail == "symbolic-link grants are not permitted"
        ));
    }

    #[test]
    fn policy_rejects_a_grant_that_would_replace_a_private_directory() {
        // Arrange
        let fixture = fixture();
        let broad_grant = fixture.workspace.parent().unwrap().to_path_buf();
        let mut config = fixture.config;
        config.private_home = broad_grant.join("private-home");
        config.pass_through_roots = vec![broad_grant.clone()];

        // Act
        let error = build_command_sandbox_policy(config).unwrap_err();

        // Assert
        assert_eq!(
            error,
            CommandSandboxPolicyError::PrivateDirectoryShadowed {
                grant: broad_grant.clone(),
                private_directory: broad_grant.join("private-home"),
                purpose: SandboxPathPurpose::PassThrough,
            }
        );
    }
}
