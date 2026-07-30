#[cfg(unix)]
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[cfg(unix)]
const UNIX_ENDPOINT_SCHEME: &str = "unix://";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DockerExecutableName {
    Docker,
    DockerCompose,
}

impl DockerExecutableName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Docker => "docker",
            Self::DockerCompose => "docker-compose",
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DockerExecutable {
    name: DockerExecutableName,
    path: PathBuf,
}

impl DockerExecutable {
    pub fn validate(
        name: DockerExecutableName,
        path: impl AsRef<Path>,
    ) -> Result<Self, DockerExecutableError> {
        let path = path.as_ref();
        if !path.is_absolute() {
            return Err(DockerExecutableError::RelativePath {
                path: path.to_path_buf(),
            });
        }

        validate_platform_executable(name, path)
    }

    pub fn name(&self) -> DockerExecutableName {
        self.name
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DockerIntegration {
    endpoint: DockerEndpoint,
    executables: Vec<DockerExecutable>,
}

impl DockerIntegration {
    pub fn new(endpoint: DockerEndpoint) -> Self {
        Self {
            endpoint,
            executables: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_executables(
        mut self,
        executables: impl IntoIterator<Item = DockerExecutable>,
    ) -> Self {
        for executable in executables {
            if let Some(existing) = self
                .executables
                .iter_mut()
                .find(|existing| existing.name == executable.name)
            {
                *existing = executable;
            } else {
                self.executables.push(executable);
            }
        }
        self.executables
            .sort_unstable_by_key(|executable| executable.name);
        self
    }

    pub fn endpoint(&self) -> &DockerEndpoint {
        &self.endpoint
    }

    pub fn executables(&self) -> &[DockerExecutable] {
        &self.executables
    }

    pub fn supports(&self, name: DockerExecutableName) -> bool {
        self.executables.is_empty()
            || self
                .executables
                .iter()
                .any(|executable| executable.name == name)
    }
}

impl From<DockerEndpoint> for DockerIntegration {
    fn from(endpoint: DockerEndpoint) -> Self {
        Self::new(endpoint)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DockerEndpoint {
    path: PathBuf,
    resource: String,
}

impl DockerEndpoint {
    pub fn validate(path: impl AsRef<Path>) -> Result<Self, DockerEndpointError> {
        let path = path.as_ref();
        if !path.is_absolute() {
            return Err(DockerEndpointError::RelativePath {
                path: path.to_path_buf(),
            });
        }

        validate_platform_endpoint(path)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn resource(&self) -> &str {
        &self.resource
    }
}

#[cfg(unix)]
fn validate_platform_endpoint(path: &Path) -> Result<DockerEndpoint, DockerEndpointError> {
    use std::os::unix::fs::FileTypeExt;

    let canonical_path =
        fs::canonicalize(path).map_err(|error| DockerEndpointError::Unavailable {
            detail: error.to_string(),
            path: path.to_path_buf(),
        })?;
    let metadata =
        fs::metadata(&canonical_path).map_err(|error| DockerEndpointError::Unavailable {
            detail: error.to_string(),
            path: canonical_path.clone(),
        })?;

    if !metadata.file_type().is_socket() {
        return Err(DockerEndpointError::NotUnixSocket {
            path: canonical_path,
        });
    }

    endpoint_from_canonical_path(canonical_path)
}

#[cfg(not(unix))]
fn validate_platform_endpoint(_path: &Path) -> Result<DockerEndpoint, DockerEndpointError> {
    Err(DockerEndpointError::UnsupportedPlatform)
}

#[cfg(unix)]
fn endpoint_from_canonical_path(
    canonical_path: PathBuf,
) -> Result<DockerEndpoint, DockerEndpointError> {
    let resource = canonical_path
        .to_str()
        .ok_or_else(|| DockerEndpointError::NonUtf8Path {
            path: canonical_path.clone(),
        })
        .map(|path| format!("{UNIX_ENDPOINT_SCHEME}{path}"))?;

    Ok(DockerEndpoint {
        path: canonical_path,
        resource,
    })
}

#[cfg(unix)]
fn validate_platform_executable(
    name: DockerExecutableName,
    path: &Path,
) -> Result<DockerExecutable, DockerExecutableError> {
    use std::os::unix::fs::PermissionsExt;

    let canonical_path =
        fs::canonicalize(path).map_err(|error| DockerExecutableError::Unavailable {
            detail: error.to_string(),
            path: path.to_path_buf(),
        })?;
    let metadata =
        fs::metadata(&canonical_path).map_err(|error| DockerExecutableError::Unavailable {
            detail: error.to_string(),
            path: canonical_path.clone(),
        })?;
    if !metadata.is_file() {
        return Err(DockerExecutableError::NotRegularFile {
            path: canonical_path,
        });
    }
    if metadata.permissions().mode() & 0o111 == 0 {
        return Err(DockerExecutableError::NotExecutable {
            path: canonical_path,
        });
    }

    Ok(DockerExecutable {
        name,
        path: canonical_path,
    })
}

#[cfg(not(unix))]
fn validate_platform_executable(
    _name: DockerExecutableName,
    _path: &Path,
) -> Result<DockerExecutable, DockerExecutableError> {
    Err(DockerExecutableError::UnsupportedPlatform)
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum DockerEndpointError {
    #[error("Docker endpoint path `{path}` must be absolute")]
    RelativePath { path: PathBuf },

    #[error("Docker endpoint path `{path}` is unavailable: {detail}")]
    Unavailable { detail: String, path: PathBuf },

    #[error("Docker endpoint path `{path}` is not a Unix socket")]
    NotUnixSocket { path: PathBuf },

    #[error("Docker endpoint path `{path}` is not valid UTF-8")]
    NonUtf8Path { path: PathBuf },

    #[error("local Unix Docker endpoints are not supported on this platform")]
    UnsupportedPlatform,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum DockerExecutableError {
    #[error("Docker executable path `{path}` must be absolute")]
    RelativePath { path: PathBuf },

    #[error("Docker executable path `{path}` is unavailable: {detail}")]
    Unavailable { detail: String, path: PathBuf },

    #[error("Docker executable path `{path}` is not a regular file")]
    NotRegularFile { path: PathBuf },

    #[error("Docker executable path `{path}` is not executable")]
    NotExecutable { path: PathBuf },

    #[error("Docker executable discovery is not supported on this platform")]
    UnsupportedPlatform,
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::os::unix::net::UnixListener;
    use tempfile::TempDir;

    #[test]
    fn endpoint_canonicalizes_a_local_unix_socket_and_builds_its_resource_identity() {
        // Arrange
        let root = TempDir::new().unwrap();
        let socket_path = root.path().join("daemon.sock");
        let link_path = root.path().join("docker.sock");
        let _listener = UnixListener::bind(&socket_path).unwrap();
        symlink(&socket_path, &link_path).unwrap();

        // Act
        let endpoint = DockerEndpoint::validate(&link_path).unwrap();

        // Assert
        assert_eq!(
            endpoint.path(),
            fs::canonicalize(&socket_path).unwrap().as_path()
        );
        assert_eq!(
            endpoint.resource(),
            format!("unix://{}", endpoint.path().display())
        );
    }

    #[test]
    fn endpoint_rejects_relative_missing_and_non_socket_paths() {
        // Arrange
        let root = TempDir::new().unwrap();
        let missing_path = root.path().join("missing.sock");
        let regular_path = root.path().join("regular-file");
        fs::write(&regular_path, "not a socket").unwrap();

        // Act
        let relative = DockerEndpoint::validate("docker.sock");
        let missing = DockerEndpoint::validate(&missing_path);
        let regular = DockerEndpoint::validate(&regular_path);

        // Assert
        assert!(matches!(
            relative,
            Err(DockerEndpointError::RelativePath { .. })
        ));
        assert!(matches!(
            missing,
            Err(DockerEndpointError::Unavailable { .. })
        ));
        assert_eq!(
            regular,
            Err(DockerEndpointError::NotUnixSocket {
                path: fs::canonicalize(regular_path).unwrap(),
            })
        );
    }

    #[test]
    fn executable_validation_canonicalizes_symlinks_and_requires_an_executable_file() {
        // Arrange
        let root = TempDir::new().unwrap();
        let target = root.path().join("docker-target");
        let link = root.path().join("docker");
        let non_executable = root.path().join("not-executable");
        fs::write(&target, "#!/bin/true\n").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(&non_executable, "not executable").unwrap();
        symlink(&target, &link).unwrap();

        // Act
        let executable = DockerExecutable::validate(DockerExecutableName::Docker, &link).unwrap();
        let rejected =
            DockerExecutable::validate(DockerExecutableName::DockerCompose, &non_executable);

        // Assert
        assert_eq!(executable.name(), DockerExecutableName::Docker);
        assert_eq!(executable.path(), fs::canonicalize(target).unwrap());
        assert!(matches!(
            rejected,
            Err(DockerExecutableError::NotExecutable { .. })
        ));
    }
}
