#[cfg(unix)]
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[cfg(unix)]
const UNIX_ENDPOINT_SCHEME: &str = "unix://";

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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
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
}
