use crate::tools::file_discovery::LocatedFile;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug)]
pub(super) struct RenderedPath {
    pub(super) text: String,
    pub(super) lossy: bool,
    pub(super) json_escaped: bool,
}

#[derive(Debug, Error)]
pub(super) enum PathDisplayError {
    #[error("path `{path}` was outside workspace root `{workspace_root}`")]
    OutsideWorkspace {
        workspace_root: PathBuf,
        path: PathBuf,
    },
}

pub(super) fn render_workspace_path(
    workspace_root: &Path,
    path: &Path,
) -> Result<RenderedPath, PathDisplayError> {
    let relative =
        path.strip_prefix(workspace_root)
            .map_err(|_| PathDisplayError::OutsideWorkspace {
                workspace_root: workspace_root.to_path_buf(),
                path: path.to_path_buf(),
            })?;

    let mut lossy = false;
    let components = relative.components().map(|component| {
        let text = component.as_os_str().to_string_lossy();
        lossy |= matches!(text, std::borrow::Cow::Owned(_));
        text.into_owned()
    });
    let plain_text = components.collect::<Vec<_>>().join("/");
    let json_escaped = plain_text.chars().any(char::is_control);
    let text = if json_escaped {
        serde_json::to_string(&plain_text).expect("serializing a string cannot fail")
    } else {
        plain_text
    };

    Ok(RenderedPath {
        text,
        lossy,
        json_escaped,
    })
}

pub(super) fn compare_located_files(
    left_file: &LocatedFile,
    left_path: &RenderedPath,
    right_file: &LocatedFile,
    right_path: &RenderedPath,
) -> Ordering {
    let left_modified = left_file
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.modified);
    let right_modified = right_file
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.modified);

    right_modified
        .cmp(&left_modified)
        .then_with(|| left_path.text.cmp(&right_path.text))
        .then_with(|| left_file.path.cmp(&right_file.path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn renders_workspace_relative_paths_with_portable_separators() {
        // Arrange
        let root = tempdir().unwrap();
        let path = root.path().join("src").join("lib.rs");

        // Act
        let rendered = render_workspace_path(root.path(), &path).unwrap();

        // Assert
        assert_eq!(rendered.text, "src/lib.rs");
        assert!(!rendered.lossy);
        assert!(!rendered.json_escaped);
    }

    #[cfg(unix)]
    #[test]
    fn json_quotes_paths_that_would_break_one_path_per_line_output() {
        // Arrange
        let root = tempdir().unwrap();
        let path = root.path().join("line\nbreak.txt");

        // Act
        let rendered = render_workspace_path(root.path(), &path).unwrap();

        // Assert
        assert_eq!(rendered.text, "\"line\\nbreak.txt\"");
        assert!(!rendered.lossy);
        assert!(rendered.json_escaped);
    }
}
