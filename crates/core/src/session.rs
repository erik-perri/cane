use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionConfig {
    cane_version: String,
    instructions: String,
    sessions_directory: PathBuf,
}

impl SessionConfig {
    pub fn new(
        cane_version: impl Into<String>,
        instructions: impl Into<String>,
        sessions_directory: impl Into<PathBuf>,
    ) -> Self {
        Self {
            cane_version: cane_version.into(),
            instructions: instructions.into(),
            sessions_directory: sessions_directory.into(),
        }
    }

    pub fn cane_version(&self) -> &str {
        &self.cane_version
    }

    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    pub fn sessions_directory(&self) -> &Path {
        &self.sessions_directory
    }
}
