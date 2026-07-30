use std::path::{Path, PathBuf};

use crate::WorkspaceCapabilityGrantStore;

#[derive(Clone, Debug)]
pub struct SessionConfig {
    cane_version: String,
    instructions: String,
    sessions_directory: PathBuf,
    workspace_grants: Option<WorkspaceCapabilityGrantStore>,
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
            workspace_grants: None,
        }
    }

    #[must_use]
    pub fn with_workspace_capability_grants(
        mut self,
        store: WorkspaceCapabilityGrantStore,
    ) -> Self {
        self.workspace_grants = Some(store);
        self
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

    pub(crate) fn workspace_grants(&self) -> Option<&WorkspaceCapabilityGrantStore> {
        self.workspace_grants.as_ref()
    }
}
