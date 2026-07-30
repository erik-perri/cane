use std::path::{Path, PathBuf};

use crate::WorkspaceCapabilityConsentStore;

#[derive(Clone, Debug)]
pub struct SessionConfig {
    cane_version: String,
    instructions: String,
    sessions_directory: PathBuf,
    workspace_consents: Option<WorkspaceCapabilityConsentStore>,
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
            workspace_consents: None,
        }
    }

    #[must_use]
    pub fn with_workspace_capability_consents(
        mut self,
        store: WorkspaceCapabilityConsentStore,
    ) -> Self {
        self.workspace_consents = Some(store);
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

    pub(crate) fn workspace_consents(&self) -> Option<&WorkspaceCapabilityConsentStore> {
        self.workspace_consents.as_ref()
    }
}
