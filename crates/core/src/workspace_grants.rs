use std::borrow::Cow;
use std::collections::HashSet;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::time::Duration;

use cane_config::{
    ArchiveError, ConfigFile, ConfigFileError, DocumentCodec, LoadOutcome, PersistenceError,
    RefreshOutcome, UpdateError,
};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const WORKSPACE_CAPABILITY_GRANTS_DOCUMENT: &str = "workspace-capability-grants.json";
pub const WORKSPACE_CAPABILITY_GRANTS_SCHEMA: &str = "workspace-capability-grants.schema.json";
pub const MAX_WORKSPACE_CAPABILITY_GRANTS: usize = 256;

const SCHEMA_VERSION: u64 = 0;
const SCHEMA_REFERENCE: &str = "./workspace-capability-grants.schema.json";

/// One capability authorization persisted for one canonical Workspace.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceCapabilityGrant {
    /// The canonical Workspace path serialized for the current platform.
    workspace: String,
    capability: PersistedCapability,
}

impl WorkspaceCapabilityGrant {
    /// Creates a grant from live domain objects whose identities have already been canonicalized.
    pub fn docker_daemon(
        workspace: &crate::Workspace,
        endpoint: &crate::command::DockerEndpoint,
    ) -> Result<Self, WorkspaceGrantDocumentError> {
        let workspace = workspace.root().to_str().ok_or_else(|| {
            WorkspaceGrantDocumentError::NonUtf8Workspace {
                path: workspace.root().to_path_buf(),
            }
        })?;
        Ok(Self {
            workspace: workspace.to_owned(),
            capability: PersistedCapability::DockerDaemon {
                resource: endpoint.resource().to_owned(),
            },
        })
    }

    pub fn workspace(&self) -> &str {
        &self.workspace
    }

    pub fn capability_kind(&self) -> crate::CapabilityKind {
        self.capability.kind()
    }

    pub fn resource(&self) -> &str {
        self.capability.resource()
    }

    fn validate(&self) -> Result<(), WorkspaceGrantDocumentError> {
        if !is_canonical_persisted_path(&self.workspace) {
            return Err(WorkspaceGrantDocumentError::InvalidWorkspace {
                workspace: self.workspace.clone(),
            });
        }
        if !is_canonical_docker_resource(self.resource()) {
            return Err(WorkspaceGrantDocumentError::InvalidDockerResource {
                resource: self.resource().to_owned(),
            });
        }
        Ok(())
    }

    fn key(&self) -> (&str, crate::CapabilityKind) {
        (&self.workspace, self.capability_kind())
    }
}

/// The current in-memory Workspace capability-grant document.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(example = document_example())]
pub struct WorkspaceCapabilityGrantDocument {
    /// Advisory editor schema reference. Runtime decoding remains authoritative.
    #[serde(
        rename = "$schema",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_advisory_schema"
    )]
    #[schemars(schema_with = "advisory_schema")]
    schema: Option<String>,
    /// Persisted format discriminator.
    #[schemars(schema_with = "schema_version_schema")]
    schema_version: u64,
    /// Complete, deterministically ordered grant set.
    #[schemars(length(max = 256))]
    grants: Vec<WorkspaceCapabilityGrant>,
}

impl WorkspaceCapabilityGrantDocument {
    pub fn empty() -> Self {
        Self {
            schema: Some(SCHEMA_REFERENCE.to_owned()),
            schema_version: SCHEMA_VERSION,
            grants: Vec::new(),
        }
    }

    pub fn grants(&self) -> &[WorkspaceCapabilityGrant] {
        &self.grants
    }

    pub(crate) fn effective_approval_grants(
        &self,
        workspace: &crate::Workspace,
        docker_endpoint: Option<&crate::command::DockerEndpoint>,
    ) -> Vec<crate::ApprovalGrant> {
        let Some(workspace) = workspace.root().to_str() else {
            return Vec::new();
        };
        self.effective_approval_grants_for(
            workspace,
            docker_endpoint.map(crate::command::DockerEndpoint::resource),
        )
    }

    fn effective_approval_grants_for(
        &self,
        workspace: &str,
        docker_resource: Option<&str>,
    ) -> Vec<crate::ApprovalGrant> {
        self.grants
            .iter()
            .filter(|grant| grant.workspace == workspace)
            .filter_map(|grant| match (&grant.capability, docker_resource) {
                (PersistedCapability::DockerDaemon { resource }, Some(endpoint))
                    if resource == endpoint =>
                {
                    Some(crate::ApprovalGrant {
                        matcher: crate::ApprovalMatcher::Capability {
                            capability: crate::NamedCapability::docker_daemon(resource),
                        },
                        scope: crate::ApprovalScope::Workspace,
                    })
                }
                (PersistedCapability::DockerDaemon { .. }, _) => None,
            })
            .collect()
    }

    fn remember(&mut self, grant: WorkspaceCapabilityGrant) {
        if let Some(existing) = self
            .grants
            .iter_mut()
            .find(|existing| existing.key() == grant.key())
        {
            *existing = grant;
        } else {
            self.grants.push(grant);
        }
        self.normalize();
    }

    fn normalize(&mut self) {
        self.schema = Some(SCHEMA_REFERENCE.to_owned());
        self.schema_version = SCHEMA_VERSION;
        self.grants.sort_unstable_by(|left, right| {
            left.workspace.cmp(&right.workspace).then_with(|| {
                capability_sort_key(&left.capability).cmp(capability_sort_key(&right.capability))
            })
        });
    }

    fn validate(&mut self) -> Result<(), WorkspaceGrantDocumentError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(WorkspaceGrantDocumentError::WrongVersion {
                expected: SCHEMA_VERSION,
                found: self.schema_version,
            });
        }
        if self.grants.len() > MAX_WORKSPACE_CAPABILITY_GRANTS {
            return Err(WorkspaceGrantDocumentError::TooManyGrants {
                count: self.grants.len(),
                maximum: MAX_WORKSPACE_CAPABILITY_GRANTS,
            });
        }

        let mut keys = HashSet::new();
        for grant in &self.grants {
            grant.validate()?;
            let key = (grant.workspace.clone(), grant.capability_kind());
            if !keys.insert(key) {
                return Err(WorkspaceGrantDocumentError::DuplicateGrant {
                    workspace: grant.workspace.clone(),
                    capability: grant.capability_kind(),
                });
            }
        }
        self.normalize();
        Ok(())
    }
}

impl Default for WorkspaceCapabilityGrantDocument {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "name", rename_all = "snake_case", deny_unknown_fields)]
enum PersistedCapability {
    /// Access to one canonical local Unix Docker daemon endpoint.
    DockerDaemon { resource: String },
}

impl PersistedCapability {
    fn kind(&self) -> crate::CapabilityKind {
        match self {
            Self::DockerDaemon { .. } => crate::CapabilityKind::DockerDaemon,
        }
    }

    fn resource(&self) -> &str {
        match self {
            Self::DockerDaemon { resource } => resource,
        }
    }
}

fn capability_sort_key(capability: &PersistedCapability) -> &'static str {
    match capability {
        PersistedCapability::DockerDaemon { .. } => "docker_daemon",
    }
}

fn is_canonical_persisted_path(path: &str) -> bool {
    if path.is_empty() || path.contains('\0') {
        return false;
    }

    if path.starts_with("//") {
        return is_canonical_unc_path(path, '/');
    }
    if path.starts_with(r"\\") {
        return is_canonical_unc_path(path, '\\');
    }

    let bytes = path.as_bytes();
    if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        let separator = bytes[2] as char;
        return matches!(separator, '/' | '\\')
            && !path[3..].contains(if separator == '/' { '\\' } else { '/' })
            && canonical_segments(&path[3..], separator, true);
    }

    path.starts_with('/') && canonical_segments(&path[1..], '/', true)
}

fn is_canonical_unc_path(path: &str, separator: char) -> bool {
    let remainder = &path[2..];
    if remainder.contains(if separator == '/' { '\\' } else { '/' }) {
        return false;
    }
    let segments = remainder.split(separator).collect::<Vec<_>>();
    segments.len() >= 2
        && !matches!(segments[0], "?" | ".")
        && segments
            .iter()
            .all(|segment| !segment.is_empty() && *segment != "." && *segment != "..")
}

fn canonical_segments(remainder: &str, separator: char, root_allowed: bool) -> bool {
    if remainder.is_empty() {
        return root_allowed;
    }
    remainder
        .split(separator)
        .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

fn is_canonical_docker_resource(resource: &str) -> bool {
    let Some(path) = resource.strip_prefix("unix://") else {
        return false;
    };
    path != "/" && path.starts_with('/') && canonical_segments(&path[1..], '/', false)
}

fn deserialize_advisory_schema<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    String::deserialize(deserializer).map(Some)
}

#[derive(Clone, Copy, Debug)]
struct WorkspaceGrantCodec;

impl DocumentCodec for WorkspaceGrantCodec {
    type Current = WorkspaceCapabilityGrantDocument;
    type Error = WorkspaceGrantDocumentError;

    fn current_version(&self) -> u64 {
        SCHEMA_VERSION
    }

    fn supports_version(&self, version: u64) -> bool {
        version == SCHEMA_VERSION
    }

    fn decode(&self, _version: u64, document: Value) -> Result<Self::Current, Self::Error> {
        let mut document = serde_json::from_value::<WorkspaceCapabilityGrantDocument>(document)
            .map_err(WorkspaceGrantDocumentError::Structure)?;
        document.validate()?;
        Ok(document)
    }
}

struct SupportedWorkspaceGrantDocumentSchema;

impl JsonSchema for SupportedWorkspaceGrantDocumentSchema {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("SupportedWorkspaceCapabilityGrantDocuments")
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let version_zero = generator.subschema_for::<WorkspaceCapabilityGrantDocument>();
        json_schema!({
            "oneOf": [version_zero]
        })
    }
}

/// Core-owned synchronous persistence for the dedicated Workspace capability-grant document.
///
/// All methods that access storage perform blocking filesystem I/O. Call them through
/// `tokio::task::spawn_blocking` (or an equivalent blocking executor) from async code.
#[derive(Clone, Debug)]
pub struct WorkspaceCapabilityGrantStore {
    file: ConfigFile,
}

impl WorkspaceCapabilityGrantStore {
    pub fn new(config_directory: impl AsRef<Path>) -> Result<Self, ConfigFileError> {
        ConfigFile::new(
            config_directory
                .as_ref()
                .join(WORKSPACE_CAPABILITY_GRANTS_DOCUMENT),
        )
        .map(|file| Self { file })
    }

    #[must_use]
    pub fn with_lock_timeout(mut self, timeout: Duration) -> Self {
        self.file = self.file.with_lock_timeout(timeout);
        self
    }

    pub fn document_path(&self) -> &Path {
        self.file.document_path()
    }

    pub fn schema_path(&self) -> &Path {
        self.file.schema_path()
    }

    /// Loads the document using blocking filesystem I/O.
    pub fn load(
        &self,
    ) -> LoadOutcome<WorkspaceCapabilityGrantDocument, WorkspaceGrantDocumentError> {
        self.file.load(&WorkspaceGrantCodec)
    }

    /// Refreshes the generated schema using blocking filesystem I/O.
    pub fn refresh_schema(&self) -> Result<RefreshOutcome, PersistenceError> {
        self.file
            .refresh_schema::<SupportedWorkspaceGrantDocumentSchema>()
    }

    /// Reloads and updates the document while holding a bounded, blocking writer lock.
    pub fn remember(
        &self,
        grant: WorkspaceCapabilityGrant,
    ) -> Result<
        WorkspaceCapabilityGrantDocument,
        UpdateError<WorkspaceGrantDocumentError, Infallible>,
    > {
        self.file.update(&WorkspaceGrantCodec, |document| {
            let mut document = document.unwrap_or_default();
            document.remember(grant);
            Ok(document)
        })
    }

    /// Explicitly archives an invalid document using blocking filesystem I/O.
    pub fn archive_invalid(&self, archive_path: impl AsRef<Path>) -> Result<(), ArchiveError> {
        self.file
            .archive_invalid(&WorkspaceGrantCodec, archive_path)
    }
}

#[derive(Debug, Error)]
pub enum WorkspaceGrantDocumentError {
    #[error("Workspace capability-grant document has invalid structure: {0}")]
    Structure(#[source] serde_json::Error),
    #[error("expected schema version {expected}, found {found}")]
    WrongVersion { expected: u64, found: u64 },
    #[error("document contains {count} grants; the maximum is {maximum}")]
    TooManyGrants { count: usize, maximum: usize },
    #[error(
        "Workspace path must be a recognized, lexically canonical absolute path: {workspace:?}"
    )]
    InvalidWorkspace { workspace: String },
    #[error(
        "Docker resource must identify a lexically canonical absolute local Unix endpoint: {resource:?}"
    )]
    InvalidDockerResource { resource: String },
    #[error("canonical Workspace path `{path}` is not valid UTF-8")]
    NonUtf8Workspace { path: PathBuf },
    #[error("duplicate {capability:?} grant for Workspace {workspace:?}")]
    DuplicateGrant {
        workspace: String,
        capability: crate::CapabilityKind,
    },
}

fn schema_version_schema(_generator: &mut SchemaGenerator) -> Schema {
    json_schema!({"type": "integer", "const": 0})
}

fn advisory_schema(_generator: &mut SchemaGenerator) -> Schema {
    json_schema!({"type": "string"})
}

fn document_example() -> Value {
    serde_json::json!({
        "$schema": SCHEMA_REFERENCE,
        "schema_version": 0,
        "grants": [{
            "workspace": "/canonical/workspace",
            "capability": {
                "name": "docker_daemon",
                "resource": "unix:///canonical/docker.sock"
            }
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cane_config::{InvalidDocument, LoadFailure};
    use serde_json::json;
    use tempfile::tempdir;

    fn grant(workspace: &str, resource: &str) -> WorkspaceCapabilityGrant {
        let grant = WorkspaceCapabilityGrant {
            workspace: workspace.to_owned(),
            capability: PersistedCapability::DockerDaemon {
                resource: resource.to_owned(),
            },
        };
        grant.validate().unwrap();
        grant
    }

    fn store(root: &Path) -> WorkspaceCapabilityGrantStore {
        WorkspaceCapabilityGrantStore::new(root.join("config")).unwrap()
    }

    #[test]
    fn missing_document_loads_without_grants() {
        // Arrange
        let root = tempdir().unwrap();
        let store = store(root.path());

        // Act
        let outcome = store.load();

        // Assert
        assert!(matches!(outcome, LoadOutcome::Missing));
    }

    #[test]
    fn current_document_loads_and_restores_advisory_schema_in_memory_only() {
        // Arrange
        let root = tempdir().unwrap();
        let store = store(root.path());
        std::fs::create_dir_all(store.document_path().parent().unwrap()).unwrap();
        let persisted = br#"{"schema_version":0,"grants":[]}"#;
        std::fs::write(store.document_path(), persisted).unwrap();

        // Act
        let outcome = store.load();

        // Assert
        let LoadOutcome::Loaded(loaded) = outcome else {
            panic!("expected a loaded document");
        };
        assert_eq!(loaded.document.schema.as_deref(), Some(SCHEMA_REFERENCE));
        assert_eq!(std::fs::read(store.document_path()).unwrap(), persisted);
    }

    #[test]
    fn unsupported_version_is_never_treated_as_invalid() {
        // Arrange
        let root = tempdir().unwrap();
        let store = store(root.path());
        std::fs::create_dir_all(store.document_path().parent().unwrap()).unwrap();
        std::fs::write(
            store.document_path(),
            br#"{"schema_version":1,"grants":[]}"#,
        )
        .unwrap();

        // Act
        let outcome = store.load();

        // Assert
        assert!(matches!(
            outcome,
            LoadOutcome::UnsupportedVersion { version: 1 }
        ));
    }

    #[test]
    fn duplicate_and_excess_grants_are_invalid_all_or_nothing() {
        for document in [
            json!({
                "schema_version": 0,
                "grants": [
                    document_example()["grants"][0].clone(),
                    document_example()["grants"][0].clone()
                ]
            }),
            json!({
                "schema_version": 0,
                "grants": (0..=MAX_WORKSPACE_CAPABILITY_GRANTS)
                    .map(|index| json!({
                        "workspace": format!("/workspace/{index}"),
                        "capability": {
                            "name": "docker_daemon",
                            "resource": "unix:///docker.sock"
                        }
                    }))
                    .collect::<Vec<_>>()
            }),
        ] {
            // Arrange
            let root = tempdir().unwrap();
            let store = store(root.path());
            std::fs::create_dir_all(store.document_path().parent().unwrap()).unwrap();
            std::fs::write(
                store.document_path(),
                serde_json::to_vec(&document).unwrap(),
            )
            .unwrap();

            // Act
            let outcome = store.load();

            // Assert
            assert!(matches!(
                outcome,
                LoadOutcome::Invalid(InvalidDocument::Rejected(_))
            ));
        }
    }

    #[test]
    fn present_advisory_schema_must_be_a_string() {
        // Arrange
        let root = tempdir().unwrap();
        let store = store(root.path());
        std::fs::create_dir_all(store.document_path().parent().unwrap()).unwrap();
        std::fs::write(
            store.document_path(),
            br#"{"$schema":null,"schema_version":0,"grants":[]}"#,
        )
        .unwrap();

        // Act
        let outcome = store.load();

        // Assert
        assert!(matches!(
            outcome,
            LoadOutcome::Invalid(InvalidDocument::Rejected(
                WorkspaceGrantDocumentError::Structure(_)
            ))
        ));
    }

    #[test]
    fn persisted_validation_accepts_portable_canonical_absolute_workspace_forms() {
        // Arrange / Act
        let unix = is_canonical_persisted_path("/workspace");
        let drive = is_canonical_persisted_path(r"C:\workspace");
        let unc = is_canonical_persisted_path(r"\\server\share\workspace");
        let relative = is_canonical_persisted_path("relative/workspace");

        // Assert
        assert!(unix);
        assert!(drive);
        assert!(unc);
        assert!(!relative);
    }

    #[test]
    fn persisted_validation_rejects_noncanonical_identity_strings() {
        // Arrange
        let paths = [
            "/workspace/../other",
            "/workspace/./child",
            "/workspace//child",
            "/workspace/",
            r"C:\workspace\..\other",
            r"C:\workspace\\child",
            r"\\server\share\workspace\..\other",
            r"\\?\C:\workspace",
        ];
        let resources = [
            "unix:///tmp/../docker.sock",
            "unix:///tmp/./docker.sock",
            "unix:///tmp//docker.sock",
            "unix:///tmp/docker.sock/",
        ];

        // Act / Assert
        assert!(
            paths
                .into_iter()
                .all(|path| !is_canonical_persisted_path(path))
        );
        assert!(
            resources
                .into_iter()
                .all(|resource| !is_canonical_docker_resource(resource))
        );
    }

    #[test]
    fn persisted_decoding_rejects_noncanonical_identity_strings() {
        // Arrange
        let documents = [
            json!({
                "schema_version": 0,
                "grants": [{
                    "workspace": "/workspace/../other",
                    "capability": {
                        "name": "docker_daemon",
                        "resource": "unix:///docker.sock"
                    }
                }]
            }),
            json!({
                "schema_version": 0,
                "grants": [{
                    "workspace": "/workspace",
                    "capability": {
                        "name": "docker_daemon",
                        "resource": "unix:///tmp/../docker.sock"
                    }
                }]
            }),
        ];

        for document in documents {
            // Act
            let result = WorkspaceGrantCodec.decode(SCHEMA_VERSION, document);

            // Assert
            assert!(matches!(
                result,
                Err(WorkspaceGrantDocumentError::InvalidWorkspace { .. }
                    | WorkspaceGrantDocumentError::InvalidDockerResource { .. })
            ));
        }
    }

    #[test]
    fn effective_grants_require_exact_workspace_and_endpoint_identities() {
        // Arrange
        let document = WorkspaceCapabilityGrantDocument {
            schema: Some(SCHEMA_REFERENCE.to_owned()),
            schema_version: SCHEMA_VERSION,
            grants: vec![grant("/workspace", "unix:///configured.sock")],
        };

        // Act
        let exact =
            document.effective_approval_grants_for("/workspace", Some("unix:///configured.sock"));
        let other_workspace =
            document.effective_approval_grants_for("/other", Some("unix:///configured.sock"));
        let other_endpoint =
            document.effective_approval_grants_for("/workspace", Some("unix:///other.sock"));
        let unavailable_endpoint = document.effective_approval_grants_for("/workspace", None);

        // Assert
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].lifetime(), crate::ApprovalLifetime::Workspace);
        assert!(other_workspace.is_empty());
        assert!(other_endpoint.is_empty());
        assert!(unavailable_endpoint.is_empty());
    }

    #[test]
    fn remembering_replaces_one_workspace_capability_key_and_sorts_output() {
        // Arrange
        let root = tempdir().unwrap();
        let store = store(root.path());
        store
            .remember(grant("/z-workspace", "unix:///old.sock"))
            .unwrap();
        store
            .remember(grant("/a-workspace", "unix:///a.sock"))
            .unwrap();

        // Act
        let document = store
            .remember(grant("/z-workspace", "unix:///new.sock"))
            .unwrap();

        // Assert
        assert_eq!(document.grants.len(), 2);
        assert_eq!(document.grants[0].workspace(), "/a-workspace");
        assert_eq!(document.grants[1].resource(), "unix:///new.sock");
        let persisted = std::fs::read_to_string(store.document_path()).unwrap();
        assert!(persisted.ends_with('\n'));
        assert!(persisted.find("/a-workspace").unwrap() < persisted.find("/z-workspace").unwrap());
        assert!(persisted.contains(SCHEMA_REFERENCE));
    }

    #[test]
    fn locked_remember_refuses_new_semantic_invalidity_without_replacing_document() {
        // Arrange
        let root = tempdir().unwrap();
        let store = store(root.path());
        std::fs::create_dir_all(store.document_path().parent().unwrap()).unwrap();
        let document = json!({
            "$schema": SCHEMA_REFERENCE,
            "schema_version": 0,
            "grants": (0..MAX_WORKSPACE_CAPABILITY_GRANTS)
                .map(|index| json!({
                    "workspace": format!("/workspace/{index:03}"),
                    "capability": {
                        "name": "docker_daemon",
                        "resource": "unix:///docker.sock"
                    }
                }))
                .collect::<Vec<_>>()
        });
        std::fs::write(
            store.document_path(),
            serde_json::to_vec_pretty(&document).unwrap(),
        )
        .unwrap();
        let original = std::fs::read(store.document_path()).unwrap();

        // Act
        let result = store.remember(grant("/one-too-many", "unix:///docker.sock"));

        // Assert
        assert!(matches!(
            result,
            Err(UpdateError::InvalidCurrent(
                WorkspaceGrantDocumentError::TooManyGrants { .. }
            ))
        ));
        assert_eq!(std::fs::read(store.document_path()).unwrap(), original);
    }

    #[test]
    fn schema_refresh_targets_the_document_sidecar() {
        // Arrange
        let root = tempdir().unwrap();
        let store = store(root.path());

        // Act
        let outcome = store.refresh_schema().unwrap();

        // Assert
        assert_eq!(outcome, RefreshOutcome::Written);
        assert_eq!(
            store.schema_path().file_name().unwrap(),
            WORKSPACE_CAPABILITY_GRANTS_SCHEMA
        );
        assert_eq!(
            std::fs::read(store.schema_path()).unwrap(),
            include_bytes!("../schema/workspace-capability-grants.schema.json")
        );
    }

    #[test]
    fn invalid_document_archiving_remains_explicit() {
        // Arrange
        let root = tempdir().unwrap();
        let store = store(root.path());
        std::fs::create_dir_all(store.document_path().parent().unwrap()).unwrap();
        std::fs::write(store.document_path(), b"invalid").unwrap();
        let archive = store.document_path().with_extension("invalid.json");

        // Act
        store.archive_invalid(&archive).unwrap();

        // Assert
        assert!(!store.document_path().exists());
        assert_eq!(std::fs::read(archive).unwrap(), b"invalid");
    }

    #[test]
    fn schema_is_draft_2020_12_useful_and_pinned_to_the_checked_in_copy() {
        // Arrange / Act
        let generated =
            cane_config::generate_schema::<SupportedWorkspaceGrantDocumentSchema>().unwrap();
        let schema: Value = serde_json::from_slice(&generated).unwrap();

        // Assert
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(
            schema["oneOf"][0]["$ref"],
            "#/$defs/WorkspaceCapabilityGrantDocument"
        );
        assert_eq!(
            schema["$defs"]["WorkspaceCapabilityGrantDocument"]["properties"]["grants"]["maxItems"],
            256
        );
        assert_eq!(
            schema["$defs"]["WorkspaceCapabilityGrantDocument"]["examples"][0],
            document_example()
        );
        assert_eq!(
            generated,
            include_bytes!("../schema/workspace-capability-grants.schema.json")
        );
    }

    #[test]
    fn remember_does_not_archive_an_invalid_document_implicitly() {
        // Arrange
        let root = tempdir().unwrap();
        let store = store(root.path());
        std::fs::create_dir_all(store.document_path().parent().unwrap()).unwrap();
        std::fs::write(store.document_path(), b"invalid").unwrap();

        // Act
        let result = store.remember(grant("/workspace", "unix:///docker.sock"));

        // Assert
        assert!(matches!(
            result,
            Err(UpdateError::Load(LoadFailure::Invalid(_)))
        ));
        assert_eq!(std::fs::read(store.document_path()).unwrap(), b"invalid");
    }
}
