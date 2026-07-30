use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use schemars::generate::SchemaSettings;
use tempfile::Builder;
use thiserror::Error;

use crate::document::{
    DecodeFailure, DocumentCodec, LoadFailure, LoadIoError, LoadOutcome, decode_document,
    pretty_json_value,
};

/// How long writer operations wait for the sidecar OS lock unless overridden.
pub const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(5);

const LOCK_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// One versioned document and its derived adjacent lock and schema sidecars.
#[derive(Debug, Clone)]
pub struct ConfigFile {
    document_path: PathBuf,
    lock_path: PathBuf,
    schema_path: PathBuf,
    lock_timeout: Duration,
}

impl ConfigFile {
    /// Creates a configuration file façade and derives its adjacent sidecar paths.
    ///
    /// For `settings.json`, the sidecars are `settings.lock` and `settings.schema.json`.
    pub fn new(document_path: impl Into<PathBuf>) -> Result<Self, ConfigFileError> {
        let document_path = document_path.into();
        if document_path
            .parent()
            .is_none_or(|parent| parent.as_os_str().is_empty())
        {
            return Err(ConfigFileError::MissingParent(document_path));
        }
        if document_path.file_name().is_none_or(|name| name.is_empty()) {
            return Err(ConfigFileError::MissingFileName(document_path));
        }
        if document_path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("lock"))
        {
            return Err(ConfigFileError::DerivedPathCollision {
                document: document_path,
                sidecar_kind: "lock",
            });
        }

        let lock_path = document_path.with_extension("lock");
        let schema_path = document_path.with_extension("schema.json");
        for (kind, sidecar) in [("lock", &lock_path), ("schema", &schema_path)] {
            if sidecar == &document_path {
                return Err(ConfigFileError::DerivedPathCollision {
                    document: document_path,
                    sidecar_kind: kind,
                });
            }
        }

        debug_assert_eq!(lock_path.parent(), document_path.parent());
        debug_assert_eq!(schema_path.parent(), document_path.parent());
        Ok(Self {
            document_path,
            lock_path,
            schema_path,
            lock_timeout: DEFAULT_LOCK_TIMEOUT,
        })
    }

    /// Overrides how long writer operations wait to acquire the sidecar OS lock.
    ///
    /// A zero duration makes acquisition non-blocking while still allowing an immediately
    /// available lock to succeed.
    #[must_use]
    pub fn with_lock_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    /// How long writer operations wait to acquire the sidecar OS lock.
    pub fn lock_timeout(&self) -> Duration {
        self.lock_timeout
    }

    /// The versioned JSON document path.
    pub fn document_path(&self) -> &Path {
        &self.document_path
    }

    /// The stable OS-lock sidecar path.
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    /// The adjacent generated schema path.
    pub fn schema_path(&self) -> &Path {
        &self.schema_path
    }

    /// Loads a document without rewriting or migrating its persisted bytes.
    ///
    /// This performs blocking filesystem I/O.
    pub fn load<C>(&self, codec: &C) -> LoadOutcome<C::Current, C::Error>
    where
        C: DocumentCodec,
    {
        match self.read_document() {
            Ok(None) => LoadOutcome::Missing,
            Ok(Some(bytes)) => match decode_document(&bytes, codec) {
                Ok(document) => LoadOutcome::Loaded(document),
                Err(DecodeFailure::Invalid(error)) => LoadOutcome::Invalid(error),
                Err(DecodeFailure::UnsupportedVersion(version)) => {
                    LoadOutcome::UnsupportedVersion { version }
                }
            },
            Err(error) => LoadOutcome::Io(error),
        }
    }

    /// Takes the writer lock, reloads the latest document, applies `mutate`, and durably replaces
    /// the document with the current model.
    ///
    /// Missing documents are passed to `mutate` as `None`. Invalid, unsupported, and unreadable
    /// documents stop the update before the callback runs. The serialized mutation is decoded
    /// again before replacement, guaranteeing that a successful write passes the codec's current
    /// structural and semantic validation. `mutate` runs while the writer lock is held; it should
    /// remain short and must not call another lock-taking operation for this configuration file.
    ///
    /// This performs blocking filesystem I/O and may wait up to the configured timeout for the
    /// writer lock. Async callers must run it through a blocking thread pool.
    pub fn update<C, F, E>(
        &self,
        codec: &C,
        mutate: F,
    ) -> Result<C::Current, UpdateError<C::Error, E>>
    where
        C: DocumentCodec,
        F: FnOnce(Option<C::Current>) -> Result<C::Current, E>,
    {
        let _lock = self
            .acquire_writer_lock()
            .map_err(UpdateError::Persistence)?;
        let current = match self.load(codec) {
            LoadOutcome::Missing => None,
            LoadOutcome::Loaded(loaded) => Some(loaded.document),
            LoadOutcome::Invalid(error) => {
                return Err(UpdateError::Load(LoadFailure::Invalid(error)));
            }
            LoadOutcome::UnsupportedVersion { version } => {
                return Err(UpdateError::Load(LoadFailure::UnsupportedVersion {
                    version,
                }));
            }
            LoadOutcome::Io(error) => {
                return Err(UpdateError::Load(LoadFailure::Io(error)));
            }
        };

        let updated = mutate(current).map_err(UpdateError::Mutation)?;
        let value = serde_json::to_value(&updated)
            .map_err(PersistenceError::Serialize)
            .map_err(UpdateError::Persistence)?;
        let found_version = value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64);
        if found_version != Some(codec.current_version()) {
            return Err(UpdateError::Persistence(PersistenceError::CurrentVersion {
                expected: codec.current_version(),
                found: found_version,
            }));
        }
        if !codec.supports_version(codec.current_version()) {
            return Err(UpdateError::UnsupportedCurrentVersion {
                version: codec.current_version(),
            });
        }
        codec
            .decode(codec.current_version(), value.clone())
            .map_err(UpdateError::InvalidCurrent)?;
        let bytes = pretty_json_value(&value)
            .map_err(PersistenceError::Serialize)
            .map_err(UpdateError::Persistence)?;
        self.atomic_replace(&self.document_path, &bytes)
            .map_err(UpdateError::Persistence)?;
        Ok(updated)
    }

    /// Creates or refreshes the generated schema only when its canonical bytes differ.
    ///
    /// This performs blocking filesystem I/O and may wait up to the configured timeout for the
    /// writer lock. Async callers must run it through a blocking thread pool.
    pub fn refresh_schema<T>(&self) -> Result<RefreshOutcome, PersistenceError>
    where
        T: JsonSchema,
    {
        let _lock = self.acquire_writer_lock()?;
        let bytes = generate_schema::<T>()?;
        match fs::read(&self.schema_path) {
            Ok(existing) if existing == bytes => {
                normalize_file_permissions(&self.schema_path)?;
                Ok(RefreshOutcome::Unchanged)
            }
            Ok(_) => {
                self.atomic_replace(&self.schema_path, &bytes)?;
                Ok(RefreshOutcome::Written)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                self.atomic_replace(&self.schema_path, &bytes)?;
                Ok(RefreshOutcome::Written)
            }
            Err(source) => Err(PersistenceError::io(
                "read schema",
                &self.schema_path,
                source,
            )),
        }
    }

    /// Archives the current document only if the supplied codec classifies it as invalid.
    ///
    /// The caller chooses the archive path and therefore owns the recovery authorization and
    /// naming policy. Unsupported versions are deliberately refused.
    ///
    /// This performs blocking filesystem I/O and may wait up to the configured timeout for the
    /// writer lock. Async callers must run it through a blocking thread pool.
    pub fn archive_invalid<C>(
        &self,
        codec: &C,
        archive_path: impl AsRef<Path>,
    ) -> Result<(), ArchiveError>
    where
        C: DocumentCodec,
    {
        let archive_path = archive_path.as_ref();
        let _lock = self
            .acquire_writer_lock()
            .map_err(ArchiveError::Persistence)?;
        match self.load(codec) {
            LoadOutcome::Missing => {
                return Err(ArchiveError::Refused(ArchiveRefusal::Missing));
            }
            LoadOutcome::Loaded(_) => {
                return Err(ArchiveError::Refused(ArchiveRefusal::Valid));
            }
            LoadOutcome::UnsupportedVersion { version } => {
                return Err(ArchiveError::Refused(ArchiveRefusal::UnsupportedVersion {
                    version,
                }));
            }
            LoadOutcome::Io(error) => return Err(ArchiveError::Load(error)),
            LoadOutcome::Invalid(_) => {}
        }

        let document_parent = parent(&self.document_path)?;
        if archive_path.parent() != Some(document_parent) {
            return Err(ArchiveError::Persistence(
                PersistenceError::ArchiveNotAdjacent {
                    document: self.document_path.clone(),
                    archive: archive_path.to_path_buf(),
                },
            ));
        }
        if archive_path
            .try_exists()
            .map_err(|source| PersistenceError::io("inspect archive", archive_path, source))?
        {
            return Err(ArchiveError::Persistence(PersistenceError::io(
                "create archive",
                archive_path,
                io::Error::new(io::ErrorKind::AlreadyExists, "archive already exists"),
            )));
        }

        fs::rename(&self.document_path, archive_path)
            .map_err(|source| PersistenceError::io("archive document", archive_path, source))?;
        normalize_file_permissions(archive_path).map_err(ArchiveError::Persistence)?;
        sync_directory(document_parent).map_err(ArchiveError::Persistence)?;
        Ok(())
    }

    fn read_document(&self) -> Result<Option<Vec<u8>>, LoadIoError> {
        if let Some(directory) = self.document_path.parent()
            && directory.exists()
        {
            normalize_directory_permissions(directory).map_err(|error| LoadIoError {
                operation: "set permissions on",
                path: directory.to_path_buf(),
                source: error.into_io(),
            })?;
        }
        match fs::read(&self.document_path) {
            Ok(bytes) => {
                normalize_file_permissions(&self.document_path).map_err(|error| LoadIoError {
                    operation: "set permissions on",
                    path: self.document_path.clone(),
                    source: error.into_io(),
                })?;
                Ok(Some(bytes))
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(LoadIoError {
                operation: "read",
                path: self.document_path.clone(),
                source,
            }),
        }
    }

    pub(crate) fn acquire_writer_lock(&self) -> Result<WriterLock, PersistenceError> {
        let directory = parent(&self.lock_path)?;
        ensure_directory(directory)?;

        let file = open_restrictive(&self.lock_path)?;
        normalize_file_permissions(&self.lock_path)?;
        let started = Instant::now();
        loop {
            match file.try_lock() {
                Ok(()) => break,
                Err(fs::TryLockError::WouldBlock) => {
                    let waited = started.elapsed();
                    if waited >= self.lock_timeout {
                        return Err(PersistenceError::LockTimeout {
                            path: self.lock_path.clone(),
                            waited,
                        });
                    }
                    thread::sleep(LOCK_POLL_INTERVAL.min(self.lock_timeout - waited));
                }
                Err(fs::TryLockError::Error(source)) => {
                    return Err(PersistenceError::io("lock", &self.lock_path, source));
                }
            }
        }
        Ok(WriterLock { _file: file })
    }

    fn atomic_replace(&self, path: &Path, bytes: &[u8]) -> Result<(), PersistenceError> {
        let directory = parent(path)?;
        ensure_directory(directory)?;
        let mut temporary = Builder::new()
            .prefix(".cane-config-")
            .tempfile_in(directory)
            .map_err(|source| {
                PersistenceError::io("create temporary file in", directory, source)
            })?;
        normalize_file_permissions(temporary.path())?;
        temporary.write_all(bytes).map_err(|source| {
            PersistenceError::io("write temporary file", temporary.path(), source)
        })?;
        temporary.flush().map_err(|source| {
            PersistenceError::io("flush temporary file", temporary.path(), source)
        })?;
        temporary.as_file().sync_all().map_err(|source| {
            PersistenceError::io("sync temporary file", temporary.path(), source)
        })?;
        temporary
            .persist(path)
            .map_err(|error| PersistenceError::io("atomically replace", path, error.error))?;
        normalize_file_permissions(path)?;
        sync_directory(directory)?;
        Ok(())
    }
}

pub(crate) struct WriterLock {
    _file: File,
}

/// Result of comparing and refreshing an adjacent generated schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshOutcome {
    /// Existing bytes already matched; no replacement occurred.
    Unchanged,
    /// Missing or differing schema bytes were atomically written.
    Written,
}

/// Invalid document paths that prevent safe sidecar derivation.
#[derive(Debug, Error)]
pub enum ConfigFileError {
    /// The document path did not identify a containing directory.
    #[error("configuration document path has no parent: {0}")]
    MissingParent(PathBuf),
    /// The document path did not contain a file name.
    #[error("configuration document path has no file name: {0}")]
    MissingFileName(PathBuf),
    /// A derived sidecar path would overwrite the document.
    #[error("derived {sidecar_kind} path aliases configuration document {document}")]
    DerivedPathCollision {
        document: PathBuf,
        sidecar_kind: &'static str,
    },
}

/// Produces a deterministic Draft 2020-12 schema document with a trailing newline.
pub fn generate_schema<T>() -> Result<Vec<u8>, PersistenceError>
where
    T: JsonSchema,
{
    let schema = SchemaSettings::draft2020_12()
        .into_generator()
        .into_root_schema_for::<T>();
    let value = serde_json::to_value(schema).map_err(PersistenceError::Serialize)?;
    pretty_json_value(&value).map_err(PersistenceError::Serialize)
}

/// A filesystem or serialization failure during persistence.
#[derive(Debug, Error)]
pub enum PersistenceError {
    /// A filesystem operation failed.
    #[error("failed to {operation} {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// JSON serialization failed.
    #[error("failed to serialize JSON: {0}")]
    Serialize(#[source] serde_json::Error),
    /// The sidecar writer lock remained occupied until the configured deadline.
    #[error("timed out after {waited:?} waiting for writer lock {path}")]
    LockTimeout { path: PathBuf, waited: Duration },
    /// The current model did not serialize with the codec's current version.
    #[error("current document must contain schema_version {expected}, found {found:?}")]
    CurrentVersion { expected: u64, found: Option<u64> },
    /// The requested archive was not adjacent to its document.
    #[error("archive {archive} is not adjacent to document {document}")]
    ArchiveNotAdjacent { document: PathBuf, archive: PathBuf },
    /// A path that requires a parent directory had none.
    #[error("configuration path has no parent: {0}")]
    MissingParent(PathBuf),
}

impl PersistenceError {
    fn io(operation: &'static str, path: &Path, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.to_path_buf(),
            source,
        }
    }

    fn into_io(self) -> io::Error {
        match self {
            Self::Io { source, .. } => source,
            other => io::Error::other(other),
        }
    }
}

/// Failure from a locked reload-before-write update.
#[derive(Debug, Error)]
pub enum UpdateError<DocumentError, MutationError> {
    /// Reloading found an invalid, unsupported, or unreadable document.
    #[error("cannot update because reloading the document failed")]
    Load(LoadFailure<DocumentError>),
    /// The owner-provided mutation rejected the update.
    #[error("configuration mutation failed")]
    Mutation(MutationError),
    /// The codec does not accept the version it declares current.
    #[error("document codec does not support its current schema version {version}")]
    UnsupportedCurrentVersion { version: u64 },
    /// The codec rejected the serialized current document before it was written.
    #[error("serialized current document failed validation")]
    InvalidCurrent(DocumentError),
    /// Locking, serialization, or durable replacement failed.
    #[error(transparent)]
    Persistence(#[from] PersistenceError),
}

/// A reason the archive operation deliberately left the document untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ArchiveRefusal {
    /// There was no document to archive.
    #[error("the document is missing")]
    Missing,
    /// The codec accepted the document.
    #[error("the document is valid")]
    Valid,
    /// Unsupported versions must never be archived automatically.
    #[error("schema version {version} is unsupported")]
    UnsupportedVersion { version: u64 },
}

/// Failure from an explicitly requested invalid-document archive.
#[derive(Debug, Error)]
pub enum ArchiveError {
    /// The operation was unsafe or inapplicable for the current document.
    #[error(transparent)]
    Refused(ArchiveRefusal),
    /// Loading failed for an I/O reason.
    #[error(transparent)]
    Load(LoadIoError),
    /// Locking, path validation, renaming, permission normalization, or syncing failed.
    #[error(transparent)]
    Persistence(#[from] PersistenceError),
}

fn parent(path: &Path) -> Result<&Path, PersistenceError> {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| PersistenceError::MissingParent(path.to_path_buf()))
}

fn ensure_directory(path: &Path) -> Result<(), PersistenceError> {
    fs::create_dir_all(path)
        .map_err(|source| PersistenceError::io("create directory", path, source))?;
    normalize_directory_permissions(path)
}

fn open_restrictive(path: &Path) -> Result<File, PersistenceError> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .map_err(|source| PersistenceError::io("open", path, source))
}

#[cfg(unix)]
fn normalize_directory_permissions(path: &Path) -> Result<(), PersistenceError> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|source| PersistenceError::io("set directory permissions on", path, source))
}

#[cfg(not(unix))]
fn normalize_directory_permissions(_path: &Path) -> Result<(), PersistenceError> {
    Ok(())
}

#[cfg(unix)]
fn normalize_file_permissions(path: &Path) -> Result<(), PersistenceError> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .map_err(|source| PersistenceError::io("set file permissions on", path, source))
}

#[cfg(not(unix))]
fn normalize_file_permissions(_path: &Path) -> Result<(), PersistenceError> {
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), PersistenceError> {
    match File::open(path).and_then(|directory| directory.sync_all()) {
        Ok(()) => Ok(()),
        Err(source)
            if matches!(
                source.kind(),
                io::ErrorKind::InvalidInput | io::ErrorKind::Unsupported
            ) =>
        {
            Ok(())
        }
        Err(source) => Err(PersistenceError::io("sync directory", path, source)),
    }
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<(), PersistenceError> {
    Ok(())
}
