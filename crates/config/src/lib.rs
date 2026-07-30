//! Domain-neutral infrastructure for versioned JSON configuration documents.
//!
//! Filesystem operations in this crate are synchronous. Writer operations may wait up to their
//! configured timeout for another writer's OS lock. Async callers must run them on a blocking
//! thread pool, such as with `tokio::task::spawn_blocking`, rather than on an async runtime worker.

mod document;
mod storage;

#[cfg(test)]
mod tests;

pub use document::{
    DocumentCodec, InvalidDocument, InvalidVersion, LoadFailure, LoadIoError, LoadOutcome,
    LoadedDocument, pretty_json,
};
pub use storage::{
    ArchiveError, ArchiveRefusal, ConfigFile, ConfigFileError, DEFAULT_LOCK_TIMEOUT,
    PersistenceError, RefreshOutcome, UpdateError, generate_schema,
};
