use super::{JournalEntry, JournalRecord, SessionId};
use jiff::Timestamp;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncWrite, AsyncWriteExt};

#[cfg(unix)]
const DIRECTORY_MODE: u32 = 0o700;
#[cfg(unix)]
const FILE_MODE: u32 = 0o600;

#[derive(Debug, Error)]
pub enum JournalError {
    #[error("failed to {operation} '{}': {source}", path.display())]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to serialize session journal '{}': {source}", path.display())]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("session journal '{}' is unusable after an earlier write failure", path.display())]
    Unusable { path: PathBuf },
}

pub struct SessionJournal {
    writer: JournalWriter<File>,
}

struct JournalWriter<W> {
    next_sequence: u64,
    path: PathBuf,
    poisoned: bool,
    session_id: SessionId,
    sink: W,
}

impl SessionJournal {
    pub async fn create(
        sessions_directory: impl AsRef<Path>,
        session_id: SessionId,
    ) -> Result<Self, JournalError> {
        let sessions_directory = sessions_directory.as_ref();
        let path = journal_path(sessions_directory, session_id);

        create_sessions_directory(sessions_directory).await?;

        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        options.mode(FILE_MODE);

        let sink = options
            .open(&path)
            .await
            .map_err(|source| JournalError::Io {
                operation: "create session journal",
                path: path.clone(),
                source,
            })?;

        Ok(Self {
            writer: JournalWriter {
                next_sequence: 1,
                path,
                poisoned: false,
                session_id,
                sink,
            },
        })
    }

    pub async fn append(&mut self, entry: JournalEntry) -> Result<JournalRecord, JournalError> {
        self.writer.append_at(Timestamp::now(), entry).await
    }

    #[cfg(test)]
    async fn append_at(
        &mut self,
        recorded_at: Timestamp,
        entry: JournalEntry,
    ) -> Result<JournalRecord, JournalError> {
        self.writer.append_at(recorded_at, entry).await
    }

    pub fn path(&self) -> &Path {
        &self.writer.path
    }

    pub fn session_id(&self) -> &SessionId {
        &self.writer.session_id
    }
}

impl<W> JournalWriter<W>
where
    W: AsyncWrite + Unpin,
{
    async fn append_at(
        &mut self,
        recorded_at: Timestamp,
        entry: JournalEntry,
    ) -> Result<JournalRecord, JournalError> {
        if self.poisoned {
            return Err(JournalError::Unusable {
                path: self.path.clone(),
            });
        }

        let record = JournalRecord::new(self.next_sequence, recorded_at, self.session_id, entry);
        let mut encoded = serde_json::to_vec(&record).map_err(|source| {
            self.poisoned = true;
            JournalError::Serialize {
                path: self.path.clone(),
                source,
            }
        })?;
        encoded.push(b'\n');

        if let Err(source) = self.sink.write_all(&encoded).await {
            self.poisoned = true;
            return Err(JournalError::Io {
                operation: "append to session journal",
                path: self.path.clone(),
                source,
            });
        }

        if let Err(source) = self.sink.flush().await {
            self.poisoned = true;
            return Err(JournalError::Io {
                operation: "flush session journal",
                path: self.path.clone(),
                source,
            });
        }

        self.next_sequence += 1;
        Ok(record)
    }
}

async fn create_sessions_directory(path: &Path) -> Result<(), JournalError> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    builder.mode(DIRECTORY_MODE);

    builder
        .create(path)
        .await
        .map_err(|source| JournalError::Io {
            operation: "create sessions directory",
            path: path.to_path_buf(),
            source,
        })?;

    #[cfg(unix)]
    fs::set_permissions(path, std::fs::Permissions::from_mode(DIRECTORY_MODE))
        .await
        .map_err(|source| JournalError::Io {
            operation: "set permissions on sessions directory",
            path: path.to_path_buf(),
            source,
        })?;

    Ok(())
}

fn journal_path(sessions_directory: &Path, session_id: SessionId) -> PathBuf {
    sessions_directory.join(format!("{session_id}.jsonl"))
}

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::{RunEndReason, RunEnded, SessionStarted};
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use tempfile::TempDir;

    struct FlushFailure;

    impl AsyncWrite for FlushFailure {
        fn poll_flush(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Err(io::Error::other("injected flush failure")))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_write(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<io::Result<usize>> {
            Poll::Ready(Ok(buffer.len()))
        }
    }

    fn session_id() -> SessionId {
        "sess_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap()
    }

    fn session_started() -> JournalEntry {
        JournalEntry::SessionStarted(SessionStarted {
            cane_version: "0.1.0".to_string(),
            instructions: "Be helpful.".to_string(),
            workspace: "/workspace".to_string(),
        })
    }

    #[tokio::test]
    async fn create_and_append_write_a_flushed_jsonl_record() {
        // Arrange
        let temporary = TempDir::new().unwrap();
        let sessions_directory = temporary.path().join("sessions");
        let mut journal = SessionJournal::create(&sessions_directory, session_id())
            .await
            .unwrap();

        // Act
        let record = journal
            .append_at(
                "2026-07-27T12:00:00.123Z".parse().unwrap(),
                session_started(),
            )
            .await
            .unwrap();
        let contents = fs::read_to_string(journal.path()).await.unwrap();

        // Assert
        assert_eq!(record.sequence, 1);
        assert_eq!(journal.session_id(), &session_id());
        assert_eq!(
            journal.path(),
            sessions_directory.join("sess_01ARZ3NDEKTSV4RRFFQ69G5FAV.jsonl")
        );
        assert!(contents.ends_with('\n'));
        assert_eq!(contents.lines().count(), 1);
        assert_eq!(
            serde_json::from_str::<JournalRecord>(contents.trim_end()).unwrap(),
            record
        );
    }

    #[tokio::test]
    async fn append_assigns_contiguous_sequences_while_the_writer_is_live() {
        // Arrange
        let temporary = TempDir::new().unwrap();
        let mut journal = SessionJournal::create(temporary.path(), session_id())
            .await
            .unwrap();

        // Act
        let first = journal
            .append_at(
                "2026-07-27T12:00:00.123Z".parse().unwrap(),
                session_started(),
            )
            .await
            .unwrap();
        let second = journal
            .append_at(
                "2026-07-27T12:00:01.456Z".parse().unwrap(),
                JournalEntry::RunEnded(RunEnded {
                    reason: RunEndReason::UserQuit,
                    run_id: "run_01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap(),
                }),
            )
            .await
            .unwrap();
        let contents = fs::read_to_string(journal.path()).await.unwrap();
        let records = contents
            .lines()
            .map(|line| serde_json::from_str::<JournalRecord>(line).unwrap())
            .collect::<Vec<_>>();

        // Assert
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert_eq!(records, vec![first, second]);
    }

    #[tokio::test]
    async fn create_never_overwrites_an_existing_session() {
        // Arrange
        let temporary = TempDir::new().unwrap();
        let first = SessionJournal::create(temporary.path(), session_id())
            .await
            .unwrap();
        let path = first.path().to_path_buf();

        // Act
        let error = SessionJournal::create(temporary.path(), session_id())
            .await
            .err()
            .unwrap();

        // Assert
        assert!(matches!(
            error,
            JournalError::Io {
                operation: "create session journal",
                ..
            }
        ));
        assert!(path.exists());
    }

    #[tokio::test]
    async fn a_failed_flush_poisons_the_writer_against_retries() {
        // Arrange
        let mut writer = JournalWriter {
            next_sequence: 1,
            path: PathBuf::from("journal.jsonl"),
            poisoned: false,
            session_id: session_id(),
            sink: FlushFailure,
        };

        // Act
        let first_error = writer
            .append_at(
                "2026-07-27T12:00:00.123Z".parse().unwrap(),
                session_started(),
            )
            .await
            .unwrap_err();
        let retry_error = writer
            .append_at(
                "2026-07-27T12:00:01.456Z".parse().unwrap(),
                session_started(),
            )
            .await
            .unwrap_err();

        // Assert
        assert!(matches!(
            first_error,
            JournalError::Io {
                operation: "flush session journal",
                ..
            }
        ));
        assert!(matches!(retry_error, JournalError::Unusable { .. }));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn create_uses_restrictive_directory_and_file_permissions() {
        // Arrange
        let temporary = TempDir::new().unwrap();
        let sessions_directory = temporary.path().join("sessions");

        // Act
        let journal = SessionJournal::create(&sessions_directory, session_id())
            .await
            .unwrap();
        let directory_mode = fs::metadata(&sessions_directory)
            .await
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let file_mode = fs::metadata(journal.path())
            .await
            .unwrap()
            .permissions()
            .mode()
            & 0o777;

        // Assert
        assert_eq!(directory_mode, DIRECTORY_MODE);
        assert_eq!(file_mode, FILE_MODE);
    }
}
