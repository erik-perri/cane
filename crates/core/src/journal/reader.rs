use super::{JOURNAL_SCHEMA_VERSION, JournalEntry, JournalRecord, SessionId};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JournalReadError {
    #[error("session journal is empty")]
    Empty,

    #[error("session journal does not begin with session_started")]
    InvalidFirstRecord,

    #[error("malformed JSON on journal line {line}: {source}")]
    MalformedLine {
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("journal line {line} belongs to {actual_session_id}, expected {expected_session_id}")]
    MismatchedSession {
        actual_session_id: SessionId,
        expected_session_id: SessionId,
        line: usize,
    },

    #[error("journal line {line} has sequence {actual}, expected {expected}")]
    SequenceGap {
        actual: u64,
        expected: u64,
        line: usize,
    },

    #[error("session journal has a non-newline-terminated tail")]
    UnterminatedTail,

    #[error(
        "journal line {line} uses unsupported schema version {actual}; this build supports {supported}"
    )]
    UnsupportedVersion {
        actual: u32,
        line: usize,
        supported: u32,
    },
}

pub fn parse_journal(input: &[u8]) -> Result<Vec<JournalRecord>, JournalReadError> {
    if input.is_empty() {
        return Err(JournalReadError::Empty);
    }
    if !input.ends_with(b"\n") {
        return Err(JournalReadError::UnterminatedTail);
    }

    let mut records = Vec::new();
    let mut expected_session_id = None;

    for (index, line) in input[..input.len() - 1]
        .split(|byte| *byte == b'\n')
        .enumerate()
    {
        let line_number = index + 1;
        let expected_sequence = u64::try_from(line_number).expect("journal line count exceeds u64");
        let record = serde_json::from_slice::<JournalRecord>(line).map_err(|source| {
            JournalReadError::MalformedLine {
                line: line_number,
                source,
            }
        })?;

        if record.schema_version != JOURNAL_SCHEMA_VERSION {
            return Err(JournalReadError::UnsupportedVersion {
                actual: record.schema_version,
                line: line_number,
                supported: JOURNAL_SCHEMA_VERSION,
            });
        }
        if record.sequence != expected_sequence {
            return Err(JournalReadError::SequenceGap {
                actual: record.sequence,
                expected: expected_sequence,
                line: line_number,
            });
        }

        match expected_session_id {
            Some(session_id) if record.session_id != session_id => {
                return Err(JournalReadError::MismatchedSession {
                    actual_session_id: record.session_id,
                    expected_session_id: session_id,
                    line: line_number,
                });
            }
            None => expected_session_id = Some(record.session_id),
            Some(_) => {}
        }

        records.push(record);
    }

    if !matches!(
        records.first().map(|record| &record.entry),
        Some(JournalEntry::SessionStarted(_))
    ) {
        return Err(JournalReadError::InvalidFirstRecord);
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::{RunEndReason, RunEnded, SessionStarted};

    fn session_id() -> SessionId {
        "sess_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap()
    }

    fn record(sequence: u64, entry: JournalEntry) -> JournalRecord {
        JournalRecord::new(
            sequence,
            "2026-07-27T12:00:00.123Z".parse().unwrap(),
            session_id(),
            entry,
        )
    }

    fn session_started() -> JournalEntry {
        JournalEntry::SessionStarted(SessionStarted {
            cane_version: "0.1.0".to_string(),
            instructions: "Be helpful.".to_string(),
            workspace: "/workspace".to_string(),
        })
    }

    fn encode(records: &[JournalRecord]) -> Vec<u8> {
        let mut encoded = Vec::new();
        for record in records {
            serde_json::to_writer(&mut encoded, record).unwrap();
            encoded.push(b'\n');
        }
        encoded
    }

    #[test]
    fn parse_accepts_complete_contiguous_records() {
        // Arrange
        let records = vec![
            record(1, session_started()),
            record(
                2,
                JournalEntry::RunEnded(RunEnded {
                    reason: RunEndReason::UserQuit,
                    run_id: "run_01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap(),
                }),
            ),
        ];
        let encoded = encode(&records);

        // Act
        let parsed = parse_journal(&encoded).unwrap();

        // Assert
        assert_eq!(parsed, records);
    }

    #[test]
    fn parse_rejects_an_unsupported_version_sequence_gap_and_session_mismatch() {
        // Arrange
        let mut unsupported = record(1, session_started());
        unsupported.schema_version = JOURNAL_SCHEMA_VERSION + 1;

        let sequence_gap = vec![record(1, session_started()), record(3, session_started())];

        let mut mismatched = record(2, session_started());
        mismatched.session_id = "sess_01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap();
        let mismatched = vec![record(1, session_started()), mismatched];

        // Act
        let unsupported_error = parse_journal(&encode(&[unsupported])).unwrap_err();
        let sequence_error = parse_journal(&encode(&sequence_gap)).unwrap_err();
        let mismatch_error = parse_journal(&encode(&mismatched)).unwrap_err();

        // Assert
        assert!(matches!(
            unsupported_error,
            JournalReadError::UnsupportedVersion { line: 1, .. }
        ));
        assert!(matches!(
            sequence_error,
            JournalReadError::SequenceGap {
                expected: 2,
                actual: 3,
                line: 2
            }
        ));
        assert!(matches!(
            mismatch_error,
            JournalReadError::MismatchedSession { line: 2, .. }
        ));
    }

    #[test]
    fn parse_rejects_malformed_complete_lines_and_unterminated_tails() {
        // Arrange
        let malformed = b"{not json}\n";
        let unterminated = b"{\"otherwise\":\"valid\"}";
        let mut invalid_timestamp = serde_json::to_value(record(1, session_started())).unwrap();
        invalid_timestamp["recorded_at"] = serde_json::json!("not-a-timestamp");
        let mut invalid_timestamp = serde_json::to_vec(&invalid_timestamp).unwrap();
        invalid_timestamp.push(b'\n');

        // Act
        let malformed_error = parse_journal(malformed).unwrap_err();
        let unterminated_error = parse_journal(unterminated).unwrap_err();
        let timestamp_error = parse_journal(&invalid_timestamp).unwrap_err();

        // Assert
        assert!(matches!(
            malformed_error,
            JournalReadError::MalformedLine { line: 1, .. }
        ));
        assert!(matches!(
            unterminated_error,
            JournalReadError::UnterminatedTail
        ));
        assert!(matches!(
            timestamp_error,
            JournalReadError::MalformedLine { line: 1, .. }
        ));
    }

    #[test]
    fn parse_requires_session_started_as_the_first_record() {
        // Arrange
        let encoded = encode(&[record(
            1,
            JournalEntry::RunEnded(RunEnded {
                reason: RunEndReason::UserQuit,
                run_id: "run_01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap(),
            }),
        )]);

        // Act
        let error = parse_journal(&encoded).unwrap_err();

        // Assert
        assert!(matches!(error, JournalReadError::InvalidFirstRecord));
    }

    #[test]
    fn parse_rejects_an_empty_journal() {
        // Arrange
        let encoded = [];

        // Act
        let error = parse_journal(&encoded).unwrap_err();

        // Assert
        assert!(matches!(error, JournalReadError::Empty));
    }
}
