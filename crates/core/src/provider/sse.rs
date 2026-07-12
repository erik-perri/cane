use crate::provider::ProviderError;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct SseEvent {
    pub(crate) event: Option<String>,
    pub(crate) data: String,
}

#[derive(Default)]
pub(crate) struct SseParser {
    buffer: Vec<u8>,
    /// Bytes at the front of `buffer` already known to contain no terminator
    /// start. Lets each `feed` scan only the newly appended region instead of
    /// rescanning the whole buffer.
    scan_from: usize,
}

impl SseParser {
    /// Feed a network chunk and return any events it completed.
    ///
    /// Supported grammar:
    /// - Events terminate on a blank line: `\n\n` or `\r\n\r\n`.
    ///   Mixed or lone-`\r` terminators are not recognized
    /// - A single leading space is stripped from each field value
    ///   (`data: x` -> `x`), per spec.
    ///
    /// Errors are **fatal for the stream**: a `Utf8Error` leaves the offending
    /// block buffered, so re-feeding after an `Err` returns the same error. The
    /// adapter must treat any `Err` as end-of-stream and stop feeding.
    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, ProviderError> {
        self.buffer.extend_from_slice(chunk);

        let mut events = Vec::new();

        // A terminator can straddle the previous scan boundary, so back up by
        // three bytes (the longest terminator is `\r\n\r\n`, and only its final
        // byte can be new) before resuming the scan.
        let mut search_from = self.scan_from.saturating_sub(3);

        while let Some(end_of_chunk) = find_end_of_chunk(&self.buffer, search_from) {
            let chunk_bytes = &self.buffer[..end_of_chunk];
            let chunk = std::str::from_utf8(chunk_bytes)?.to_string();

            let mut event = None;
            let mut data_lines = Vec::new();

            for line in chunk.lines() {
                if line.is_empty() {
                    continue;
                }

                // Comment
                if line.starts_with(":") {
                    continue;
                }

                if let Some((line_type, line_data)) = line.split_once(':') {
                    let trimmed_line_data = line_data.strip_prefix(' ').unwrap_or(line_data);
                    match line_type {
                        "event" => event = Some(trimmed_line_data.to_string()),
                        "data" => data_lines.push(trimmed_line_data.to_string()),
                        _ => (),
                    }
                } else {
                    continue;
                }
            }

            if !data_lines.is_empty() {
                events.push(SseEvent {
                    data: data_lines.join("\n"),
                    event,
                })
            }

            self.buffer.drain(..end_of_chunk);

            // The buffer shifted; rescan the remainder from the front.
            search_from = 0;
            self.scan_from = 0;
        }

        // No complete event remains. The next feed only needs to reconsider
        // the final three bytes, which may begin a split CRLF terminator.
        self.scan_from = self.buffer.len();

        Ok(events)
    }
}

fn find_end_of_chunk(vec: &[u8], from: usize) -> Option<usize> {
    (from..vec.len()).find_map(|i| {
        let remainder = &vec[i..];

        if remainder.starts_with(b"\r\n\r\n") {
            Some(i + 4)
        } else if remainder.starts_with(b"\n\n") {
            Some(i + 2)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_event(data: &str) -> SseEvent {
        SseEvent {
            event: None,
            data: data.to_string(),
        }
    }

    #[test]
    fn feed_decodes_one_event_in_one_chunk() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"data: {\"content\":\"Hello\"}\n\n").unwrap();

        // Assert
        assert_eq!(events, vec![data_event("{\"content\":\"Hello\"}")]);
    }

    #[test]
    fn feed_buffers_an_event_split_mid_data_across_two_chunks() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let first = parser.feed(b"data: {\"content\":\"Hel").unwrap();
        let second = parser.feed(b"lo\"}\n\n").unwrap();

        // Assert
        assert!(first.is_empty(), "no complete event yet: {first:?}");
        assert_eq!(second, vec![data_event("{\"content\":\"Hello\"}")]);
    }

    #[test]
    fn feed_decodes_two_events_in_one_chunk() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"data: first\n\ndata: second\n\n").unwrap();

        // Assert
        assert_eq!(events, vec![data_event("first"), data_event("second")]);
    }

    #[test]
    fn feed_skips_comment_lines() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser
            .feed(b": keep-alive\ndata: after comment\n\n")
            .unwrap();

        // Assert
        assert_eq!(events, vec![data_event("after comment")]);
    }

    #[test]
    fn feed_emits_nothing_for_a_lone_comment_block() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b": keep-alive\n\n").unwrap();

        // Assert
        assert!(
            events.is_empty(),
            "keep-alive should emit no event: {events:?}"
        );
    }

    #[test]
    fn feed_decodes_an_anthropic_style_event_and_data_pair() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser
            .feed(b"event: content_block_delta\ndata: {\"text\":\"hi\"}\n\n")
            .unwrap();

        // Assert
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("content_block_delta".to_string()),
                data: "{\"text\":\"hi\"}".to_string(),
            }]
        );
    }

    #[test]
    fn feed_concatenates_multiple_data_lines_with_newlines() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"data: line one\ndata: line two\n\n").unwrap();

        // Assert
        assert_eq!(events, vec![data_event("line one\nline two")]);
    }

    #[test]
    fn feed_handles_crlf_line_endings() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"data: hello\r\n\r\n").unwrap();

        // Assert
        assert_eq!(events, vec![data_event("hello")]);
    }

    #[test]
    fn feed_reassembles_a_multibyte_codepoint_split_across_chunks() {
        // Arrange
        let mut parser = SseParser::default();
        let wave = "👋".as_bytes();

        // Act
        let mut first_chunk = b"data: wave ".to_vec();
        first_chunk.extend_from_slice(&wave[..2]);
        let first = parser.feed(&first_chunk).unwrap();

        let mut second_chunk = wave[2..].to_vec();
        second_chunk.extend_from_slice(b"\n\n");
        let second = parser.feed(&second_chunk).unwrap();

        // Assert
        assert!(first.is_empty(), "no complete event yet: {first:?}");
        assert_eq!(second, vec![data_event("wave 👋")]);
    }

    #[test]
    fn feed_treats_the_done_sentinel_as_ordinary_data() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"data: [DONE]\n\n").unwrap();

        // Assert
        assert_eq!(events, vec![data_event("[DONE]")]);
    }

    #[test]
    fn feed_completes_a_terminator_split_across_the_scan_boundary() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let first = parser.feed(b"data: hi\r\n\r").unwrap();
        let second = parser.feed(b"\n").unwrap();

        // Assert
        assert!(first.is_empty(), "no complete event yet: {first:?}");
        assert_eq!(second, vec![data_event("hi")]);
    }

    #[test]
    fn feed_fails_on_invalid_utf8_and_repeats_the_error_on_the_next_feed() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let first_error = parser
            .feed(b"data: {\"content\":\"Hello\xff\"}\n\n")
            .unwrap_err();

        // Assert
        let ProviderError::Parsing(first_utf8_error) = first_error else {
            panic!("expected parsing error, got {first_error:?}")
        };
        matches!(first_utf8_error, std::str::Utf8Error { .. });

        let events = parser.feed(&[]).unwrap_err();
        let ProviderError::Parsing(second_utf8_error) = events else {
            panic!("expected parsing error, got {events:?}")
        };
        assert_eq!(first_utf8_error, second_utf8_error);
    }

    #[test]
    fn feed_emits_nothing_for_an_event_field_without_data() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"event: ping\n\n").unwrap();

        // Assert
        assert!(
            events.is_empty(),
            "block with no data should emit no event: {events:?}"
        );
    }

    #[test]
    fn feed_advances_the_scan_cursor_for_an_incomplete_event() {
        // Arrange
        let mut parser = SseParser::default();

        // Act
        let first = parser.feed(b"data: partial").unwrap();
        let first_cursor = parser.scan_from;
        let second = parser.feed(b" event").unwrap();
        let second_cursor = parser.scan_from;

        // Assert
        assert!(first.is_empty());
        assert_eq!(first_cursor, b"data: partial".len());
        assert!(second.is_empty());
        assert_eq!(second_cursor, b"data: partial event".len());
    }
}
