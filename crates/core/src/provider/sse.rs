use crate::provider::ProviderError;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct SseEvent {
    pub(crate) event: Option<String>,
    pub(crate) data: String,
}

#[derive(Default)]
pub(crate) struct SseParser {
    buffer: Vec<u8>,
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

        loop {
            match find_end_of_chunk(&self.buffer) {
                Some(end_of_chunk) => {
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
                }
                None => break,
            }
        }

        Ok(events)
    }
}

fn find_end_of_chunk(vec: &[u8]) -> Option<usize> {
    (0..vec.len()).find_map(|i| {
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

    /// A data-only event (the OpenAI-compat shape: no `event:` line).
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
        // Chunks are not aligned to events and can be split mid-value.
        // The first chunk yields nothing (no terminator yet); the
        // second completes it.

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
        // Lines starting with `:` are comments (proxy keep-alives). The
        // comment shares a block with a real data line here, so the event
        // still emits, carrying only the data.

        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b": keep-alive\ndata: after comment\n\n").unwrap();

        // Assert
        assert_eq!(events, vec![data_event("after comment")]);
    }

    #[test]
    fn feed_emits_nothing_for_a_lone_comment_block() {
        // A `:keepalive` followed by a blank line is a complete block with no
        // data, not an empty event.

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
        // Anthropic tags each event with an `event:` line preceding its `data:`.

        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"event: content_block_delta\ndata: {\"text\":\"hi\"}\n\n").unwrap();

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
        // Per spec, multiple `data:` lines in one event join with `\n`. Rare on
        // OpenAI-compat servers but valid and Anthropic-relevant.

        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"data: line one\ndata: line two\n\n").unwrap();

        // Assert
        assert_eq!(events, vec![data_event("line one\nline two")]);
    }

    #[test]
    fn feed_handles_crlf_line_endings() {
        // The spec allows `\r\n`, so `\r\n\r\n` also terminates an event, and
        // some proxies re-chunk with CRLF even when the origin didn't.

        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"data: hello\r\n\r\n").unwrap();

        // Assert
        assert_eq!(events, vec![data_event("hello")]);
    }

    #[test]
    fn feed_reassembles_a_multibyte_codepoint_split_across_chunks() {
        // 👋 is a 4-byte codepoint that the network chunk can split anywhere,
        // including mid-codepoint. A per-chunk String conversion would panic
        // or corrupt here.

        // Arrange
        let mut parser = SseParser::default();
        let wave = "👋".as_bytes(); // [0xF0, 0x9F, 0x91, 0x8B]

        // Act: split the emoji down the middle across two chunks
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
        // `[DONE]` gets no special treatment in the parser since it's just data.
        // The stream adapter, not this parser, decides it means end-of-stream.

        // Arrange
        let mut parser = SseParser::default();

        // Act
        let events = parser.feed(b"data: [DONE]\n\n").unwrap();

        // Assert
        assert_eq!(events, vec![data_event("[DONE]")]);
    }
}
