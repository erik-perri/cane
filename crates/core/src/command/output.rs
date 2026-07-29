pub const MAX_COMMAND_RESULT_BYTES: usize = 32 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandOutputChunk {
    pub bytes: Vec<u8>,
    pub stream: CommandOutputStream,
}

impl CommandOutputChunk {
    pub fn stderr(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            stream: CommandOutputStream::Stderr,
        }
    }

    pub fn stdout(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            stream: CommandOutputStream::Stdout,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandOutputStream {
    Stderr,
    Stdout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedOutput {
    pub chunks: Vec<CommandOutputChunk>,
    pub stderr_bytes: u64,
    pub stdout_bytes: u64,
}

impl CapturedOutput {
    pub fn complete(chunks: Vec<CommandOutputChunk>) -> Self {
        let mut stderr_bytes = 0_u64;
        let mut stdout_bytes = 0_u64;

        for chunk in &chunks {
            let bytes = u64::try_from(chunk.bytes.len()).unwrap_or(u64::MAX);

            match chunk.stream {
                CommandOutputStream::Stderr => {
                    stderr_bytes = stderr_bytes.saturating_add(bytes);
                }
                CommandOutputStream::Stdout => {
                    stdout_bytes = stdout_bytes.saturating_add(bytes);
                }
            }
        }

        Self {
            chunks,
            stderr_bytes,
            stdout_bytes,
        }
    }

    pub fn total_bytes(&self) -> u64 {
        observed_stream_bytes(self, CommandOutputStream::Stderr)
            .saturating_add(observed_stream_bytes(self, CommandOutputStream::Stdout))
    }

    pub fn stderr_truncated(&self) -> bool {
        stream_was_truncated(self, CommandOutputStream::Stderr)
    }

    pub fn stdout_truncated(&self) -> bool {
        stream_was_truncated(self, CommandOutputStream::Stdout)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandResult {
    pub output: CapturedOutput,
    pub termination: CommandTermination,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandTermination {
    Exited { code: i32 },
    Signaled { signal: i32 },
    TimedOut,
}

pub fn format_command_result(result: &CommandResult) -> String {
    let retained_bytes = combined_bytes(&result.output);
    let output = String::from_utf8_lossy(&retained_bytes);
    let total_bytes = result.output.total_bytes();
    let termination = termination_line(&result.termination);
    let complete_header = output_header(total_bytes, false);
    let complete_budget = MAX_COMMAND_RESULT_BYTES
        .saturating_sub(termination.len())
        .saturating_sub(complete_header.len());
    let truncated = total_bytes > retained_bytes.len() as u64 || output.len() > complete_budget;
    let header = output_header(total_bytes, truncated);
    let content_budget = MAX_COMMAND_RESULT_BYTES
        .saturating_sub(termination.len())
        .saturating_sub(header.len());

    let mut formatted = String::with_capacity(MAX_COMMAND_RESULT_BYTES);
    formatted.push_str(&termination);
    formatted.push_str(&header);
    formatted.push_str(tail(&output, content_budget));
    debug_assert!(formatted.len() <= MAX_COMMAND_RESULT_BYTES);

    formatted
}

fn combined_bytes(output: &CapturedOutput) -> Vec<u8> {
    let retained_bytes = output
        .chunks
        .iter()
        .map(|chunk| chunk.bytes.len())
        .fold(0_usize, usize::saturating_add);
    let mut combined = Vec::with_capacity(retained_bytes);

    for chunk in &output.chunks {
        combined.extend_from_slice(&chunk.bytes);
    }

    combined
}

fn observed_stream_bytes(output: &CapturedOutput, stream: CommandOutputStream) -> u64 {
    let retained_bytes = output
        .chunks
        .iter()
        .filter(|chunk| chunk.stream == stream)
        .map(|chunk| u64::try_from(chunk.bytes.len()).unwrap_or(u64::MAX))
        .fold(0_u64, u64::saturating_add);
    let observed_bytes = match stream {
        CommandOutputStream::Stderr => output.stderr_bytes,
        CommandOutputStream::Stdout => output.stdout_bytes,
    };

    observed_bytes.max(retained_bytes)
}

fn stream_was_truncated(output: &CapturedOutput, stream: CommandOutputStream) -> bool {
    let retained_bytes = output
        .chunks
        .iter()
        .filter(|chunk| chunk.stream == stream)
        .map(|chunk| u64::try_from(chunk.bytes.len()).unwrap_or(u64::MAX))
        .fold(0_u64, u64::saturating_add);

    observed_stream_bytes(output, stream) > retained_bytes
}

fn output_header(total_bytes: u64, truncated: bool) -> String {
    let status = if truncated {
        "truncated; showing tail"
    } else {
        "complete"
    };

    format!("output ({total_bytes} bytes, {status}):\n")
}

fn tail(value: &str, budget: usize) -> &str {
    if value.len() <= budget {
        return value;
    }

    let mut start = value.len() - budget;
    while !value.is_char_boundary(start) {
        start += 1;
    }

    &value[start..]
}

fn termination_line(termination: &CommandTermination) -> String {
    match termination {
        CommandTermination::Exited { code } => {
            format!("process exited with code {code}\n")
        }
        CommandTermination::Signaled { signal } => {
            format!("process terminated by signal {signal}\n")
        }
        CommandTermination::TimedOut => "process timed out\n".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(output: CapturedOutput, termination: CommandTermination) -> CommandResult {
        CommandResult {
            output,
            termination,
        }
    }

    #[test]
    fn captured_output_counts_each_stream_without_losing_chunk_order() {
        // Arrange
        let chunks = vec![
            CommandOutputChunk::stdout(b"starting\n".to_vec()),
            CommandOutputChunk::stderr(b"warning\n".to_vec()),
            CommandOutputChunk::stdout(b"done\n".to_vec()),
        ];

        // Act
        let output = CapturedOutput::complete(chunks.clone());

        // Assert
        assert_eq!(output.chunks, chunks);
        assert_eq!(output.stderr_bytes, 8);
        assert_eq!(output.stdout_bytes, 14);
        assert_eq!(output.total_bytes(), 22);
    }

    #[test]
    fn format_command_result_preserves_interleaved_output_and_exit_status() {
        // Arrange
        let output = CapturedOutput::complete(vec![
            CommandOutputChunk::stdout(b"compiled successfully\n".to_vec()),
            CommandOutputChunk::stderr(b"one warning\n".to_vec()),
            CommandOutputChunk::stdout(b"done\n".to_vec()),
        ]);
        let result = result(output, CommandTermination::Exited { code: 7 });

        // Act
        let formatted = format_command_result(&result);

        // Assert
        assert_eq!(
            formatted,
            "process exited with code 7\n\
             output (39 bytes, complete):\n\
             compiled successfully\n\
             one warning\n\
             done\n"
        );
    }

    #[test]
    fn format_command_result_decodes_invalid_utf8_lossily() {
        // Arrange
        let result = result(
            CapturedOutput::complete(vec![CommandOutputChunk::stdout(vec![
                b'o', b'k', b':', 0xff,
            ])]),
            CommandTermination::Exited { code: 0 },
        );

        // Act
        let formatted = format_command_result(&result);

        // Assert
        assert!(formatted.contains("ok:\u{fffd}"));
        assert!(formatted.contains("process exited with code 0"));
    }

    #[test]
    fn format_command_result_decodes_utf8_split_across_observed_chunks() {
        // Arrange
        let crab = "🦀".as_bytes();
        let result = result(
            CapturedOutput::complete(vec![
                CommandOutputChunk::stdout(crab[..2].to_vec()),
                CommandOutputChunk::stdout(crab[2..].to_vec()),
            ]),
            CommandTermination::Exited { code: 0 },
        );

        // Act
        let formatted = format_command_result(&result);

        // Assert
        assert!(formatted.contains('🦀'));
        assert!(!formatted.contains('\u{fffd}'));
    }

    #[test]
    fn format_command_result_favors_the_combined_output_tail() {
        // Arrange
        let mut stdout = vec![b'a'; MAX_COMMAND_RESULT_BYTES];
        let result = result(
            CapturedOutput::complete(vec![
                CommandOutputChunk::stdout(std::mem::take(&mut stdout)),
                CommandOutputChunk::stderr(b"STDERR_MIDDLE".to_vec()),
                CommandOutputChunk::stdout(b"STDOUT_TAIL".to_vec()),
            ]),
            CommandTermination::Signaled { signal: 9 },
        );

        // Act
        let formatted = format_command_result(&result);

        // Assert
        assert_eq!(formatted.len(), MAX_COMMAND_RESULT_BYTES);
        assert!(formatted.contains("process terminated by signal 9"));
        assert!(formatted.contains("output (32792 bytes, truncated; showing tail)"));
        assert!(formatted.contains("STDERR_MIDDLE"));
        assert!(formatted.contains("STDOUT_TAIL"));
        assert!(formatted.find("STDERR_MIDDLE").unwrap() < formatted.find("STDOUT_TAIL").unwrap());
    }

    #[test]
    fn format_command_result_reports_bytes_discarded_during_capture() {
        // Arrange
        let result = result(
            CapturedOutput {
                chunks: vec![
                    CommandOutputChunk::stderr(b"retained stderr tail\n".to_vec()),
                    CommandOutputChunk::stdout(b"retained stdout tail".to_vec()),
                ],
                stderr_bytes: 75_000,
                stdout_bytes: 50_000,
            },
            CommandTermination::TimedOut,
        );

        // Act
        let formatted = format_command_result(&result);

        // Assert
        assert!(formatted.contains("process timed out"));
        assert!(formatted.contains("output (125000 bytes, truncated; showing tail)"));
        assert!(formatted.contains("retained stderr tail"));
        assert!(formatted.contains("retained stdout tail"));
        assert!(
            formatted.find("retained stderr tail").unwrap()
                < formatted.find("retained stdout tail").unwrap()
        );
        assert!(result.output.stderr_truncated());
        assert!(result.output.stdout_truncated());
    }

    #[test]
    fn format_command_result_keeps_utf8_valid_when_truncating_output() {
        // Arrange
        let repeated = "🦀".repeat(MAX_COMMAND_RESULT_BYTES);
        let result = result(
            CapturedOutput::complete(vec![CommandOutputChunk::stdout(repeated.into_bytes())]),
            CommandTermination::Exited { code: 0 },
        );

        // Act
        let formatted = format_command_result(&result);

        // Assert
        assert!(formatted.len() <= MAX_COMMAND_RESULT_BYTES);
        assert!(formatted.contains('🦀'));
        assert!(formatted.is_char_boundary(formatted.len()));
    }
}
