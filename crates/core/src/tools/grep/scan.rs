use super::{GrepLimits, OutputMode};
use crate::tools::file_discovery::LocatedFile;
use crate::tools::path_display::RenderedPath;
use crate::tools::{MAX_FILE_SIZE_BYTES, ToolExecutionError, operation_failed};
use grep_matcher::Matcher;
use grep_regex::RegexMatcher;
use grep_searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkContext, SinkMatch};
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::ops::Range;
use tokio_util::sync::CancellationToken;

pub(super) struct ScanOptions {
    pub(super) cancel: CancellationToken,
    pub(super) context: usize,
    pub(super) explicit_file: bool,
    pub(super) limits: GrepLimits,
    pub(super) matcher: RegexMatcher,
    pub(super) multiline: bool,
    pub(super) output_mode: OutputMode,
    pub(super) requested_path: String,
}

#[derive(Debug)]
pub(super) struct GrepCandidate {
    pub(super) file: LocatedFile,
    pub(super) rendered_path: RenderedPath,
}

pub(super) enum GrepFileResult {
    Binary,
    Content {
        lines: BTreeMap<u64, GrepLine>,
        match_truncated: bool,
        path: RenderedPath,
    },
    Path(RenderedPath),
}

impl GrepFileResult {
    fn matched_line_count(&self) -> usize {
        match self {
            Self::Content { lines, .. } => lines.values().filter(|line| line.matched).count(),
            Self::Binary | Self::Path(_) => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct GrepWarnings {
    pub(super) oversized_files: usize,
    pub(super) unsearchable_files: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct GrepTruncation {
    pub(super) global_matches: bool,
    pub(super) paths: bool,
}

pub(super) struct GrepScan {
    pub(super) results: Vec<GrepFileResult>,
    pub(super) truncation: GrepTruncation,
    pub(super) warnings: GrepWarnings,
}

pub(super) fn scan_candidates(
    candidates: Vec<GrepCandidate>,
    options: ScanOptions,
) -> Result<GrepScan, ToolExecutionError> {
    let ScanOptions {
        cancel,
        context,
        explicit_file,
        limits,
        matcher,
        multiline,
        output_mode,
        requested_path,
    } = options;

    if cancel.is_cancelled() {
        return Err(ToolExecutionError::Cancelled);
    }

    let effective_context = match output_mode {
        OutputMode::Content => context,
        OutputMode::FilesWithMatches => 0,
    };
    let mut builder = SearcherBuilder::new();
    builder
        .line_number(true)
        .multi_line(multiline)
        .bom_sniffing(true)
        .binary_detection(BinaryDetection::quit(b'\0'))
        .heap_limit(Some(MAX_FILE_SIZE_BYTES as usize))
        .before_context(effective_context)
        .after_context(effective_context);

    let mut searcher = builder.build();
    let candidate_count = candidates.len();
    let mut results = Vec::new();
    let mut truncation = GrepTruncation::default();
    let mut warnings = GrepWarnings::default();
    let mut committed_matches = 0;
    let mut committed_paths = 0;

    for (index, candidate) in candidates.into_iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(ToolExecutionError::Cancelled);
        }

        let result = match scan_one_file(
            &mut searcher,
            &matcher,
            candidate,
            output_mode,
            limits.matched_lines_per_file,
            cancel.clone(),
        ) {
            Ok(result) => result,
            Err(FileScanError::Cancelled) => return Err(ToolExecutionError::Cancelled),
            Err(FileScanError::Oversized(size)) if explicit_file => {
                let detail = size.map_or_else(
                    || {
                        format!(
                            "file grew beyond the {MAX_FILE_SIZE_BYTES} byte grep limit while \
                             being searched"
                        )
                    },
                    |size| {
                        format!(
                            "file is {size} bytes, which exceeds the {MAX_FILE_SIZE_BYTES} byte \
                             grep limit"
                        )
                    },
                );
                return Err(operation_failed("grep", &requested_path, detail).into());
            }
            Err(FileScanError::Oversized(_)) => {
                warnings.oversized_files += 1;
                continue;
            }
            Err(FileScanError::Unsearchable(error)) if explicit_file => {
                return Err(operation_failed("grep", &requested_path, error).into());
            }
            Err(FileScanError::Unsearchable(_)) => {
                warnings.unsearchable_files += 1;
                continue;
            }
        };

        match result {
            None => {}
            Some(GrepFileResult::Binary) => {
                if explicit_file {
                    results.push(GrepFileResult::Binary);
                }
            }
            Some(result @ GrepFileResult::Content { .. }) => {
                let result_matches = result.matched_line_count();
                let remaining_matches = limits.matched_lines.saturating_sub(committed_matches);
                let accepted_matches = result_matches.min(remaining_matches);
                let result_proves_more = result_matches > remaining_matches
                    || matches!(
                        result,
                        GrepFileResult::Content {
                            match_truncated: true,
                            ..
                        }
                    );

                if accepted_matches > 0 {
                    results.push(result);
                }
                committed_matches += accepted_matches;

                if committed_matches == limits.matched_lines {
                    truncation.global_matches = result_proves_more || index + 1 < candidate_count;
                    break;
                }
            }
            Some(result @ GrepFileResult::Path(_)) => {
                if committed_paths < limits.paths {
                    results.push(result);
                    committed_paths += 1;
                }

                if committed_paths == limits.paths {
                    truncation.paths = index + 1 < candidate_count;
                    break;
                }
            }
        }
    }

    if cancel.is_cancelled() {
        return Err(ToolExecutionError::Cancelled);
    }

    Ok(GrepScan {
        results,
        truncation,
        warnings,
    })
}

enum FileScanError {
    Cancelled,
    Oversized(Option<u64>),
    Unsearchable(io::Error),
}

fn scan_one_file(
    searcher: &mut Searcher,
    matcher: &RegexMatcher,
    candidate: GrepCandidate,
    output_mode: OutputMode,
    matched_line_limit: usize,
    cancel: CancellationToken,
) -> Result<Option<GrepFileResult>, FileScanError> {
    if cancel.is_cancelled() {
        return Err(FileScanError::Cancelled);
    }

    let file = std::fs::File::open(&candidate.file.path).map_err(FileScanError::Unsearchable)?;
    let metadata = file.metadata().map_err(FileScanError::Unsearchable)?;
    if !metadata.is_file() {
        return Err(FileScanError::Unsearchable(io::Error::other(
            "path is no longer a regular file",
        )));
    }
    if metadata.len() > MAX_FILE_SIZE_BYTES {
        return Err(FileScanError::Oversized(Some(metadata.len())));
    }

    let reader = GrepReader::new(file, cancel.clone());
    let mut sink = GrepSink::new(output_mode, matcher, matched_line_limit, cancel);
    if let Err(error) = searcher.search_reader(matcher, reader, &mut sink) {
        return match scan_control(&error) {
            Some(ScanControl::Cancelled) => Err(FileScanError::Cancelled),
            Some(ScanControl::Oversized) => Err(FileScanError::Oversized(None)),
            None => Err(FileScanError::Unsearchable(error)),
        };
    }

    let result = match (output_mode, sink.stop_reason, sink.has_match) {
        (_, Some(StopReason::Binary), _) => Some(GrepFileResult::Binary),
        (OutputMode::FilesWithMatches, _, true) => {
            Some(GrepFileResult::Path(candidate.rendered_path))
        }
        (OutputMode::Content, stop_reason, true) => Some(GrepFileResult::Content {
            lines: sink.lines,
            match_truncated: stop_reason == Some(StopReason::MatchLimit),
            path: candidate.rendered_path,
        }),
        (_, _, false) => None,
    };

    Ok(result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum ScanControl {
    #[error("grep scan was cancelled")]
    Cancelled,
    #[error("file exceeded the grep size limit while being searched")]
    Oversized,
}

fn scan_control_error(control: ScanControl) -> io::Error {
    io::Error::other(control)
}

pub(super) fn scan_control(error: &io::Error) -> Option<ScanControl> {
    error
        .get_ref()
        .and_then(|source| source.downcast_ref::<ScanControl>())
        .copied()
}

pub(super) struct GrepReader<R> {
    bytes_read: u64,
    cancel: CancellationToken,
    inner: R,
}

impl<R> GrepReader<R> {
    pub(super) fn new(inner: R, cancel: CancellationToken) -> Self {
        Self {
            bytes_read: 0,
            cancel,
            inner,
        }
    }
}

impl<R: Read> Read for GrepReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.cancel.is_cancelled() {
            return Err(scan_control_error(ScanControl::Cancelled));
        }
        if buffer.is_empty() {
            return Ok(0);
        }
        if self.bytes_read == MAX_FILE_SIZE_BYTES {
            let mut probe = [0_u8; 1];
            return match self.inner.read(&mut probe)? {
                0 => Ok(0),
                _ => Err(scan_control_error(ScanControl::Oversized)),
            };
        }

        let remaining = (MAX_FILE_SIZE_BYTES - self.bytes_read) as usize;
        let read_len = buffer.len().min(remaining);
        let read = self.inner.read(&mut buffer[..read_len])?;
        self.bytes_read += read as u64;

        if self.cancel.is_cancelled() {
            return Err(scan_control_error(ScanControl::Cancelled));
        }

        Ok(read)
    }
}

#[derive(Debug)]
pub(super) struct GrepLine {
    pub(super) bytes: Vec<u8>,
    pub(super) first_match: Option<Range<usize>>,
    pub(super) matched: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StopReason {
    FirstMatch,
    MatchLimit,
    Binary,
}

pub(super) struct GrepSink {
    cancel: CancellationToken,
    pub(super) has_match: bool,
    pub(super) lines: BTreeMap<u64, GrepLine>,
    matched_line_limit: usize,
    pub(super) matched_lines: usize,
    matcher: RegexMatcher,
    output_mode: OutputMode,
    pub(super) stop_reason: Option<StopReason>,
}

impl GrepSink {
    pub(super) fn new(
        output_mode: OutputMode,
        matcher: &RegexMatcher,
        matched_line_limit: usize,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            cancel,
            output_mode,
            lines: BTreeMap::new(),
            has_match: false,
            matched_line_limit,
            matched_lines: 0,
            matcher: matcher.clone(),
            stop_reason: None,
        }
    }
}

impl Sink for GrepSink {
    type Error = io::Error;

    fn matched(&mut self, _searcher: &Searcher, matched: &SinkMatch<'_>) -> io::Result<bool> {
        if self.cancel.is_cancelled() {
            return Err(scan_control_error(ScanControl::Cancelled));
        }

        self.has_match = true;

        if self.output_mode == OutputMode::FilesWithMatches {
            self.stop_reason = Some(StopReason::FirstMatch);
            return Ok(false);
        }

        let first_line = matched
            .line_number()
            .ok_or_else(|| io::Error::other("grep searcher omitted line numbers"))?;
        let first_match = self
            .matcher
            .find(matched.bytes())
            .map_err(|error| io::Error::other(error.to_string()))?
            .map(|matched| matched.start()..matched.end());
        let mut line_offset = 0;

        for (offset, bytes) in matched.lines().enumerate() {
            if self.cancel.is_cancelled() {
                return Err(scan_control_error(ScanControl::Cancelled));
            }

            let line_number = first_line + offset as u64;
            let line_end = line_offset + bytes.len();
            let line_match = first_match.as_ref().and_then(|matched| {
                let intersects = if matched.is_empty() {
                    matched.start >= line_offset && matched.start <= line_end
                } else {
                    matched.start < line_end && matched.end > line_offset
                };

                intersects.then(|| {
                    matched.start.saturating_sub(line_offset).min(bytes.len())
                        ..matched.end.saturating_sub(line_offset).min(bytes.len())
                })
            });
            let already_matched = self
                .lines
                .get(&line_number)
                .is_some_and(|line| line.matched);
            if !already_matched && self.matched_lines == self.matched_line_limit {
                self.stop_reason = Some(StopReason::MatchLimit);
                return Ok(false);
            }
            if !already_matched {
                self.matched_lines += 1;
            }

            self.lines
                .entry(line_number)
                .and_modify(|line| {
                    line.bytes = bytes.to_vec();
                    if line.first_match.is_none() {
                        line.first_match = line_match.clone();
                    }
                    line.matched = true;
                })
                .or_insert_with(|| GrepLine {
                    bytes: bytes.to_vec(),
                    first_match: line_match,
                    matched: true,
                });
            line_offset = line_end;
        }

        Ok(true)
    }

    fn context(&mut self, _searcher: &Searcher, context: &SinkContext<'_>) -> io::Result<bool> {
        if self.cancel.is_cancelled() {
            return Err(scan_control_error(ScanControl::Cancelled));
        }

        let line_number = context
            .line_number()
            .ok_or_else(|| io::Error::other("grep searcher omitted line numbers"))?;

        // An existing matched record wins over context.
        self.lines.entry(line_number).or_insert_with(|| GrepLine {
            bytes: context.bytes().to_vec(),
            first_match: None,
            matched: false,
        });

        Ok(true)
    }

    fn binary_data(&mut self, _searcher: &Searcher, _offset: u64) -> io::Result<bool> {
        if self.cancel.is_cancelled() {
            return Err(scan_control_error(ScanControl::Cancelled));
        }

        self.lines.clear();
        self.has_match = false;
        self.stop_reason = Some(StopReason::Binary);
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        MAX_GREP_MATCHED_LINES, MAX_GREP_MATCHED_LINES_PER_FILE, MAX_GREP_OUTPUT_BYTES,
        MAX_GREP_PATHS, MAX_GREP_SOURCE_LINE_BYTES, compile_matcher,
    };
    use super::*;
    use crate::tools::file_discovery::LocatedFile;
    use crate::tools::path_display::RenderedPath;
    use grep_searcher::{BinaryDetection, SearcherBuilder};
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    fn limits() -> GrepLimits {
        GrepLimits {
            matched_lines: MAX_GREP_MATCHED_LINES,
            matched_lines_per_file: MAX_GREP_MATCHED_LINES_PER_FILE,
            output_bytes: MAX_GREP_OUTPUT_BYTES,
            paths: MAX_GREP_PATHS,
            source_line_bytes: MAX_GREP_SOURCE_LINE_BYTES,
        }
    }

    fn search_sink(
        pattern: &str,
        contents: &[u8],
        case_insensitive: bool,
        multiline: bool,
        context: usize,
        output_mode: OutputMode,
    ) -> GrepSink {
        let matcher = compile_matcher(pattern, case_insensitive, multiline).unwrap();
        let mut builder = SearcherBuilder::new();

        builder
            .line_number(true)
            .multi_line(multiline)
            .binary_detection(BinaryDetection::quit(b'\0'))
            .before_context(context)
            .after_context(context);

        let mut searcher = builder.build();
        let mut sink = GrepSink::new(
            output_mode,
            &matcher,
            MAX_GREP_MATCHED_LINES_PER_FILE,
            CancellationToken::new(),
        );
        searcher
            .search_slice(&matcher, contents, &mut sink)
            .unwrap();
        sink
    }

    fn line_numbers(sink: &GrepSink) -> Vec<u64> {
        sink.lines.keys().copied().collect()
    }

    fn candidate(path: &Path, rendered_path: &str) -> GrepCandidate {
        GrepCandidate {
            file: LocatedFile::from_metadata(path.to_path_buf(), fs::metadata(path).unwrap()),
            rendered_path: RenderedPath {
                text: rendered_path.to_string(),
                lossy: false,
                json_escaped: false,
            },
        }
    }

    fn scan_options(
        output_mode: OutputMode,
        limits: GrepLimits,
        explicit_file: bool,
        requested_path: &str,
    ) -> ScanOptions {
        ScanOptions {
            cancel: CancellationToken::new(),
            context: 0,
            explicit_file,
            limits,
            matcher: compile_matcher("needle", false, false).unwrap(),
            multiline: false,
            output_mode,
            requested_path: requested_path.to_string(),
        }
    }

    #[test]
    fn case_sensitivity_is_controlled_by_the_option_or_inline_flags() {
        // Arrange / Act
        let sensitive = search_sink("needle", b"Needle\n", false, false, 0, OutputMode::Content);
        let insensitive = search_sink("needle", b"Needle\n", true, false, 0, OutputMode::Content);
        let inline = search_sink(
            "(?i)needle",
            b"Needle\n",
            false,
            false,
            0,
            OutputMode::Content,
        );

        // Assert
        assert!(!sensitive.has_match);
        assert!(insensitive.has_match);
        assert!(inline.has_match);
    }

    #[test]
    fn slash_delimited_javascript_regex_syntax_is_literal_not_a_flag_wrapper() {
        // Arrange / Act
        let ordinary_line = search_sink(
            "/needle/i",
            b"needle\n",
            false,
            false,
            0,
            OutputMode::Content,
        );
        let literal_line = search_sink(
            "/needle/i",
            b"/needle/i\n",
            false,
            false,
            0,
            OutputMode::Content,
        );

        // Assert
        assert!(!ordinary_line.has_match);
        assert!(literal_line.has_match);
    }

    #[test]
    fn multiple_occurrences_on_one_source_line_are_stored_once() {
        // Arrange / Act
        let sink = search_sink(
            "needle",
            b"needle and needle and needle\n",
            false,
            false,
            0,
            OutputMode::Content,
        );

        // Assert
        assert!(sink.has_match);
        assert_eq!(line_numbers(&sink), vec![1]);
        assert_eq!(sink.lines[&1].bytes, b"needle and needle and needle\n");
        assert!(sink.lines[&1].matched);
    }

    #[test]
    fn multiline_matches_mark_each_source_line_the_span_touches() {
        // Arrange / Act
        let sink = search_sink(
            r"start\nend",
            b"before\nstart\nend\nafter\n",
            false,
            true,
            0,
            OutputMode::Content,
        );

        // Assert
        assert_eq!(line_numbers(&sink), vec![2, 3]);
        assert!(sink.lines[&2].matched);
        assert!(sink.lines[&3].matched);
        assert_eq!(sink.lines[&2].bytes, b"start\n");
        assert_eq!(sink.lines[&3].bytes, b"end\n");
    }

    #[test]
    fn multiline_does_not_enable_dot_all_without_an_inline_flag() {
        // Arrange / Act
        let ordinary = search_sink(
            r"start.*end",
            b"start\nend\n",
            false,
            true,
            0,
            OutputMode::Content,
        );
        let dot_all = search_sink(
            r"(?s)start.*end",
            b"start\nend\n",
            false,
            true,
            0,
            OutputMode::Content,
        );

        // Assert
        assert!(!ordinary.has_match);
        assert!(dot_all.has_match);
    }

    #[test]
    fn context_windows_merge_and_matched_lines_win_over_context() {
        // Arrange / Act
        let sink = search_sink(
            "match",
            b"zero\none\nmatch\nmatch\nfour\nfive\n",
            false,
            false,
            1,
            OutputMode::Content,
        );

        // Assert
        assert_eq!(line_numbers(&sink), vec![2, 3, 4, 5]);
        assert!(!sink.lines[&2].matched);
        assert!(sink.lines[&3].matched);
        assert!(sink.lines[&4].matched);
        assert!(!sink.lines[&5].matched);
    }

    #[test]
    fn line_anchors_and_zero_width_matches_have_normal_grep_behavior() {
        // Arrange / Act
        let anchored = search_sink(
            r"^needle$",
            b"not needle\nneedle\nneedle plus\n",
            false,
            false,
            0,
            OutputMode::Content,
        );
        let zero_width = search_sink(
            r"^",
            b"first\nsecond\n",
            false,
            false,
            0,
            OutputMode::Content,
        );

        // Assert
        assert_eq!(line_numbers(&anchored), vec![2]);
        assert_eq!(line_numbers(&zero_width), vec![1, 2]);
    }

    #[test]
    fn source_bytes_preserve_crlf_indentation_and_trailing_whitespace() {
        // Arrange / Act
        let sink = search_sink(
            "match",
            b"  before  \r\n\tmatch  \r\nafter\r\n",
            false,
            false,
            1,
            OutputMode::Content,
        );

        // Assert
        assert_eq!(line_numbers(&sink), vec![1, 2, 3]);
        assert_eq!(sink.lines[&1].bytes, b"  before  \r\n");
        assert_eq!(sink.lines[&2].bytes, b"\tmatch  \r\n");
        assert_eq!(sink.lines[&3].bytes, b"after\r\n");
    }

    #[test]
    fn bom_marked_utf16_is_transcoded_before_matching() {
        // Arrange
        let mut contents = vec![0xFF, 0xFE];
        contents.extend(
            "before\nneedle\nafter\n"
                .encode_utf16()
                .flat_map(u16::to_le_bytes),
        );

        // Act
        let sink = search_sink("needle", &contents, false, false, 0, OutputMode::Content);

        // Assert
        assert_eq!(line_numbers(&sink), vec![2]);
        assert_eq!(sink.lines[&2].bytes, b"needle\n");
    }

    #[test]
    fn malformed_utf8_is_retained_for_the_formatter_to_render_lossily() {
        // Arrange / Act
        let sink = search_sink(
            "needle",
            b"needle \xFF\n",
            false,
            false,
            0,
            OutputMode::Content,
        );

        // Assert
        assert_eq!(line_numbers(&sink), vec![1]);
        assert_eq!(sink.lines[&1].bytes, b"needle \xFF\n");
    }

    #[test]
    fn files_with_matches_stops_after_the_first_match_without_storing_lines() {
        // Arrange / Act
        let sink = search_sink(
            "needle",
            b"needle\nneedle\n",
            false,
            false,
            0,
            OutputMode::FilesWithMatches,
        );

        // Assert
        assert!(sink.has_match);
        assert!(sink.lines.is_empty());
        assert_eq!(sink.stop_reason, Some(StopReason::FirstMatch));
    }

    #[test]
    fn content_sink_stops_when_the_next_distinct_match_proves_the_file_limit() {
        // Arrange
        let matcher = compile_matcher("needle", false, false).unwrap();
        let mut searcher = SearcherBuilder::new().line_number(true).build();
        let mut sink = GrepSink::new(OutputMode::Content, &matcher, 2, CancellationToken::new());

        // Act
        searcher
            .search_slice(&matcher, b"needle\nneedle\nneedle\nneedle\n", &mut sink)
            .unwrap();

        // Assert
        assert!(sink.has_match);
        assert_eq!(sink.matched_lines, 2);
        assert_eq!(line_numbers(&sink), vec![1, 2]);
        assert_eq!(sink.stop_reason, Some(StopReason::MatchLimit));
    }

    #[test]
    fn binary_detection_discards_matches_and_records_its_distinct_stop_reason() {
        // Arrange / Act
        let sink = search_sink(
            "needle",
            b"needle\n\0binary\n",
            false,
            false,
            0,
            OutputMode::Content,
        );

        // Assert
        assert!(!sink.has_match);
        assert!(sink.lines.is_empty());
        assert_eq!(sink.stop_reason, Some(StopReason::Binary));
    }

    #[test]
    fn reader_observes_cancellation_that_occurs_during_a_read() {
        struct CancelOnRead {
            cancel: CancellationToken,
        }

        impl Read for CancelOnRead {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                buffer[0] = b'x';
                self.cancel.cancel();
                Ok(1)
            }
        }

        // Arrange
        let cancel = CancellationToken::new();
        let inner = CancelOnRead {
            cancel: cancel.clone(),
        };
        let mut reader = GrepReader::new(inner, cancel);
        let mut buffer = [0_u8; 8];

        // Act
        let error = reader.read(&mut buffer).unwrap_err();

        // Assert
        assert_eq!(scan_control(&error), Some(ScanControl::Cancelled));
    }

    #[test]
    fn recursive_scan_races_are_skipped_and_counted() {
        // Arrange
        let root = tempdir().unwrap();
        let stale_path = root.path().join("stale.txt");
        let good_path = root.path().join("good.txt");
        fs::write(&stale_path, "needle\n").unwrap();
        fs::write(&good_path, "needle\n").unwrap();
        let stale = candidate(&stale_path, "stale.txt");
        let good = candidate(&good_path, "good.txt");
        fs::remove_file(stale_path).unwrap();

        // Act
        let scan = scan_candidates(
            vec![stale, good],
            scan_options(OutputMode::Content, limits(), false, "."),
        )
        .unwrap();

        // Assert
        assert_eq!(scan.warnings.unsearchable_files, 1);
        assert_eq!(scan.results.len(), 1);
        assert!(matches!(
            &scan.results[0],
            GrepFileResult::Content { path, .. } if path.text == "good.txt"
        ));
    }

    #[test]
    fn explicit_file_scan_races_remain_path_specific_failures() {
        // Arrange
        let root = tempdir().unwrap();
        let stale_path = root.path().join("stale.txt");
        fs::write(&stale_path, "needle\n").unwrap();
        let stale = candidate(&stale_path, "stale.txt");
        fs::remove_file(&stale_path).unwrap();
        let expected_io_error = fs::File::open(stale_path).unwrap_err();

        // Act
        let result = scan_candidates(
            vec![stale],
            scan_options(OutputMode::Content, limits(), true, "stale.txt"),
        );
        let Err(error) = result else {
            panic!("expected an explicit file scan race to fail");
        };

        // Assert
        assert_eq!(
            error,
            ToolExecutionError::ToolError(format!(
                "failed to grep `stale.txt`: {expected_io_error}"
            ))
        );
    }

    #[test]
    fn global_match_limit_stops_before_scanning_lower_priority_candidates() {
        // Arrange
        let root = tempdir().unwrap();
        let newest_path = root.path().join("newest.txt");
        let stale_path = root.path().join("stale.txt");
        fs::write(&newest_path, "needle\n").unwrap();
        fs::write(&stale_path, "needle\n").unwrap();
        let newest = candidate(&newest_path, "newest.txt");
        let stale = candidate(&stale_path, "stale.txt");
        fs::remove_file(stale_path).unwrap();
        let limits = GrepLimits {
            matched_lines: 1,
            ..limits()
        };

        // Act
        let scan = scan_candidates(
            vec![newest, stale],
            scan_options(OutputMode::Content, limits, false, "."),
        )
        .unwrap();

        // Assert
        assert_eq!(scan.results.len(), 1);
        assert!(scan.truncation.global_matches);
        assert_eq!(scan.warnings.unsearchable_files, 0);
    }

    #[test]
    fn path_limit_stops_before_scanning_lower_priority_candidates() {
        // Arrange
        let root = tempdir().unwrap();
        let newest_path = root.path().join("newest.txt");
        let stale_path = root.path().join("stale.txt");
        fs::write(&newest_path, "needle\n").unwrap();
        fs::write(&stale_path, "needle\n").unwrap();
        let newest = candidate(&newest_path, "newest.txt");
        let stale = candidate(&stale_path, "stale.txt");
        fs::remove_file(stale_path).unwrap();
        let limits = GrepLimits {
            paths: 1,
            ..limits()
        };

        // Act
        let scan = scan_candidates(
            vec![newest, stale],
            scan_options(OutputMode::FilesWithMatches, limits, false, "."),
        )
        .unwrap();

        // Assert
        assert_eq!(scan.results.len(), 1);
        assert!(scan.truncation.paths);
        assert_eq!(scan.warnings.unsearchable_files, 0);
    }
}
