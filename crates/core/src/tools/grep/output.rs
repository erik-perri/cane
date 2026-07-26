use super::scan::{GrepFileResult, GrepLine, GrepTruncation, GrepWarnings};
use super::{
    BINARY_FILE_SKIPPED, ESCAPED_PATH_WARNING, EXCERPT_WARNING, GrepLimits, LOSSY_PATH_WARNING,
    LOSSY_SOURCE_WARNING, NO_MATCHES, OMITTED_SOURCE_MARKER, OutputMode, RECURSIVE_INCLUDE_HINT,
};
use crate::tools::path_display::RenderedPath;
use std::borrow::Cow;
use std::ops::Range;

pub(super) fn format_results(
    results: Vec<GrepFileResult>,
    output_mode: OutputMode,
    limits: GrepLimits,
    warnings: GrepWarnings,
    truncation: GrepTruncation,
) -> String {
    if results
        .iter()
        .any(|result| matches!(result, GrepFileResult::Binary))
    {
        return bounded_result(BINARY_FILE_SKIPPED, limits.output_bytes);
    }

    match output_mode {
        OutputMode::Content => {
            format_content_results(results, limits, warnings, truncation.global_matches)
        }
        OutputMode::FilesWithMatches => {
            format_path_results(results, limits, warnings, truncation.paths)
        }
    }
}

pub(super) fn append_recursive_include_hint(mut output: String, max_bytes: usize) -> String {
    let additional_bytes = 1usize.saturating_add(RECURSIVE_INCLUDE_HINT.len());
    if output.len().saturating_add(additional_bytes) <= max_bytes {
        output.push('\n');
        output.push_str(RECURSIVE_INCLUDE_HINT);
    }
    output
}

#[derive(Debug)]
struct ContentDocument {
    blocks: Vec<ContentBlock>,
    global_match_truncated: bool,
    had_matches: bool,
    warnings: GrepWarnings,
}

#[derive(Debug)]
struct ContentBlock {
    lines: Vec<RenderedSourceLine>,
    match_truncated: bool,
    path: String,
    path_json_escaped: bool,
    path_lossy: bool,
}

#[derive(Debug)]
struct RenderedSourceLine {
    excerpted: bool,
    lossy: bool,
    matched: bool,
    number: u64,
    text: String,
}

fn format_content_results(
    results: Vec<GrepFileResult>,
    limits: GrepLimits,
    warnings: GrepWarnings,
    scan_truncated: bool,
) -> String {
    let mut document = build_content_document(results, limits, warnings, scan_truncated);
    let full_output = render_content_document(&document, false, limits.output_bytes);

    if full_output.len() <= limits.output_bytes {
        return full_output;
    }

    loop {
        let output = render_content_document(&document, true, limits.output_bytes);
        if output.len() <= limits.output_bytes {
            return output;
        }

        let Some(block) = document.blocks.last_mut() else {
            return bounded_truncation_notice(limits.output_bytes);
        };

        block.lines.pop();
        if !block.lines.iter().any(|line| line.matched) {
            document.blocks.pop();
        }
    }
}

fn build_content_document(
    results: Vec<GrepFileResult>,
    limits: GrepLimits,
    warnings: GrepWarnings,
    scan_truncated: bool,
) -> ContentDocument {
    let had_matches = results
        .iter()
        .any(|result| matches!(result, GrepFileResult::Content { .. }));
    let total_matched_lines = results
        .iter()
        .filter_map(|result| match result {
            GrepFileResult::Content { lines, .. } => Some(lines),
            GrepFileResult::Binary | GrepFileResult::Path(_) => None,
        })
        .flat_map(|lines| lines.values())
        .filter(|line| line.matched)
        .count();
    let global_match_truncated = scan_truncated || total_matched_lines > limits.matched_lines;
    let mut remaining_global_matches = limits.matched_lines;
    let mut blocks = Vec::new();

    for result in results {
        let GrepFileResult::Content {
            lines: source_lines,
            match_truncated,
            path,
        } = result
        else {
            continue;
        };

        let file_matched_lines = source_lines.values().filter(|line| line.matched).count();
        let file_match_limit = limits.matched_lines_per_file.min(remaining_global_matches);
        let mut accepted_matches = 0;
        let mut lines = Vec::new();

        for (number, line) in source_lines {
            if line.matched {
                if accepted_matches == file_match_limit {
                    break;
                }
                accepted_matches += 1;
                remaining_global_matches = remaining_global_matches.saturating_sub(1);
            }

            lines.push(render_source_line(number, line, limits.source_line_bytes));
        }

        if lines.iter().any(|line| line.matched) {
            blocks.push(ContentBlock {
                lines,
                match_truncated: match_truncated
                    || (file_matched_lines > limits.matched_lines_per_file
                        && accepted_matches == limits.matched_lines_per_file),
                path: path.text,
                path_json_escaped: path.json_escaped,
                path_lossy: path.lossy,
            });
        }

        if remaining_global_matches == 0 {
            break;
        }
    }

    ContentDocument {
        blocks,
        global_match_truncated,
        had_matches,
        warnings,
    }
}

fn render_content_document(
    document: &ContentDocument,
    byte_truncated: bool,
    max_bytes: usize,
) -> String {
    let mut sections = Vec::new();

    for block in &document.blocks {
        let mut section = format!("{}:", block.path);
        let mut previous_number = None;

        for line in &block.lines {
            if previous_number.is_some_and(|previous| line.number != previous + 1) {
                section.push_str("\n  --");
            }

            let marker = if line.matched { ':' } else { '-' };
            section.push_str(&format!("\n  {}{marker}", line.number));
            if !line.text.is_empty() {
                section.push(' ');
                section.push_str(&line.text);
            }
            previous_number = Some(line.number);
        }

        if block.match_truncated {
            section.push_str(&format!(
                "\n[truncated: showing the first {} matched lines in `{}`; more matches may \
                 exist; narrow path, include, or pattern and search again]",
                block.lines.iter().filter(|line| line.matched).count(),
                block.path
            ));
        }

        sections.push(section);
    }

    if sections.is_empty() && !document.had_matches {
        sections.push(NO_MATCHES.to_string());
    }

    let included_lines = document
        .blocks
        .iter()
        .flat_map(|block| &block.lines)
        .collect::<Vec<_>>();

    if document.blocks.iter().any(|block| block.path_lossy) {
        sections.push(LOSSY_PATH_WARNING.to_string());
    }
    if document.blocks.iter().any(|block| block.path_json_escaped) {
        sections.push(ESCAPED_PATH_WARNING.to_string());
    }
    if included_lines.iter().any(|line| line.lossy) {
        sections.push(LOSSY_SOURCE_WARNING.to_string());
    }
    if included_lines.iter().any(|line| line.excerpted) {
        sections.push(EXCERPT_WARNING.to_string());
    }
    append_scan_warnings(&mut sections, document.warnings);
    if document.global_match_truncated {
        sections.push(format!(
            "[truncated: showing at most {} matched lines; more matches may exist; narrow path, \
             include, or pattern and search again]",
            document
                .blocks
                .iter()
                .flat_map(|block| &block.lines)
                .filter(|line| line.matched)
                .count()
        ));
    }
    if byte_truncated {
        sections.push(output_truncation_notice(max_bytes));
    }

    sections.join("\n\n")
}

fn format_path_results(
    results: Vec<GrepFileResult>,
    limits: GrepLimits,
    warnings: GrepWarnings,
    scan_truncated: bool,
) -> String {
    let matching_paths = results
        .into_iter()
        .filter_map(|result| match result {
            GrepFileResult::Path(path) => Some(path),
            GrepFileResult::Binary | GrepFileResult::Content { .. } => None,
        })
        .collect::<Vec<_>>();

    let match_truncated = scan_truncated || matching_paths.len() > limits.paths;
    let mut shown_paths = matching_paths.len().min(limits.paths);
    let full_output = render_path_document(
        &matching_paths[..shown_paths],
        match_truncated,
        false,
        limits.output_bytes,
        warnings,
    );

    if full_output.len() <= limits.output_bytes {
        return full_output;
    }

    loop {
        let output = render_path_document(
            &matching_paths[..shown_paths],
            match_truncated,
            true,
            limits.output_bytes,
            warnings,
        );
        if output.len() <= limits.output_bytes {
            return output;
        }
        if shown_paths == 0 {
            return bounded_truncation_notice(limits.output_bytes);
        }
        shown_paths -= 1;
    }
}

fn render_path_document(
    paths: &[RenderedPath],
    match_truncated: bool,
    byte_truncated: bool,
    max_bytes: usize,
    warnings: GrepWarnings,
) -> String {
    let mut lines = paths
        .iter()
        .map(|path| path.text.clone())
        .collect::<Vec<_>>();

    if paths.is_empty() && !match_truncated {
        lines.push(NO_MATCHES.to_string());
    }
    if paths.iter().any(|path| path.lossy) {
        lines.push(LOSSY_PATH_WARNING.to_string());
    }
    if paths.iter().any(|path| path.json_escaped) {
        lines.push(ESCAPED_PATH_WARNING.to_string());
    }
    append_scan_warnings(&mut lines, warnings);
    if match_truncated {
        let noun = if paths.len() == 1 { "path" } else { "paths" };
        lines.push(format!(
            "[truncated: showing {} most recently modified matching {noun}; more matches may \
             exist; narrow path, include, or pattern and search again]",
            paths.len()
        ));
    }
    if byte_truncated {
        lines.push(output_truncation_notice(max_bytes));
    }

    lines.join("\n")
}

fn append_scan_warnings(output: &mut Vec<String>, warnings: GrepWarnings) {
    if warnings.oversized_files > 0 {
        output.push(oversized_files_warning(warnings.oversized_files));
    }
    if warnings.unsearchable_files > 0 {
        output.push(unsearchable_files_warning(warnings.unsearchable_files));
    }
}

fn oversized_files_warning(count: usize) -> String {
    let noun = if count == 1 { "file" } else { "files" };
    format!("[warning: skipped {count} oversized {noun}]")
}

fn unsearchable_files_warning(count: usize) -> String {
    let noun = if count == 1 { "file" } else { "files" };
    format!("[warning: skipped {count} unsearchable {noun}]")
}

pub(super) fn output_truncation_notice(max_bytes: usize) -> String {
    format!(
        "[truncated: output limited to {max_bytes} bytes; additional results omitted; narrow path, \
         include, or pattern and search again]"
    )
}

fn bounded_truncation_notice(max_bytes: usize) -> String {
    let notice = output_truncation_notice(max_bytes);
    if notice.len() <= max_bytes {
        notice
    } else if "[truncated]".len() <= max_bytes {
        "[truncated]".to_string()
    } else {
        String::new()
    }
}

fn bounded_result(result: &str, max_bytes: usize) -> String {
    if result.len() <= max_bytes {
        result.to_string()
    } else {
        bounded_truncation_notice(max_bytes)
    }
}

fn render_source_line(number: u64, line: GrepLine, max_source_bytes: usize) -> RenderedSourceLine {
    let source_bytes = strip_line_terminator(&line.bytes);
    let decoded = String::from_utf8_lossy(source_bytes);
    let lossy = matches!(decoded, Cow::Owned(_));
    let (text, excerpted) = excerpt_source(
        &decoded,
        line.matched.then_some(line.first_match).flatten(),
        source_bytes,
        max_source_bytes,
    );

    RenderedSourceLine {
        excerpted,
        lossy,
        matched: line.matched,
        number,
        text,
    }
}

fn strip_line_terminator(mut bytes: &[u8]) -> &[u8] {
    if bytes.ends_with(b"\n") {
        bytes = &bytes[..bytes.len() - 1];
        if bytes.ends_with(b"\r") {
            bytes = &bytes[..bytes.len() - 1];
        }
    }
    bytes
}

fn excerpt_source(
    decoded: &str,
    first_match: Option<Range<usize>>,
    source_bytes: &[u8],
    max_source_bytes: usize,
) -> (String, bool) {
    if decoded.len() <= max_source_bytes {
        return (decoded.to_string(), false);
    }

    let (start, end) = match first_match {
        None => (0, floor_char_boundary(decoded, max_source_bytes)),
        Some(first_match) => {
            let raw_start = first_match.start.min(source_bytes.len());
            let raw_end = first_match.end.min(source_bytes.len());
            let decoded_start = String::from_utf8_lossy(&source_bytes[..raw_start]).len();
            let decoded_end = String::from_utf8_lossy(&source_bytes[..raw_end]).len();
            excerpt_window(decoded, decoded_start..decoded_end, max_source_bytes)
        }
    };

    let mut excerpt = String::new();
    if start > 0 {
        excerpt.push_str(OMITTED_SOURCE_MARKER);
    }
    excerpt.push_str(&decoded[start..end]);
    if end < decoded.len() {
        excerpt.push_str(OMITTED_SOURCE_MARKER);
    }

    (excerpt, true)
}

fn excerpt_window(text: &str, anchor: Range<usize>, max_bytes: usize) -> (usize, usize) {
    if max_bytes == 0 {
        return (0, 0);
    }

    let anchor_start = floor_char_boundary(text, anchor.start.min(text.len()));
    let anchor_end = floor_char_boundary(text, anchor.end.min(text.len()));
    let anchor_len = anchor_end.saturating_sub(anchor_start).min(max_bytes);
    let leading_budget = (max_bytes - anchor_len) / 2;
    let mut start = floor_char_boundary(text, anchor_start.saturating_sub(leading_budget));
    let mut end = floor_char_boundary(text, (start + max_bytes).min(text.len()));

    if end == text.len() && end - start < max_bytes {
        start = floor_char_boundary(text, end.saturating_sub(max_bytes));
        end = floor_char_boundary(text, (start + max_bytes).min(text.len()));
    }

    (start, end)
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::grep::scan::{GrepFileResult, GrepLine, GrepTruncation, GrepWarnings};
    use std::collections::BTreeMap;

    fn path(text: &str) -> RenderedPath {
        RenderedPath {
            text: text.to_string(),
            lossy: false,
            json_escaped: false,
        }
    }

    fn line(bytes: impl Into<Vec<u8>>, matched: bool) -> GrepLine {
        GrepLine {
            bytes: bytes.into(),
            first_match: matched.then_some(0..1),
            matched,
        }
    }

    fn content_result(path: RenderedPath, lines: BTreeMap<u64, GrepLine>) -> GrepFileResult {
        GrepFileResult::Content {
            lines,
            match_truncated: false,
            path,
        }
    }

    fn generous_limits() -> GrepLimits {
        GrepLimits {
            matched_lines: usize::MAX,
            matched_lines_per_file: usize::MAX,
            output_bytes: usize::MAX,
            paths: usize::MAX,
            source_line_bytes: usize::MAX,
        }
    }

    #[test]
    fn recursive_include_hint_is_appended_only_when_it_fits_the_output_limit() {
        // Arrange
        let output = NO_MATCHES.to_string();
        let hinted_len = output.len() + 1 + RECURSIVE_INCLUDE_HINT.len();

        // Act
        let exact = append_recursive_include_hint(output.clone(), hinted_len);
        let too_small = append_recursive_include_hint(output.clone(), hinted_len - 1);

        // Assert
        assert_eq!(exact, format!("{NO_MATCHES}\n{RECURSIVE_INCLUDE_HINT}"));
        assert_eq!(too_small, output);
    }

    #[test]
    fn content_caps_do_not_count_context_lines() {
        // Arrange
        let limits = GrepLimits {
            matched_lines: 10,
            matched_lines_per_file: 1,
            ..generous_limits()
        };
        let per_file_result = content_result(
            path("main.rs"),
            BTreeMap::from([
                (1, line(b"before\n", false)),
                (2, line(b"first\n", true)),
                (3, line(b"after\n", false)),
                (4, line(b"second\n", true)),
            ]),
        );

        // Act
        let per_file_output = format_results(
            vec![per_file_result],
            OutputMode::Content,
            limits,
            GrepWarnings::default(),
            GrepTruncation::default(),
        );
        let global_output = format_results(
            vec![
                content_result(path("new.rs"), BTreeMap::from([(1, line(b"new\n", true))])),
                content_result(path("old.rs"), BTreeMap::from([(1, line(b"old\n", true))])),
            ],
            OutputMode::Content,
            GrepLimits {
                matched_lines: 1,
                ..generous_limits()
            },
            GrepWarnings::default(),
            GrepTruncation::default(),
        );

        // Assert
        assert!(per_file_output.contains("  1- before"));
        assert!(per_file_output.contains("  2: first"));
        assert!(per_file_output.contains("  3- after"));
        assert!(!per_file_output.contains("second"));
        assert!(per_file_output.contains("first 1 matched lines"));
        assert!(global_output.contains("new.rs:"));
        assert!(!global_output.contains("old.rs:"));
        assert!(global_output.contains("showing at most 1 matched lines"));
    }

    #[test]
    fn recursive_scan_warnings_are_rendered_once() {
        // Arrange
        let warnings = GrepWarnings {
            oversized_files: 0,
            unsearchable_files: 1,
        };

        // Act
        let output = format_results(
            Vec::new(),
            OutputMode::Content,
            generous_limits(),
            warnings,
            GrepTruncation::default(),
        );

        // Assert
        assert_eq!(
            output,
            "No matches found.\n\n[warning: skipped 1 unsearchable file]"
        );
    }

    #[test]
    fn formatter_accounts_for_utf8_bytes_at_exact_boundaries() {
        // Arrange
        let limits = generous_limits();
        let make_result = || {
            content_result(
                path("main.rs"),
                (1..=40)
                    .map(|number| {
                        (
                            number,
                            GrepLine {
                                bytes: format!("é line {number}\n").into_bytes(),
                                first_match: Some(0..2),
                                matched: true,
                            },
                        )
                    })
                    .collect(),
            )
        };
        let full = format_results(
            vec![make_result()],
            OutputMode::Content,
            limits,
            GrepWarnings::default(),
            GrepTruncation::default(),
        );

        // Act
        let exact = format_results(
            vec![make_result()],
            OutputMode::Content,
            GrepLimits {
                output_bytes: full.len(),
                ..limits
            },
            GrepWarnings::default(),
            GrepTruncation::default(),
        );
        let one_byte_over_limit = full.len() - 1;
        let bounded = format_results(
            vec![make_result()],
            OutputMode::Content,
            GrepLimits {
                output_bytes: one_byte_over_limit,
                ..limits
            },
            GrepWarnings::default(),
            GrepTruncation::default(),
        );
        let excerpted = format_results(
            vec![content_result(
                path("unicode.rs"),
                BTreeMap::from([(
                    1,
                    GrepLine {
                        bytes: "aaaaéébbbb\n".as_bytes().to_vec(),
                        first_match: Some(4..6),
                        matched: true,
                    },
                )]),
            )],
            OutputMode::Content,
            GrepLimits {
                source_line_bytes: 3,
                ..limits
            },
            GrepWarnings::default(),
            GrepTruncation::default(),
        );

        // Assert
        assert_eq!(exact, full);
        assert!(bounded.len() <= one_byte_over_limit);
        assert!(bounded.ends_with(&output_truncation_notice(one_byte_over_limit)));
        assert!(
            bounded
                .lines()
                .filter(|line| line.starts_with("  ") && *line != "  --")
                .all(|line| line.contains("é line "))
        );
        assert!(excerpted.contains("  1: [...]é[...]"));
        assert!(excerpted.ends_with(EXCERPT_WARNING));
    }
}
