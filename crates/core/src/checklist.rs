use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;

const MAX_CHECKLIST_STEPS: usize = 20;
const MAX_STEP_CHARS: usize = 200;
const REASSESSMENT_REMINDER: &str = "Reassess this checklist against the latest user request, and replace or clear it if it is no longer appropriate.";

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Checklist {
    steps: Vec<ChecklistStep>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChecklistStep {
    text: String,
    status: ChecklistStepStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChecklistStepStatus {
    Pending,
    InProgress,
    Completed,
}

impl Checklist {
    pub fn steps(&self) -> &[ChecklistStep] {
        &self.steps
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub(crate) fn is_fully_completed(&self) -> bool {
        !self.steps.is_empty()
            && self
                .steps
                .iter()
                .all(|step| step.status == ChecklistStepStatus::Completed)
    }

    pub(crate) fn render_dynamic_context(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }

        let mut rendered = String::from("<current_checklist>\n");
        for step in &self.steps {
            rendered.push('[');
            rendered.push_str(step.status.context_label());
            rendered.push_str("] ");
            push_escaped_markup(&mut rendered, &step.text);
            rendered.push('\n');
        }
        rendered.push_str("</current_checklist>");

        if !self.is_fully_completed() {
            rendered.push_str("\n\n");
            rendered.push_str(REASSESSMENT_REMINDER);
        }

        Some(rendered)
    }
}

impl ChecklistStep {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn status(&self) -> ChecklistStepStatus {
        self.status
    }
}

impl ChecklistStepStatus {
    fn context_label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChecklistInput {
    checklist: Vec<ChecklistStepInput>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChecklistStepInput {
    step: String,
    status: ChecklistStepStatusInput,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ChecklistStepStatusInput {
    Pending,
    InProgress,
    Completed,
}

impl From<ChecklistStepStatusInput> for ChecklistStepStatus {
    fn from(status: ChecklistStepStatusInput) -> Self {
        match status {
            ChecklistStepStatusInput::Pending => Self::Pending,
            ChecklistStepStatusInput::InProgress => Self::InProgress,
            ChecklistStepStatusInput::Completed => Self::Completed,
        }
    }
}

/// Parse the checklist format accepted from the model.
///
/// This format is also persisted in successful historical tool calls, so the accepted shape is a
/// compatibility contract. Do not change it incompatibly in place. If a second incompatible
/// format becomes necessary, retain this parser for legacy input and add explicit dispatch.
pub(crate) fn parse_checklist(input: &Value) -> Result<Checklist, String> {
    let input: ChecklistInput = serde_json::from_value(input.clone())
        .map_err(|error| format!("invalid checklist input: {error}"))?;

    if input.checklist.len() > MAX_CHECKLIST_STEPS {
        return Err(format!(
            "checklist must contain at most {MAX_CHECKLIST_STEPS} steps"
        ));
    }

    for (index, step) in input.checklist.iter().enumerate() {
        if step.step.trim().is_empty() {
            return Err(format!("checklist step {} must not be empty", index + 1));
        }
    }

    for (index, step) in input.checklist.iter().enumerate() {
        if step.step.chars().any(char::is_control) {
            return Err(format!(
                "checklist step {} must not contain control characters",
                index + 1
            ));
        }
        if step.step.trim().chars().count() > MAX_STEP_CHARS {
            return Err(format!(
                "checklist step {} must contain at most {MAX_STEP_CHARS} characters",
                index + 1
            ));
        }
    }

    let steps: Vec<ChecklistStep> = input
        .checklist
        .into_iter()
        .map(|step| ChecklistStep {
            text: step.step.trim().to_string(),
            status: step.status.into(),
        })
        .collect();

    let mut unique_steps = HashSet::with_capacity(steps.len());
    for (index, step) in steps.iter().enumerate() {
        if !unique_steps.insert(step.text.as_str()) {
            return Err(format!(
                "checklist step {} duplicates an earlier step",
                index + 1
            ));
        }
    }

    validate_status_order(&steps)?;

    Ok(Checklist { steps })
}

fn validate_status_order(steps: &[ChecklistStep]) -> Result<(), String> {
    let mut saw_in_progress = false;
    let mut saw_pending = false;

    for (index, step) in steps.iter().enumerate() {
        match step.status {
            ChecklistStepStatus::Completed if saw_in_progress || saw_pending => {
                return Err(format!(
                    "checklist step {} has status completed after unfinished steps",
                    index + 1
                ));
            }
            ChecklistStepStatus::Completed => {}
            ChecklistStepStatus::InProgress if saw_pending => {
                return Err(format!(
                    "checklist step {} has status in_progress after pending steps",
                    index + 1
                ));
            }
            ChecklistStepStatus::InProgress if saw_in_progress => {
                return Err("checklist must contain at most one in_progress step".to_string());
            }
            ChecklistStepStatus::InProgress => saw_in_progress = true,
            ChecklistStepStatus::Pending => saw_pending = true,
        }
    }

    Ok(())
}

fn push_escaped_markup(rendered: &mut String, text: &str) {
    for character in text.chars() {
        match character {
            '&' => rendered.push_str("&amp;"),
            '<' => rendered.push_str("&lt;"),
            '>' => rendered.push_str("&gt;"),
            _ => rendered.push(character),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(steps: Value) -> Result<Checklist, String> {
        parse_checklist(&json!({ "checklist": steps }))
    }

    fn step(text: impl Into<String>, status: &str) -> Value {
        json!({ "step": text.into(), "status": status })
    }

    fn parsed(steps: Value) -> Checklist {
        parse(steps).unwrap()
    }

    #[test]
    fn valid_empty_and_common_status_arrangements_are_accepted() {
        let cases = [
            json!([]),
            json!([step("Pending", "pending")]),
            json!([step("Completed", "completed")]),
            json!([
                step("Completed", "completed"),
                step("Current", "in_progress"),
                step("Later", "pending")
            ]),
            json!([step("Completed", "completed"), step("Later", "pending")]),
            json!([step("First", "pending"), step("Second", "pending")]),
            json!([
                step("Already done", "completed"),
                step("Also done", "completed")
            ]),
        ];

        for case in cases {
            assert!(parse(case).is_ok());
        }
    }

    #[test]
    fn exact_limits_and_normal_unicode_are_accepted() {
        let twenty_steps: Vec<_> = (1..=20)
            .map(|index| step(format!("Step {index}"), "pending"))
            .collect();
        let two_hundred_scalars = "🦀".repeat(200);

        assert_eq!(parsed(json!(twenty_steps)).steps().len(), 20);
        assert_eq!(
            parsed(json!([step(&two_hundred_scalars, "pending")])).steps()[0]
                .text()
                .chars()
                .count(),
            200
        );
        assert!(parse(json!([step("Crème brûlée 東京 🚀", "pending")])).is_ok());
        assert!(parse(json!([step("Compare <old> & >new<", "pending")])).is_ok());
    }

    #[test]
    fn surrounding_whitespace_is_trimmed_without_other_normalization() {
        let checklist = parsed(json!([step("  Keep  INTERNAL spacing 🚀  ", "pending")]));

        assert_eq!(checklist.steps()[0].text(), "Keep  INTERNAL spacing 🚀");
    }

    #[test]
    fn structural_input_must_be_exact() {
        let invalid = [
            json!({}),
            json!({ "checklist": [], "extra": true }),
            json!({ "checklist": [{ "step": "A", "status": "pending", "extra": true }] }),
            json!({ "checklist": [{ "step": "A", "status": "blocked" }] }),
            json!({ "checklist": "not an array" }),
        ];

        for input in invalid {
            assert!(parse_checklist(&input).is_err());
        }
    }

    #[test]
    fn more_than_twenty_steps_is_rejected() {
        let steps: Vec<_> = (1..=21)
            .map(|index| step(format!("Step {index}"), "pending"))
            .collect();

        assert_eq!(
            parse(json!(steps)).unwrap_err(),
            "checklist must contain at most 20 steps"
        );
    }

    #[test]
    fn empty_and_whitespace_only_steps_are_rejected() {
        assert_eq!(
            parse(json!([step("", "pending")])).unwrap_err(),
            "checklist step 1 must not be empty"
        );
        assert_eq!(
            parse(json!([step(" \u{2003} ", "pending")])).unwrap_err(),
            "checklist step 1 must not be empty"
        );
    }

    #[test]
    fn control_characters_are_rejected() {
        for control in ['\n', '\r', '\t', '\u{1b}', '\u{85}'] {
            for text in [
                format!("before{control}after"),
                format!("{control}before"),
                format!("after{control}"),
            ] {
                assert_eq!(
                    parse(json!([step(text, "pending")])).unwrap_err(),
                    "checklist step 1 must not contain control characters"
                );
            }
        }
    }

    #[test]
    fn more_than_two_hundred_unicode_scalars_is_rejected() {
        assert_eq!(
            parse(json!([step("🚀".repeat(201), "pending")])).unwrap_err(),
            "checklist step 1 must contain at most 200 characters"
        );
    }

    #[test]
    fn exact_duplicates_are_rejected_after_trimming_but_case_is_significant() {
        assert_eq!(
            parse(json!([step("Same", "completed"), step("Same", "pending")])).unwrap_err(),
            "checklist step 2 duplicates an earlier step"
        );
        assert_eq!(
            parse(json!([
                step(" Same ", "completed"),
                step("Same", "pending")
            ]))
            .unwrap_err(),
            "checklist step 2 duplicates an earlier step"
        );
        assert!(parse(json!([step("Same", "completed"), step("same", "pending")])).is_ok());
    }

    #[test]
    fn multiple_in_progress_steps_are_rejected() {
        assert_eq!(
            parse(json!([
                step("One", "in_progress"),
                step("Two", "in_progress")
            ]))
            .unwrap_err(),
            "checklist must contain at most one in_progress step"
        );
    }

    #[test]
    fn statuses_must_follow_completed_current_pending_order() {
        let cases = [
            (
                json!([step("Pending", "pending"), step("Completed", "completed")]),
                "checklist step 2 has status completed after unfinished steps",
            ),
            (
                json!([
                    step("Current", "in_progress"),
                    step("Completed", "completed")
                ]),
                "checklist step 2 has status completed after unfinished steps",
            ),
            (
                json!([step("Pending", "pending"), step("Current", "in_progress")]),
                "checklist step 2 has status in_progress after pending steps",
            ),
        ];

        for (steps, expected) in cases {
            assert_eq!(parse(steps).unwrap_err(), expected);
        }
    }

    #[test]
    fn semantic_validation_order_is_stable() {
        let mut oversized_before_other_failures: Vec<_> = (1..=21)
            .map(|index| step(format!("Step {index}"), "pending"))
            .collect();
        oversized_before_other_failures[0] = step("", "completed");
        let empty_before_other_failures = json!([
            step("Pending", "pending"),
            step("   ", "completed"),
            step("bad\ncontrol", "in_progress")
        ]);
        let control_before_duplicate_and_status = json!([
            step("Same", "pending"),
            step("Same", "completed"),
            step("bad\ncontrol", "pending")
        ]);

        assert_eq!(
            parse(json!(oversized_before_other_failures)).unwrap_err(),
            "checklist must contain at most 20 steps"
        );
        assert_eq!(
            parse(empty_before_other_failures).unwrap_err(),
            "checklist step 2 must not be empty"
        );
        assert_eq!(
            parse(control_before_duplicate_and_status).unwrap_err(),
            "checklist step 3 must not contain control characters"
        );
    }

    #[test]
    fn fully_completed_requires_at_least_one_step_and_no_unfinished_steps() {
        assert!(!Checklist::default().is_fully_completed());
        assert!(
            parsed(json!([step("One", "completed"), step("Two", "completed")]))
                .is_fully_completed()
        );
        assert!(!parsed(json!([step("One", "pending")])).is_fully_completed());
        assert!(!parsed(json!([step("One", "in_progress")])).is_fully_completed());
    }

    #[test]
    fn empty_checklist_has_no_dynamic_context() {
        assert_eq!(Checklist::default().render_dynamic_context(), None);
    }

    #[test]
    fn unfinished_dynamic_context_renders_every_status_and_escapes_markup() {
        let checklist = parsed(json!([
            step("Inspect <code> & tests", "completed"),
            step("Implement > parser", "in_progress"),
            step("Verify", "pending")
        ]));

        assert_eq!(
            checklist.render_dynamic_context().unwrap(),
            concat!(
                "<current_checklist>\n",
                "[completed] Inspect &lt;code&gt; &amp; tests\n",
                "[in_progress] Implement &gt; parser\n",
                "[pending] Verify\n",
                "</current_checklist>\n\n",
                "Reassess this checklist against the latest user request, and replace or clear it if it is no longer appropriate."
            )
        );
    }

    #[test]
    fn fully_completed_dynamic_context_keeps_steps_without_a_reminder() {
        let checklist = parsed(json!([
            step("Inspect", "completed"),
            step("Implement", "completed")
        ]));

        assert_eq!(
            checklist.render_dynamic_context().unwrap(),
            concat!(
                "<current_checklist>\n",
                "[completed] Inspect\n",
                "[completed] Implement\n",
                "</current_checklist>"
            )
        );
    }
}
