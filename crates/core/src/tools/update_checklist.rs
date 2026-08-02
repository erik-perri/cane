use crate::checklist::parse_checklist;
use crate::protocol::ApprovalRequirement;
use crate::tools::{
    PreparedInvocation, Tool, ToolDefinition, ToolExecutionError, ToolExecutionOutput,
};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

pub(crate) struct UpdateChecklistTool;

#[async_trait::async_trait]
impl Tool for UpdateChecklistTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "update_checklist".to_string(),
            description: "Maintain a user-visible checklist for meaningful multi-step work. This \
                call replaces the complete checklist: always include every step that should \
                remain, normally 3-8 meaningful user-facing steps rather than individual file \
                reads, tool calls, or trivial operations. Put completed steps first, then at most \
                one in-progress step, then pending steps. Completed means you claim the step is \
                finished; Cane does not independently verify it. Submit an explicit empty array \
                to clear the checklist. Keep it current when progress or your understanding \
                changes."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "checklist": {
                        "type": "array",
                        "description": "The complete replacement checklist.",
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {
                                    "type": "string",
                                    "description": "A meaningful user-facing work step without embedded numbering or checkmarks.",
                                    "maxLength": 200
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"]
                                }
                            },
                            "required": ["step", "status"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["checklist"],
                "additionalProperties": false
            }),
        }
    }

    async fn prepare(&self, input: Value) -> Result<Box<dyn PreparedInvocation>, String> {
        let checklist = parse_checklist(&input)?;
        Ok(Box::new(PreparedChecklistUpdate { checklist }))
    }
}

struct PreparedChecklistUpdate {
    checklist: crate::Checklist,
}

#[async_trait::async_trait]
impl PreparedInvocation for PreparedChecklistUpdate {
    fn approval_requirement(&self) -> ApprovalRequirement {
        ApprovalRequirement::None
    }

    async fn execute(
        self: Box<Self>,
        _cancel: CancellationToken,
    ) -> Result<ToolExecutionOutput, ToolExecutionError> {
        Ok(ToolExecutionOutput::checklist_update(self.checklist))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChecklistStepStatus;
    use serde_json::json;

    #[test]
    fn definition_describes_the_strict_full_replacement_contract() {
        let definition = UpdateChecklistTool.definition();

        assert_eq!(definition.name, "update_checklist");
        assert!(
            definition
                .description
                .contains("replaces the complete checklist")
        );
        assert_eq!(definition.input_schema["type"], "object");
        assert_eq!(definition.input_schema["required"], json!(["checklist"]));
        assert_eq!(definition.input_schema["additionalProperties"], false);
        assert_eq!(
            definition.input_schema["properties"]["checklist"]["maxItems"],
            20
        );
        let item = &definition.input_schema["properties"]["checklist"]["items"];
        assert_eq!(item["required"], json!(["step", "status"]));
        assert_eq!(item["additionalProperties"], false);
        assert_eq!(item["properties"]["step"]["maxLength"], 200);
        assert_eq!(
            item["properties"]["status"]["enum"],
            json!(["pending", "in_progress", "completed"])
        );
    }

    #[tokio::test]
    async fn valid_input_prepares_without_approval_and_returns_a_checklist_update() {
        let invocation = UpdateChecklistTool
            .prepare(json!({
                "checklist": [{ "step": " Implement it ", "status": "in_progress" }]
            }))
            .await
            .unwrap();
        assert_eq!(invocation.approval_requirement(), ApprovalRequirement::None);

        let output = invocation.execute(CancellationToken::new()).await.unwrap();
        let (content, execution, checklist) = output.into_parts();

        assert_eq!(content, "Checklist updated.");
        assert_eq!(execution, None);
        let checklist = checklist.unwrap();
        assert_eq!(checklist.steps()[0].text(), "Implement it");
        assert_eq!(
            checklist.steps()[0].status(),
            ChecklistStepStatus::InProgress
        );
    }

    #[tokio::test]
    async fn invalid_input_is_rejected_during_preparation() {
        let error = UpdateChecklistTool
            .prepare(json!({
                "checklist": [
                    { "step": "Later", "status": "pending" },
                    { "step": "Earlier", "status": "completed" }
                ]
            }))
            .await
            .err()
            .unwrap();

        assert_eq!(
            error,
            "checklist step 2 has status completed after unfinished steps"
        );
    }

    #[tokio::test]
    async fn explicit_empty_array_produces_an_empty_checklist() {
        let invocation = UpdateChecklistTool
            .prepare(json!({ "checklist": [] }))
            .await
            .unwrap();

        let (_, _, checklist) = invocation
            .execute(CancellationToken::new())
            .await
            .unwrap()
            .into_parts();

        assert!(checklist.unwrap().is_empty());
    }
}
