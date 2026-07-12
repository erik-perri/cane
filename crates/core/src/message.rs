use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolResultData {
    pub tool_use_id: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    pub content: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw_input: Option<String>,
    },
    ToolResult(ToolResultData),
}

impl From<ToolResultData> for ContentBlock {
    fn from(result: ToolResultData) -> Self {
        ContentBlock::ToolResult(result)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ContentBlock::{Text, ToolResult, ToolUse};
    use serde_json::json;

    #[test]
    fn message_serializes_to_expected_json_and_round_trips() {
        // Arrange
        let message = Message {
            role: Role::Assistant,
            content: vec![
                Text {
                    text: "Mock Text".to_string(),
                },
                ToolUse {
                    id: "Mock ID".to_string(),
                    name: "Mock Name".to_string(),
                    input: serde_json::Value::String("Mock Input".to_string()),
                    raw_input: None,
                },
                ToolResult(ToolResultData {
                    tool_use_id: "Mock ID".to_string(),
                    content: "Mock Content".to_string(),
                    is_error: true,
                }),
            ],
        };

        // Act
        let serialized = serde_json::to_string(&message).unwrap();
        let unserialized: Message = serde_json::from_str(&serialized).unwrap();

        // Assert
        assert_eq!(message, unserialized);
        assert_eq!(
            serde_json::to_value(&message).unwrap(),
            json!({
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "Mock Text" },
                    { "type": "tool_use", "id": "Mock ID", "name": "Mock Name", "input": "Mock Input" },
                    { "type": "tool_result", "tool_use_id": "Mock ID", "content": "Mock Content", "is_error": true }
                ]
            })
        );
    }

    #[test]
    fn message_excludes_is_error_when_false() {
        // Arrange
        let message = Message {
            role: Role::Assistant,
            content: vec![ToolResult(ToolResultData {
                tool_use_id: "Mock ID".to_string(),
                content: "Mock Content".to_string(),
                is_error: false,
            })],
        };

        // Act
        let shape = serde_json::to_value(&message).unwrap();

        // Assert
        assert_eq!(
            shape,
            json!({
                "role": "assistant",
                "content": [
                    { "type": "tool_result", "tool_use_id": "Mock ID", "content": "Mock Content" }
                ]
            })
        );
    }
}
