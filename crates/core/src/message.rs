use serde::{Deserialize, Serialize};

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

pub enum AgentEvent {
    TextDelta(String),
    ToolStarted {
        name: String,
        input: serde_json::Value,
    },
    ToolFinished {
        name: String,
        output: String,
    },
    TurnComplete {
        stop_reason: StopReason,
    },
    Error(String),
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
    fn message_round_trips_through_json() {
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
                },
                ToolResult {
                    tool_use_id: "Mock ID".to_string(),
                    content: "Mock Content".to_string(),
                    is_error: false,
                },
            ],
        };

        // Act
        let serialized = serde_json::to_string(&message).unwrap();
        let unserialized: Message = serde_json::from_str(&serialized).unwrap();

        // Assert
        assert_eq!(message, unserialized);
        // Shape check — this pins the format M2's JSONL sessions will write to disk
        assert_eq!(
            serde_json::to_value(&message).unwrap(),
            json!({
                "role": "Assistant",
                "content": [
                    { "Text": { "text": "Mock Text" } },
                    { "ToolUse": { "id": "Mock ID", "name": "Mock Name", "input": "Mock Input" } },
                    { "ToolResult": { "tool_use_id": "Mock ID", "content": "Mock Content", "is_error": false } }
                ]
            })
        );
    }
}
