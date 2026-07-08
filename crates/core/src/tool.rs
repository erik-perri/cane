/// A tool the model can call. Definitions are data, not behavior (DESIGN §4);
/// the `Tool` trait that produces them arrives in Step 6 — providers only need
/// this shape to advertise tools on the wire.
#[derive(Clone, Debug)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema; maps to `function.parameters` on the OpenAI wire.
    pub input_schema: serde_json::Value,
}
