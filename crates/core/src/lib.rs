mod agent;
mod message;
mod provider;
mod tool;

pub use tool::{FileTool, Tool, ToolDefinition, dispatch};

use tracing::debug;

pub fn hello() {
    debug!("Hello, world!");
}
