use crate::message::{Message, StopReason};
use serde::{Deserialize, Serialize};
use std::str::Utf8Error;
use thiserror::Error;

mod openai;
mod sse;

pub(crate) use openai::OpenAiClient;

pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReportedCost {
    pub amount: String,
    pub currency: String,
    pub source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelTurn {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: Option<ModelUsage>,
    pub request_id: Option<String>,
    pub provider_cost: Option<ReportedCost>,
}

#[derive(Error, Debug)]
pub enum ProviderError {
    /// A real non-2xx HTTP response. Status and body are the server's own.
    #[error("api error ({status}): {body}")]
    Api { status: u16, body: String },

    /// The turn was cancelled before its stream produced a complete message.
    #[error("cancelled")]
    Cancelled,

    #[error("invalid base URL '{base_url}': {detail}")]
    InvalidBaseUrl { base_url: String, detail: String },

    #[error("network error")]
    Network(#[from] reqwest::Error),

    #[error("parsing error")]
    Parsing(#[from] Utf8Error),

    /// The transport succeeded, but the payload broke the protocol contract.
    #[error("protocol error: {detail}")]
    Protocol { detail: String },
}
