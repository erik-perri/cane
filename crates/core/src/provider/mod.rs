use std::str::Utf8Error;
use thiserror::Error;

mod openai;
mod sse;

pub(crate) use openai::OpenAiClient;

pub(crate) struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
}

#[derive(Error, Debug)]
pub enum ProviderError {
    /// A real non-2xx HTTP response. Status and body are the server's own.
    #[error("api error ({status}): {body}")]
    Api { status: u16, body: String },

    /// The turn was cancelled before its stream produced a complete message.
    #[error("cancelled")]
    Cancelled,

    #[error("network error")]
    Network(#[from] reqwest::Error),

    #[error("parsing error")]
    Parsing(#[from] Utf8Error),

    /// The transport succeeded, but the payload broke the protocol contract.
    #[error("protocol error: {detail}")]
    Protocol { detail: String },
}
