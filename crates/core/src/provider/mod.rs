use std::str::Utf8Error;
use thiserror::Error;

mod openai;
mod sse;

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("api error")]
    Api { status: u16, body: String },

    #[error("network error")]
    Network(#[from] reqwest::Error),

    #[error("parsing error")]
    Parsing(#[from] Utf8Error),
}
