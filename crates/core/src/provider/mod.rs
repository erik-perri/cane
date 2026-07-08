use thiserror::Error;

mod openai;

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("api error")]
    Api { status: u16, body: String },

    #[error("network error")]
    Network(#[from] reqwest::Error),
}
