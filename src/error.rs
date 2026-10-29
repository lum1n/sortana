use std::time::Duration;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("task text is empty")]
    EmptyInput,

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("backend timed out after {0:?}")]
    Timeout(Duration),

    /// Network or connection failure before an HTTP response arrived.
    #[error("transport error: {0}")]
    Transport(String),

    /// The backend answered with a non-success HTTP status.
    #[error("backend returned HTTP {status}: {body}")]
    Http { status: u16, body: String },

    /// The backend answered, but not with a usable tier distribution.
    #[error("invalid backend response: {0}")]
    InvalidResponse(String),

    #[error(transparent)]
    Preset(#[from] crate::PresetError),

    /// Any other backend-specific failure. Set `retryable` to opt into retries.
    #[error("backend error: {message}")]
    Backend { message: String, retryable: bool },
}

impl Error {
    /// Whether retrying the same request may succeed: timeouts, transport errors,
    /// HTTP 408/429/5xx, and `Backend` errors marked retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Timeout(_) | Error::Transport(_) => true,
            Error::Http { status, .. } => matches!(status, 408 | 429 | 500..=599),
            Error::Backend { retryable, .. } => *retryable,
            Error::EmptyInput | Error::Config(_) | Error::InvalidResponse(_) | Error::Preset(_) => {
                false
            }
        }
    }
}
