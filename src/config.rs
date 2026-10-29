use std::time::Duration;

use crate::Tier;

/// What `classify` does when the backend fails after all retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnError {
    /// Return this tier with `degraded = true`.
    Fallback(Tier),
    /// Return the error.
    Fail,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Deadline for each backend attempt.
    pub timeout: Duration,
    /// Extra attempts after the first one, for retryable errors only.
    pub retries: u32,
    /// Delay before the first retry; doubles on each further retry.
    pub backoff: Duration,
    pub on_error: OnError,
    /// Below this backend confidence, `bias_up` moves the tier up by one.
    pub min_confidence: f32,
    pub bias_up: bool,
    /// Longer inputs keep their head and tail. `0` disables truncation.
    pub max_input_chars: usize,
    /// Backend model id; used by the Jev backend built through `ClassifierBuilder::jev`.
    pub model: String,
    /// Maximum in-flight requests in `classify_many`.
    pub concurrency: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            timeout: Duration::from_secs(2),
            retries: 2,
            backoff: Duration::from_millis(100),
            on_error: OnError::Fallback(Tier::Medium),
            min_confidence: 0.5,
            bias_up: true,
            max_input_chars: 8000,
            model: "jev-latest".to_owned(),
            concurrency: 8,
        }
    }
}
