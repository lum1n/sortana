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

/// How the tier is picked from the backend's probabilities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    /// The most likely tier, moved up one when confidence is below
    /// `min_confidence` and `bias_up` is on.
    Confidence,
    /// The tier with the lowest expected cost, from the models' list prices.
    /// Choosing a tier costs its model's price. When the task turns out to
    /// need a stronger model, it is re-run there, which costs that model's
    /// price times `1 + miss_penalty`; the penalty stands for the time and
    /// the bad answer lost. Tiers sharing a model count as one. Falls back to
    /// `Confidence` when a tier's model has no known price.
    Cost { miss_penalty: f32 },
}

impl Decision {
    /// [`Decision::Cost`] with `miss_penalty = 1`: a miss costs double the
    /// stronger model's price.
    pub const COST: Decision = Decision::Cost { miss_penalty: 1.0 };
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
    pub decision: Decision,
    /// Lowest tier to route to, e.g. to keep the smallest model out.
    pub min_tier: Tier,
    /// Highest tier to route to, e.g. to cap spend on a free plan.
    pub max_tier: Tier,
    /// Ask the backend how much reasoning effort the task needs.
    pub ask_effort: bool,
    /// Ask the backend whether the task needs vision and tools.
    pub ask_needs: bool,
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
            decision: Decision::Confidence,
            min_tier: Tier::Simple,
            max_tier: Tier::VeryHard,
            ask_effort: true,
            ask_needs: true,
        }
    }
}
