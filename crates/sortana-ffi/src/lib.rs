//! UniFFI surface of sortana for Swift (iOS/macOS) and Kotlin.
//!
//! A [`Suggester`] classifies a task with Jev and names the model to run it
//! on, picked from the lineup and, when the app has its own model list, from
//! that list. Everything else stays in the `sortana` crate.

use std::sync::Arc;

use sortana::{Classifier, JevBackend, Lineup, OnError};

uniffi::setup_scaffolding!();

/// Complexity tier of a task, easiest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum Tier {
    Simple,
    Medium,
    Hard,
    VeryHard,
}

impl From<sortana::Tier> for Tier {
    fn from(t: sortana::Tier) -> Self {
        match t {
            sortana::Tier::Simple => Tier::Simple,
            sortana::Tier::Medium => Tier::Medium,
            sortana::Tier::Hard => Tier::Hard,
            sortana::Tier::VeryHard => Tier::VeryHard,
        }
    }
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum SortanaError {
    #[error("task text is empty")]
    EmptyInput,
    #[error("unknown lineup {lineup:?}")]
    UnknownLineup { lineup: String },
    #[error("invalid configuration: {message}")]
    Config { message: String },
    /// The backend failed after all retries (network, HTTP, bad response).
    #[error("{message}")]
    Backend { message: String },
}

impl From<sortana::Error> for SortanaError {
    fn from(e: sortana::Error) -> Self {
        match e {
            sortana::Error::EmptyInput => SortanaError::EmptyInput,
            sortana::Error::Config(message) => SortanaError::Config { message },
            e => SortanaError::Backend {
                message: e.to_string(),
            },
        }
    }
}

/// A model suggestion for one task.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct Suggestion {
    pub tier: Tier,
    /// The model for `tier`; `None` when none of the app's models fits the lineup.
    pub model: Option<String>,
    /// Probabilities indexed like `[simple, medium, hard, very_hard]`.
    pub probabilities: Vec<f32>,
    pub confidence: f32,
    /// True when low confidence moved the tier up one.
    pub bumped: bool,
    pub rubric_version: String,
}

/// Classifies tasks with Jev and suggests a model from one lineup.
#[derive(Debug, uniffi::Object)]
pub struct Suggester {
    classifier: Classifier,
    lineup: &'static Lineup,
}

#[uniffi::export(async_runtime = "tokio")]
impl Suggester {
    /// `lineup` is a sortana lineup id or alias (`anthropic`, `claude`,
    /// `openai`, ...). `base_url` overrides Jev's endpoint, for tests.
    #[uniffi::constructor]
    pub fn new(
        api_key: String,
        lineup: String,
        base_url: Option<String>,
    ) -> Result<Arc<Self>, SortanaError> {
        let l = Lineup::get(&lineup).ok_or(SortanaError::UnknownLineup { lineup })?;
        let mut jev = JevBackend::new(api_key)?;
        if let Some(url) = base_url {
            jev = jev.base_url(url);
        }
        let classifier = Classifier::builder()
            .backend(jev)
            .on_error(OnError::Fail)
            .build()?;
        Ok(Arc::new(Suggester {
            classifier,
            lineup: l,
        }))
    }

    /// The lineup's id, e.g. `anthropic`.
    pub fn lineup(&self) -> String {
        self.lineup.id.to_owned()
    }

    /// Classify `task` and pick its model from `available`, the ids the
    /// app's model picker offers (see `sortana::Lineup::model_among`). An
    /// empty `available` gives the maker's own model id.
    pub async fn suggest(
        &self,
        task: String,
        available: Vec<String>,
    ) -> Result<Suggestion, SortanaError> {
        let c = self.classifier.classify(&task).await?;
        Ok(Suggestion {
            tier: c.tier.into(),
            model: self.lineup.model_among(c.tier, &available),
            probabilities: c.probabilities.to_vec(),
            confidence: c.confidence,
            bumped: c.bumped,
            rubric_version: c.rubric_version.to_owned(),
        })
    }
}

/// The lineup for an agent harness's models, by ACC harness id: `claude`
/// and `opencode` run Anthropic's lineup, `codex` OpenAI's. `None` for
/// harnesses without one.
#[uniffi::export]
pub fn lineup_for_harness(harness: String) -> Option<String> {
    let id = match harness.trim().to_ascii_lowercase().as_str() {
        "claude" | "opencode" => "anthropic",
        "codex" => "openai",
        _ => return None,
    };
    Some(id.to_owned())
}

/// Version of the rubric the classifier sends.
#[uniffi::export]
pub fn rubric_version() -> String {
    sortana::rubric::RUBRIC_VERSION.to_owned()
}
