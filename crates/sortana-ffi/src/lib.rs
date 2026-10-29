//! UniFFI surface of sortana for Swift (iOS/macOS) and Kotlin.
//!
//! A [`Suggester`] classifies a task with Jev and names the model to run it
//! on, picked from the lineup and, when the app has its own model list, from
//! that list, along with the reasoning effort and capabilities the task
//! needs. Everything else stays in the `sortana` crate.

use std::sync::Arc;
use std::time::Duration;

use sortana::{Classifier, JevBackend, Lineup, ModelMap, OnError};

uniffi::setup_scaffolding!();

/// Complexity tier of a task, easiest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum Tier {
    Simple,
    Medium,
    Hard,
    VeryHard,
}

impl From<Tier> for sortana::Tier {
    fn from(t: Tier) -> Self {
        match t {
            Tier::Simple => sortana::Tier::Simple,
            Tier::Medium => sortana::Tier::Medium,
            Tier::Hard => sortana::Tier::Hard,
            Tier::VeryHard => sortana::Tier::VeryHard,
        }
    }
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

/// Fields are `detail`, not `message`, which would clash with Kotlin's
/// `Throwable.message`.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum SortanaError {
    #[error("task text is empty")]
    EmptyInput,
    #[error("unknown lineup {lineup:?}")]
    UnknownLineup { lineup: String },
    #[error("invalid configuration: {detail}")]
    Config { detail: String },
    /// The backend failed after all retries (network, HTTP, bad response).
    #[error("{detail}")]
    Backend { detail: String },
}

impl From<sortana::Error> for SortanaError {
    fn from(e: sortana::Error) -> Self {
        match e {
            sortana::Error::EmptyInput => SortanaError::EmptyInput,
            sortana::Error::Config(detail) => SortanaError::Config { detail },
            e => SortanaError::Backend {
                detail: e.to_string(),
            },
        }
    }
}

/// How much step-by-step reasoning a task needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum Effort {
    Low,
    Medium,
    High,
}

impl From<sortana::Effort> for Effort {
    fn from(e: sortana::Effort) -> Self {
        match e {
            sortana::Effort::Low => Effort::Low,
            sortana::Effort::Medium => Effort::Medium,
            sortana::Effort::High => Effort::High,
        }
    }
}

impl From<Effort> for sortana::Effort {
    fn from(e: Effort) -> Self {
        match e {
            Effort::Low => sortana::Effort::Low,
            Effort::Medium => sortana::Effort::Medium,
            Effort::High => sortana::Effort::High,
        }
    }
}

/// Capabilities the model has to have for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct Needs {
    /// Image input.
    pub vision: bool,
    /// Tool calling.
    pub tools: bool,
    /// Prompt size in tokens, from `TaskContext.promptTokens`.
    pub context_tokens: Option<u64>,
}

impl From<sortana::Needs> for Needs {
    fn from(n: sortana::Needs) -> Self {
        Needs {
            vision: n.vision,
            tools: n.tools,
            context_tokens: n.context_tokens,
        }
    }
}

impl From<Needs> for sortana::Needs {
    fn from(n: Needs) -> Self {
        sortana::Needs {
            vision: n.vision,
            tools: n.tools,
            context_tokens: n.context_tokens,
        }
    }
}

/// models.dev metadata of a lineup model. Prices are USD per million tokens.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct ModelInfo {
    /// Canonical id, e.g. `anthropic/claude-opus-5-5`.
    pub id: String,
    pub name: String,
    /// Context window in tokens; 0 when unknown.
    pub context: u32,
    pub max_input: u32,
    pub max_output: u32,
    pub input_cost: Option<f32>,
    pub output_cost: Option<f32>,
    pub vision: bool,
    pub pdf: bool,
    pub tools: bool,
    pub reasoning: bool,
    /// Values of the model's reasoning effort setting.
    pub efforts: Vec<String>,
}

impl From<&sortana::ModelInfo> for ModelInfo {
    fn from(m: &sortana::ModelInfo) -> Self {
        ModelInfo {
            id: m.id.to_owned(),
            name: m.name.to_owned(),
            context: m.context,
            max_input: m.max_input,
            max_output: m.max_output,
            input_cost: m.input_cost,
            output_cost: m.output_cost,
            vision: m.vision,
            pdf: m.pdf,
            tools: m.tools,
            reasoning: m.reasoning,
            efforts: m.efforts.iter().map(|e| (*e).to_owned()).collect(),
        }
    }
}

/// What the app knows about a task beyond its text. Every field is optional.
#[derive(Debug, Clone, Default, PartialEq, Eq, uniffi::Record)]
pub struct TaskContext {
    /// Files attached or in scope.
    #[uniffi(default = None)]
    pub files: Option<u32>,
    /// Images attached. Any image means the model needs vision.
    #[uniffi(default = 0)]
    pub images: u32,
    /// Prompt size in tokens; models whose input limit is smaller are skipped.
    #[uniffi(default = None)]
    pub prompt_tokens: Option<u64>,
    /// Conversation turns so far.
    #[uniffi(default = None)]
    pub turns: Option<u32>,
    /// Whether the request will offer tools; overrides Jev's guess.
    #[uniffi(default = None)]
    pub tools: Option<bool>,
    /// Other context, as short text.
    #[uniffi(default = None)]
    pub notes: Option<String>,
}

impl From<TaskContext> for sortana::Context {
    fn from(c: TaskContext) -> Self {
        sortana::Context {
            files: c.files,
            images: c.images,
            prompt_tokens: c.prompt_tokens,
            turns: c.turns,
            tools: c.tools,
            notes: c.notes,
        }
    }
}

/// How the tier is picked (see `sortana::Decision`).
#[derive(Debug, Clone, Copy, PartialEq, uniffi::Enum)]
pub enum Decision {
    /// The most likely tier, one up when Jev is unsure.
    Confidence,
    /// The tier with the lowest expected cost from list prices. A miss (the
    /// task needed a stronger model) costs that model's price times
    /// `1 + miss_penalty`.
    Cost { miss_penalty: f32 },
}

impl From<Decision> for sortana::Decision {
    fn from(d: Decision) -> Self {
        match d {
            Decision::Confidence => sortana::Decision::Confidence,
            Decision::Cost { miss_penalty } => sortana::Decision::Cost { miss_penalty },
        }
    }
}

/// Settings for [`Suggester::with_options`]. Unset fields keep sortana's defaults.
#[derive(Debug, Clone, Default, PartialEq, uniffi::Record)]
pub struct SuggesterOptions {
    /// Overrides Jev's endpoint, for tests.
    #[uniffi(default = None)]
    pub base_url: Option<String>,
    /// Never suggest below this tier.
    #[uniffi(default = None)]
    pub min_tier: Option<Tier>,
    /// Never suggest above this tier.
    #[uniffi(default = None)]
    pub max_tier: Option<Tier>,
    /// Defaults to `.confidence`.
    #[uniffi(default = None)]
    pub decision: Option<Decision>,
    /// Ask how much reasoning effort the task needs (default true).
    #[uniffi(default = None)]
    pub ask_effort: Option<bool>,
    /// Ask whether the task needs vision and tools (default true).
    #[uniffi(default = None)]
    pub ask_needs: Option<bool>,
    /// Deadline per Jev attempt in milliseconds (default 2000).
    #[uniffi(default = None)]
    pub timeout_ms: Option<u32>,
}

/// A model suggestion for one task.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct Suggestion {
    pub tier: Tier,
    /// The most likely tier, before confidence, limits, needs or cost moved it.
    pub raw_tier: Tier,
    /// The model for `tier`; `None` when none of the app's models fits the lineup.
    pub model: Option<String>,
    /// Probabilities indexed like `[simple, medium, hard, very_hard]`.
    pub probabilities: Vec<f32>,
    pub confidence: f32,
    /// Reasoning effort the task needs; `None` when not asked.
    pub effort: Option<Effort>,
    /// `effort` as the value of the model's own effort setting (`"high"`, `"xhigh"`, ...).
    pub reasoning_effort: Option<String>,
    pub needs: Needs,
    /// Metadata of the lineup model behind `model`.
    pub model_info: Option<ModelInfo>,
    /// Whether the model meets `needs`; `false` when neither it nor any
    /// stronger model up to `max_tier` does, `None` when unknown.
    pub fits: Option<bool>,
    /// True when low confidence moved the tier up one.
    pub bumped: bool,
    /// True when the tier limits moved the tier.
    pub clamped: bool,
    /// True when the tier moved up to a model that meets `needs`.
    pub refit: bool,
    pub rubric_version: String,
}

impl Suggestion {
    fn new(c: sortana::Classification, matched: bool) -> Self {
        Suggestion {
            tier: c.tier.into(),
            raw_tier: c.raw_tier.into(),
            model: matched.then_some(c.model),
            probabilities: c.probabilities.to_vec(),
            confidence: c.confidence,
            effort: c.effort.map(Into::into),
            reasoning_effort: c.reasoning_effort.map(str::to_owned),
            needs: c.needs.into(),
            model_info: c.info.map(Into::into),
            fits: c.fits,
            bumped: c.bumped,
            clamped: c.clamped,
            refit: c.refit,
            rubric_version: c.rubric_version.to_owned(),
        }
    }
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
        Self::with_options(
            api_key,
            lineup,
            SuggesterOptions {
                base_url,
                ..Default::default()
            },
        )
    }

    /// Like `new`, with tier limits, the decision rule and other settings.
    #[uniffi::constructor]
    pub fn with_options(
        api_key: String,
        lineup: String,
        options: SuggesterOptions,
    ) -> Result<Arc<Self>, SortanaError> {
        let l = Lineup::get(&lineup).ok_or(SortanaError::UnknownLineup { lineup })?;
        let mut jev = JevBackend::new(api_key)?;
        if let Some(url) = options.base_url {
            jev = jev.base_url(url);
        }
        let mut b = Classifier::builder()
            .backend(jev)
            .models(ModelMap::preset(l.id).map_err(sortana::Error::from)?)
            .on_error(OnError::Fail);
        if let Some(t) = options.min_tier {
            b = b.min_tier(t.into());
        }
        if let Some(t) = options.max_tier {
            b = b.max_tier(t.into());
        }
        if let Some(d) = options.decision {
            b = b.decision(d.into());
        }
        if let Some(on) = options.ask_effort {
            b = b.ask_effort(on);
        }
        if let Some(on) = options.ask_needs {
            b = b.ask_needs(on);
        }
        if let Some(ms) = options.timeout_ms {
            b = b.timeout(Duration::from_millis(ms.into()));
        }
        Ok(Arc::new(Suggester {
            classifier: b.build()?,
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
        self.suggest_with(task, TaskContext::default(), available)
            .await
    }

    /// `suggest` for a task that comes with `context`: attachments, prompt
    /// size and the like.
    pub async fn suggest_with(
        &self,
        task: String,
        context: TaskContext,
        available: Vec<String>,
    ) -> Result<Suggestion, SortanaError> {
        let (models, matched) = self.models(&available);
        let c = self
            .classifier
            .classify_for(&task, &context.into(), &models)
            .await?;
        Ok(Suggestion::new(c, matched))
    }

    /// The next model up from `from` among `available`, for when its model
    /// failed or the user wants a stronger one: the lowest higher tier (up
    /// to `max_tier`) with a different model, preferring models that meet
    /// `from.needs`. `None` at the top.
    pub fn escalate(&self, from: Suggestion, available: Vec<String>) -> Option<Suggestion> {
        let (models, matched) = self.models(&available);
        let tier = from.tier.into();
        let c = sortana::Classification {
            tier,
            model: from.model?,
            raw_tier: from.raw_tier.into(),
            probabilities: from.probabilities.try_into().unwrap_or_default(),
            confidence: from.confidence,
            effort: from.effort.map(Into::into),
            reasoning_effort: None,
            needs: from.needs.into(),
            info: models.info(tier),
            fits: from.fits,
            degraded: false,
            bumped: from.bumped,
            clamped: from.clamped,
            refit: from.refit,
            cached: false,
            error: None,
            rubric_version: sortana::rubric::RUBRIC_VERSION,
        };
        let up = self.classifier.escalate_for(&c, &models)?;
        Some(Suggestion::new(up, matched))
    }
}

impl Suggester {
    /// The model map for `available`, and whether any entry matched.
    fn models(&self, available: &[String]) -> (ModelMap, bool) {
        match self.lineup.models_among(available) {
            Some(m) => (m, true),
            None => (
                self.lineup.models_among::<&str>(&[]).expect("native map"),
                false,
            ),
        }
    }
}

/// Metadata of a lineup model by any id it goes by (canonical, native, a
/// host's id or `host/id`).
#[uniffi::export]
pub fn model_info(id: String) -> Option<ModelInfo> {
    sortana::ModelInfo::find(&id).map(Into::into)
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
