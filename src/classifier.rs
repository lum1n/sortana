use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;

use crate::backend::{Ask, Backend, ClassifyRequest};
use crate::config::{Config, Decision, OnError};
use crate::rubric::{RUBRIC, RUBRIC_VERSION};
use crate::{
    Assessment, Context, Effort, Error, ModelInfo, ModelMap, Needs, Result, Tier, TierProbs,
};

/// Marker inserted where a long input was cut.
pub const TRUNCATION_MARKER: &str = "\n[...]\n";

/// Result of classifying one task.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Classification {
    /// The tier to route to.
    pub tier: Tier,
    /// Model id for `tier` from the [`ModelMap`].
    pub model: String,
    /// The most likely tier, before confidence, limits, needs or cost moved it.
    pub raw_tier: Tier,
    /// Normalized probabilities, indexed like [`Tier::ALL`]. All zero when `degraded`.
    pub probabilities: [f32; 4],
    /// Backend confidence in `[0, 1]`. Zero when `degraded`.
    pub confidence: f32,
    /// Reasoning effort the task needs. `None` when not asked or `degraded`.
    pub effort: Option<Effort>,
    /// `effort` as the value of the model's own effort setting (see
    /// [`ModelInfo::effort_value`]). `None` without an effort, model info or
    /// effort setting.
    pub reasoning_effort: Option<&'static str>,
    /// What the model has to support.
    pub needs: Needs,
    /// Metadata of `model`, when it is a known lineup model.
    pub info: Option<&'static ModelInfo>,
    /// Whether `model` meets `needs`: `Some(false)` when neither it nor any
    /// stronger model within `max_tier` does, `None` when its metadata is unknown.
    pub fits: Option<bool>,
    /// True when `tier` is the configured fallback because the backend failed.
    pub degraded: bool,
    /// True when low confidence moved the tier one above `raw_tier`.
    pub bumped: bool,
    /// True when `min_tier` or `max_tier` moved the tier.
    pub clamped: bool,
    /// True when the tier moved up to a model that meets `needs`.
    pub refit: bool,
    /// True when served from the cache.
    pub cached: bool,
    /// The backend error behind a degraded result.
    pub error: Option<String>,
    pub rubric_version: &'static str,
}

/// Classifies task descriptions into complexity tiers. Cheap to clone.
#[derive(Clone)]
pub struct Classifier {
    inner: Arc<Inner>,
}

struct Inner {
    backend: Box<dyn Backend>,
    config: Config,
    models: ModelMap,
    #[cfg(feature = "cache")]
    cache: Option<std::sync::Mutex<lru::LruCache<String, Assessment>>>,
}

impl std::fmt::Debug for Classifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Classifier")
            .field("config", &self.inner.config)
            .field("models", &self.inner.models)
            .finish_non_exhaustive()
    }
}

impl Classifier {
    pub fn builder() -> ClassifierBuilder {
        ClassifierBuilder::default()
    }

    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    /// The configured [`ModelMap`].
    pub fn models(&self) -> &ModelMap {
        &self.inner.models
    }

    /// Model id for a tier, from the configured [`ModelMap`].
    pub fn model_for(&self, tier: Tier) -> &str {
        self.inner.models.get(tier)
    }

    /// Classify one task. Backend failures yield the fallback tier with
    /// `degraded = true` unless `on_error` is [`OnError::Fail`]. Empty input is
    /// always an error.
    pub async fn classify(&self, text: &str) -> Result<Classification> {
        self.classify_for(text, &Context::default(), &self.inner.models)
            .await
    }

    /// Classify a task that comes with `ctx`: attachments, prompt size and
    /// the like.
    pub async fn classify_with(&self, text: &str, ctx: &Context) -> Result<Classification> {
        self.classify_for(text, ctx, &self.inner.models).await
    }

    /// Like [`classify_with`](Self::classify_with), routing to `models`
    /// instead of the configured map, e.g. the ids an app's model picker
    /// offers (see [`Lineup::models_among`](crate::Lineup::models_among)).
    pub async fn classify_for(
        &self,
        text: &str,
        ctx: &Context,
        models: &ModelMap,
    ) -> Result<Classification> {
        let text = text.trim();
        if text.is_empty() {
            return Err(Error::EmptyInput);
        }
        let cfg = &self.inner.config;
        let req = ClassifyRequest {
            text: truncate(text, cfg.max_input_chars),
            context: ctx.render(),
            rubric: &RUBRIC,
            ask: Ask {
                effort: cfg.ask_effort,
                needs: cfg.ask_needs,
            },
        };

        #[cfg(feature = "cache")]
        let key = cache_key(text, &req);
        #[cfg(feature = "cache")]
        if let Some(cache) = &self.inner.cache
            && let Some(a) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key)
        {
            let mut c = self.decide(a, ctx, models);
            c.cached = true;
            return Ok(c);
        }

        match self.call_with_retries(req).await {
            Ok(a) => {
                #[cfg(feature = "cache")]
                if let Some(cache) = &self.inner.cache {
                    cache.lock().unwrap_or_else(|e| e.into_inner()).put(key, a);
                }
                Ok(self.decide(&a, ctx, models))
            }
            Err(e) => match cfg.on_error {
                OnError::Fail => Err(e),
                OnError::Fallback(tier) => {
                    let needs = Needs::from_context(ctx);
                    let mut c = self.route(tier, false, needs, None, models);
                    c.raw_tier = tier;
                    c.degraded = true;
                    c.error = Some(e.to_string());
                    Ok(c)
                }
            },
        }
    }

    /// Classify several tasks concurrently (up to `Config::concurrency` in
    /// flight). Results are in input order.
    pub async fn classify_many<I, S>(&self, texts: I) -> Vec<Result<Classification>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        futures_util::stream::iter(texts)
            .map(|t| async move { self.classify(t.as_ref()).await })
            .buffered(self.inner.config.concurrency)
            .collect()
            .await
    }

    /// The next model up from `from`, for when its model failed or the user
    /// asked for a stronger one: the lowest tier above `from.tier`, up to
    /// `max_tier`, whose model differs from `from.model`, preferring models
    /// that meet `from.needs`. `None` when there is no such tier.
    pub fn escalate(&self, from: &Classification) -> Option<Classification> {
        self.escalate_for(from, &self.inner.models)
    }

    /// [`escalate`](Self::escalate) within `models`.
    pub fn escalate_for(&self, from: &Classification, models: &ModelMap) -> Option<Classification> {
        let up: Vec<Tier> = Tier::ALL
            .into_iter()
            .filter(|&t| t > from.tier && t <= self.inner.config.max_tier)
            .filter(|&t| models.get(t) != from.model)
            .collect();
        let fit = |t: Tier| fit(models, t, &from.needs);
        let tier = up
            .iter()
            .copied()
            .find(|&t| fit(t) != Some(false))
            .or_else(|| up.first().copied())?;
        let info = models.info(tier);
        Some(Classification {
            tier,
            model: models.get(tier).to_owned(),
            reasoning_effort: effort_value(info, from.effort),
            info,
            fits: fit(tier),
            bumped: false,
            clamped: false,
            refit: false,
            cached: false,
            ..from.clone()
        })
    }

    async fn call_with_retries(&self, req: ClassifyRequest) -> Result<Assessment> {
        let cfg = &self.inner.config;
        let mut attempt = 0;
        loop {
            let result =
                match tokio::time::timeout(cfg.timeout, self.inner.backend.assess(req.clone()))
                    .await
                {
                    Ok(r) => r.and_then(normalize),
                    Err(_) => Err(Error::Timeout(cfg.timeout)),
                };
            match result {
                Err(e) if e.is_retryable() && attempt < cfg.retries => {
                    tokio::time::sleep(backoff(cfg.backoff, attempt)).await;
                    attempt += 1;
                }
                r => return r,
            }
        }
    }

    fn decide(&self, a: &Assessment, ctx: &Context, models: &ModelMap) -> Classification {
        let cfg = &self.inner.config;
        let probs = &a.tiers;
        let raw_tier = probs.argmax();
        let needs = a.needs(ctx);
        let cheapest = match cfg.decision {
            Decision::Cost { miss_penalty } => {
                cheapest(probs, models, &self.allowed(), miss_penalty)
            }
            Decision::Confidence => None,
        };
        let (tier, bumped) = match cheapest {
            Some(t) => (t, false),
            None => {
                let bump = cfg.bias_up && probs.confidence < cfg.min_confidence;
                let tier = if bump { raw_tier.up() } else { raw_tier };
                (tier, tier != raw_tier)
            }
        };
        let effort = a.effort.as_ref().map(Effort::argmax);
        let mut c = self.route(tier, bumped, needs, effort, models);
        c.raw_tier = raw_tier;
        c.probabilities = probs.probabilities;
        c.confidence = probs.confidence;
        c
    }

    /// Tiers within `min_tier..=max_tier`, ascending.
    fn allowed(&self) -> Vec<Tier> {
        let cfg = &self.inner.config;
        Tier::ALL
            .into_iter()
            .filter(|t| (cfg.min_tier..=cfg.max_tier).contains(t))
            .collect()
    }

    /// Apply the tier limits and needs to `tier` and build the result. Needs
    /// only move the tier up: to the lowest tier from the limited one up to
    /// `max_tier` whose model meets them (unknown models count as meeting
    /// them). When none does, the tier stays and `fits` is `Some(false)`.
    fn route(
        &self,
        tier: Tier,
        bumped: bool,
        needs: Needs,
        effort: Option<Effort>,
        models: &ModelMap,
    ) -> Classification {
        let cfg = &self.inner.config;
        let wanted = tier;
        let limited = tier.clamp(cfg.min_tier, cfg.max_tier);
        let tier = Tier::ALL
            .into_iter()
            .filter(|&t| t >= limited && t <= cfg.max_tier)
            .find(|&t| fit(models, t, &needs) != Some(false))
            .unwrap_or(limited);
        let info = models.info(tier);
        Classification {
            tier,
            model: models.get(tier).to_owned(),
            raw_tier: tier,
            probabilities: [0.0; 4],
            confidence: 0.0,
            effort,
            reasoning_effort: effort_value(info, effort),
            needs,
            info,
            fits: fit(models, tier, &needs),
            degraded: false,
            bumped,
            clamped: limited != wanted,
            refit: tier != limited,
            cached: false,
            error: None,
            rubric_version: RUBRIC_VERSION,
        }
    }
}

/// Whether the tier's model meets `needs`; `None` when its info is unknown.
fn fit(models: &ModelMap, tier: Tier, needs: &Needs) -> Option<bool> {
    models.info(tier).map(|i| i.meets(needs))
}

fn effort_value(info: Option<&ModelInfo>, effort: Option<Effort>) -> Option<&'static str> {
    info?.effort_value(effort?)
}

/// The tier in `pool` with the lowest expected cost (see [`Decision::Cost`]),
/// before needs are applied.
/// Ties go to the stronger tier. Tasks harder than the strongest tier in
/// `pool` count as needing that tier. `None` when a model has no known price.
fn cheapest(
    probs: &TierProbs,
    models: &ModelMap,
    pool: &[Tier],
    miss_penalty: f32,
) -> Option<Tier> {
    let mut price = [0.0f32; 4];
    for t in Tier::ALL {
        price[t.index()] = models.info(t)?.price()?;
    }
    // A task harder than the strongest allowed tier can only be re-run there.
    let mut p = probs.probabilities;
    let ceiling = pool.iter().max()?.index();
    p[ceiling] += p[ceiling + 1..].iter().sum::<f32>();
    p[ceiling + 1..].fill(0.0);
    let mut best: Option<(Tier, f32)> = None;
    for &t in pool {
        // The strongest tier this tier's model also serves.
        let top = (t.index()..4)
            .take_while(|&j| models.get(Tier::ALL[j]) == models.get(t))
            .last()
            .unwrap_or(t.index());
        let miss: f32 = (top + 1..4).map(|k| p[k] * price[k]).sum();
        let cost = price[t.index()] + (1.0 + miss_penalty) * miss;
        if best.is_none_or(|(_, b)| cost <= b) {
            best = Some((t, cost));
        }
    }
    best.map(|(t, _)| t)
}

fn backoff(base: Duration, attempt: u32) -> Duration {
    base.saturating_mul(1 << attempt.min(16))
}

/// Normalize the tier probabilities and drop unusable extra answers.
fn normalize(mut a: Assessment) -> Result<Assessment> {
    a.tiers = normalize_tiers(a.tiers)?;
    let unit = |x: f32| (0.0..=1.0).contains(&x);
    a.vision = a.vision.filter(|&x| unit(x));
    a.tools = a.tools.filter(|&x| unit(x));
    a.effort = a
        .effort
        .filter(|e| e.iter().all(|x| x.is_finite() && *x >= 0.0) && e.iter().sum::<f32>() > 0.0);
    Ok(a)
}

/// Make probabilities sum to 1 and clamp confidence to `[0, 1]`.
fn normalize_tiers(mut p: TierProbs) -> Result<TierProbs> {
    for x in &mut p.probabilities {
        if !x.is_finite() || *x < 0.0 {
            *x = 0.0;
        }
    }
    let sum: f32 = p.probabilities.iter().sum();
    if sum <= 0.0 {
        return Err(Error::InvalidResponse(
            "all tier probabilities are zero".into(),
        ));
    }
    for x in &mut p.probabilities {
        *x /= sum;
    }
    p.confidence = if p.confidence.is_finite() {
        p.confidence.clamp(0.0, 1.0)
    } else {
        0.0
    };
    Ok(p)
}

/// Keep the head and tail of `text` when it is longer than `max_chars`
/// characters, joined by [`TRUNCATION_MARKER`]. `0` means no limit.
pub fn truncate(text: &str, max_chars: usize) -> String {
    if max_chars == 0 || text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let head_chars = max_chars.div_ceil(2);
    let tail_chars = max_chars - head_chars;
    let head_end = text
        .char_indices()
        .nth(head_chars)
        .map_or(text.len(), |(i, _)| i);
    let tail_start = if tail_chars == 0 {
        text.len()
    } else {
        text.char_indices()
            .nth_back(tail_chars - 1)
            .map_or(0, |(i, _)| i)
    };
    format!(
        "{}{TRUNCATION_MARKER}{}",
        &text[..head_end],
        &text[tail_start..]
    )
}

/// Case- and whitespace-insensitive cache key of the text, plus the
/// context and the questions asked.
#[cfg(feature = "cache")]
fn cache_key(text: &str, req: &ClassifyRequest) -> String {
    let mut key = text
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ");
    key.push_str(match (req.ask.effort, req.ask.needs) {
        (false, false) => "\0--",
        (true, false) => "\0e-",
        (false, true) => "\0-n",
        (true, true) => "\0en",
    });
    if let Some(c) = &req.context {
        key.push('\0');
        key.push_str(c);
    }
    key
}

enum PendingBackend {
    None,
    Custom(Box<dyn Backend>),
    #[cfg(feature = "jev")]
    Jev(String),
}

/// Builder for [`Classifier`]. Set a backend with [`jev`](Self::jev) or
/// [`backend`](Self::backend); the last one set wins.
pub struct ClassifierBuilder {
    backend: PendingBackend,
    config: Config,
    models: ModelMap,
    #[cfg(feature = "cache")]
    cache_capacity: usize,
}

impl Default for ClassifierBuilder {
    fn default() -> Self {
        ClassifierBuilder {
            backend: PendingBackend::None,
            config: Config::default(),
            models: ModelMap::default(),
            #[cfg(feature = "cache")]
            cache_capacity: 0,
        }
    }
}

impl ClassifierBuilder {
    /// Use the Jev backend with this API key and `Config::model`.
    #[cfg(feature = "jev")]
    pub fn jev(mut self, api_key: impl Into<String>) -> Self {
        self.backend = PendingBackend::Jev(api_key.into());
        self
    }

    pub fn backend(mut self, backend: impl Backend + 'static) -> Self {
        self.backend = PendingBackend::Custom(Box::new(backend));
        self
    }

    /// Replace the whole config.
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn retries(mut self, retries: u32) -> Self {
        self.config.retries = retries;
        self
    }

    pub fn backoff(mut self, backoff: Duration) -> Self {
        self.config.backoff = backoff;
        self
    }

    pub fn on_error(mut self, on_error: OnError) -> Self {
        self.config.on_error = on_error;
        self
    }

    pub fn min_confidence(mut self, min_confidence: f32) -> Self {
        self.config.min_confidence = min_confidence;
        self
    }

    pub fn bias_up(mut self, bias_up: bool) -> Self {
        self.config.bias_up = bias_up;
        self
    }

    pub fn max_input_chars(mut self, max_input_chars: usize) -> Self {
        self.config.max_input_chars = max_input_chars;
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.config.concurrency = concurrency;
        self
    }

    pub fn decision(mut self, decision: Decision) -> Self {
        self.config.decision = decision;
        self
    }

    /// Never route below `tier`.
    pub fn min_tier(mut self, tier: Tier) -> Self {
        self.config.min_tier = tier;
        self
    }

    /// Never route above `tier`.
    pub fn max_tier(mut self, tier: Tier) -> Self {
        self.config.max_tier = tier;
        self
    }

    /// Ask for the task's reasoning effort (default on).
    pub fn ask_effort(mut self, on: bool) -> Self {
        self.config.ask_effort = on;
        self
    }

    /// Ask whether the task needs vision and tools (default on).
    pub fn ask_needs(mut self, on: bool) -> Self {
        self.config.ask_needs = on;
        self
    }

    /// Route to these models. Defaults to the Anthropic lineup.
    pub fn models(mut self, models: ModelMap) -> Self {
        self.models = models;
        self
    }

    /// Cache up to `capacity` results by normalized text. `0` disables it.
    #[cfg(feature = "cache")]
    pub fn cache(mut self, capacity: usize) -> Self {
        self.cache_capacity = capacity;
        self
    }

    pub fn build(self) -> Result<Classifier> {
        let cfg = &self.config;
        if !(0.0..=1.0).contains(&cfg.min_confidence) {
            return Err(Error::Config("min_confidence must be within [0, 1]".into()));
        }
        if cfg.timeout.is_zero() {
            return Err(Error::Config("timeout must be positive".into()));
        }
        if cfg.min_tier > cfg.max_tier {
            return Err(Error::Config("min_tier is above max_tier".into()));
        }
        if let Decision::Cost { miss_penalty } = cfg.decision
            && !(miss_penalty.is_finite() && miss_penalty >= 0.0)
        {
            return Err(Error::Config(
                "miss_penalty must be finite and non-negative".into(),
            ));
        }
        if cfg.concurrency == 0 {
            return Err(Error::Config("concurrency must be at least 1".into()));
        }
        let backend: Box<dyn Backend> = match self.backend {
            PendingBackend::None => {
                return Err(Error::Config(
                    "no backend set; call .jev() or .backend()".into(),
                ));
            }
            PendingBackend::Custom(b) => b,
            #[cfg(feature = "jev")]
            PendingBackend::Jev(key) => {
                if key.trim().is_empty() {
                    return Err(Error::Config("Jev API key is empty".into()));
                }
                Box::new(crate::backend::jev::JevBackend::new(key)?.model(cfg.model.clone()))
            }
        };
        Ok(Classifier {
            inner: Arc::new(Inner {
                backend,
                #[cfg(feature = "cache")]
                cache: std::num::NonZeroUsize::new(self.cache_capacity)
                    .map(|n| std::sync::Mutex::new(lru::LruCache::new(n))),
                config: self.config,
                models: self.models,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_head_and_tail() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdefghij", 0), "abcdefghij");
        assert_eq!(
            truncate("abcdefghij", 4),
            format!("ab{TRUNCATION_MARKER}ij")
        );
        assert_eq!(
            truncate("abcdefghij", 5),
            format!("abc{TRUNCATION_MARKER}ij")
        );
        assert_eq!(truncate("abcdefghij", 1), format!("a{TRUNCATION_MARKER}"));
        // Multi-byte characters are counted as characters and never split.
        assert_eq!(truncate("æøåæøå", 4), format!("æø{TRUNCATION_MARKER}øå"));
    }

    #[test]
    fn normalize_rescales_and_rejects_zero() {
        let p = normalize_tiers(TierProbs {
            probabilities: [1.0, 3.0, f32::NAN, -1.0],
            confidence: 2.0,
        })
        .unwrap();
        assert_eq!(p.probabilities, [0.25, 0.75, 0.0, 0.0]);
        assert_eq!(p.confidence, 1.0);
        assert!(
            normalize_tiers(TierProbs {
                probabilities: [0.0; 4],
                confidence: 1.0
            })
            .is_err()
        );
    }

    #[test]
    fn backoff_doubles() {
        let b = Duration::from_millis(100);
        assert_eq!(backoff(b, 0), b);
        assert_eq!(backoff(b, 2), Duration::from_millis(400));
    }

    #[cfg(feature = "cache")]
    #[test]
    fn cache_key_normalizes() {
        let req = |context: Option<&str>| ClassifyRequest {
            text: String::new(),
            context: context.map(str::to_owned),
            rubric: &RUBRIC,
            ask: Ask {
                effort: true,
                needs: false,
            },
        };
        assert_eq!(
            cache_key("  Fix   the\tTypo \n", &req(None)),
            "fix the typo\0e-"
        );
        assert_ne!(
            cache_key("fix the typo", &req(None)),
            cache_key("fix the typo", &req(Some("Context:")))
        );
    }
}
