use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;

use crate::backend::{Backend, ClassifyRequest};
use crate::config::{Config, OnError};
use crate::rubric::{RUBRIC, RUBRIC_VERSION};
use crate::{Error, ModelMap, Result, Tier, TierProbs};

/// Marker inserted where a long input was cut.
pub const TRUNCATION_MARKER: &str = "\n[...]\n";

/// Result of classifying one task.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Classification {
    /// The tier to route to.
    pub tier: Tier,
    /// The most likely tier before any confidence bump.
    pub raw_tier: Tier,
    /// Normalized probabilities, indexed like [`Tier::ALL`]. All zero when `degraded`.
    pub probabilities: [f32; 4],
    /// Backend confidence in `[0, 1]`. Zero when `degraded`.
    pub confidence: f32,
    /// True when `tier` is the configured fallback because the backend failed.
    pub degraded: bool,
    /// True when low confidence moved `tier` one above `raw_tier`.
    pub bumped: bool,
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
    cache: Option<std::sync::Mutex<lru::LruCache<String, TierProbs>>>,
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

    /// Model id for a tier, from the configured [`ModelMap`].
    pub fn model_for(&self, tier: Tier) -> &str {
        self.inner.models.get(tier)
    }

    /// Classify one task. Backend failures yield the fallback tier with
    /// `degraded = true` unless `on_error` is [`OnError::Fail`]. Empty input is
    /// always an error.
    pub async fn classify(&self, text: &str) -> Result<Classification> {
        let text = text.trim();
        if text.is_empty() {
            return Err(Error::EmptyInput);
        }
        let cfg = &self.inner.config;

        #[cfg(feature = "cache")]
        let key = cache_key(text);
        #[cfg(feature = "cache")]
        if let Some(cache) = &self.inner.cache
            && let Some(probs) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key)
        {
            let mut c = self.decide(*probs);
            c.cached = true;
            return Ok(c);
        }

        let req = ClassifyRequest {
            text: truncate(text, cfg.max_input_chars),
            rubric: &RUBRIC,
        };
        match self.call_with_retries(req).await {
            Ok(probs) => {
                #[cfg(feature = "cache")]
                if let Some(cache) = &self.inner.cache {
                    cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .put(key, probs);
                }
                Ok(self.decide(probs))
            }
            Err(e) => match cfg.on_error {
                OnError::Fail => Err(e),
                OnError::Fallback(tier) => Ok(Classification {
                    tier,
                    raw_tier: tier,
                    probabilities: [0.0; 4],
                    confidence: 0.0,
                    degraded: true,
                    bumped: false,
                    cached: false,
                    error: Some(e.to_string()),
                    rubric_version: RUBRIC_VERSION,
                }),
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

    async fn call_with_retries(&self, req: ClassifyRequest) -> Result<TierProbs> {
        let cfg = &self.inner.config;
        let mut attempt = 0;
        loop {
            let result =
                match tokio::time::timeout(cfg.timeout, self.inner.backend.classify(req.clone()))
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

    fn decide(&self, probs: TierProbs) -> Classification {
        let cfg = &self.inner.config;
        let raw_tier = probs.argmax();
        let bump = cfg.bias_up && probs.confidence < cfg.min_confidence;
        let tier = if bump { raw_tier.up() } else { raw_tier };
        Classification {
            tier,
            raw_tier,
            probabilities: probs.probabilities,
            confidence: probs.confidence,
            degraded: false,
            bumped: tier != raw_tier,
            cached: false,
            error: None,
            rubric_version: RUBRIC_VERSION,
        }
    }
}

fn backoff(base: Duration, attempt: u32) -> Duration {
    base.saturating_mul(1 << attempt.min(16))
}

/// Make probabilities sum to 1 and clamp confidence to `[0, 1]`.
fn normalize(mut p: TierProbs) -> Result<TierProbs> {
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

/// Case- and whitespace-insensitive cache key.
#[cfg(feature = "cache")]
fn cache_key(text: &str) -> String {
    text.split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
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
        let p = normalize(TierProbs {
            probabilities: [1.0, 3.0, f32::NAN, -1.0],
            confidence: 2.0,
        })
        .unwrap();
        assert_eq!(p.probabilities, [0.25, 0.75, 0.0, 0.0]);
        assert_eq!(p.confidence, 1.0);
        assert!(
            normalize(TierProbs {
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
        assert_eq!(cache_key("  Fix   the\tTypo \n"), "fix the typo");
    }
}
