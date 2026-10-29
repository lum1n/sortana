//! Classifier behaviour against a scripted mock backend (no network).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sortana::{
    Backend, BoxFuture, Classifier, ClassifyRequest, Error, ModelMap, OnError, Result,
    TRUNCATION_MARKER, Tier, TierProbs,
};

type Step = (Duration, Result<TierProbs>);

/// Replays scripted results in order and records every request text.
#[derive(Clone, Default)]
struct Mock {
    steps: Arc<Mutex<VecDeque<Step>>>,
    seen: Arc<Mutex<Vec<String>>>,
}

impl Mock {
    fn new(steps: impl IntoIterator<Item = Result<TierProbs>>) -> Self {
        Self::delayed(steps.into_iter().map(|r| (Duration::ZERO, r)))
    }

    fn delayed(steps: impl IntoIterator<Item = Step>) -> Self {
        Mock {
            steps: Arc::new(Mutex::new(steps.into_iter().collect())),
            ..Default::default()
        }
    }

    fn calls(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    fn texts(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }
}

impl Backend for Mock {
    fn classify(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
        self.seen.lock().unwrap().push(req.text);
        let (delay, result) = self
            .steps
            .lock()
            .unwrap()
            .pop_front()
            .expect("mock called more often than scripted");
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            result
        })
    }
}

fn probs(p: [f32; 4], confidence: f32) -> Result<TierProbs> {
    Ok(TierProbs {
        probabilities: p,
        confidence,
    })
}

fn http(status: u16) -> Result<TierProbs> {
    Err(Error::Http {
        status,
        body: String::new(),
    })
}

fn classifier(mock: &Mock) -> sortana::ClassifierBuilder {
    Classifier::builder()
        .backend(mock.clone())
        .backoff(Duration::from_millis(10))
}

#[tokio::test]
async fn confident_answer_uses_argmax() {
    let mock = Mock::new([probs([0.05, 0.1, 0.8, 0.05], 0.9)]);
    let r = classifier(&mock)
        .build()
        .unwrap()
        .classify("refactor billing")
        .await
        .unwrap();
    assert_eq!(r.tier, Tier::Hard);
    assert_eq!(r.raw_tier, Tier::Hard);
    assert!(!r.bumped && !r.degraded && !r.cached);
    assert_eq!(r.rubric_version, sortana::rubric::RUBRIC_VERSION);
}

#[tokio::test]
async fn probabilities_are_normalized() {
    let mock = Mock::new([probs([2.0, 6.0, 0.0, 0.0], 0.9)]);
    let r = classifier(&mock)
        .build()
        .unwrap()
        .classify("x")
        .await
        .unwrap();
    assert_eq!(r.probabilities, [0.25, 0.75, 0.0, 0.0]);
    assert_eq!(r.tier, Tier::Medium);
}

#[tokio::test]
async fn low_confidence_bumps_one_tier() {
    let mock = Mock::new([probs([0.45, 0.4, 0.1, 0.05], 0.3)]);
    let r = classifier(&mock)
        .build()
        .unwrap()
        .classify("x")
        .await
        .unwrap();
    assert_eq!(r.raw_tier, Tier::Simple);
    assert_eq!(r.tier, Tier::Medium);
    assert!(r.bumped);
}

#[tokio::test]
async fn no_bump_when_disabled_or_at_top() {
    let mock = Mock::new([probs([0.45, 0.4, 0.1, 0.05], 0.3)]);
    let r = classifier(&mock)
        .bias_up(false)
        .build()
        .unwrap()
        .classify("x")
        .await
        .unwrap();
    assert_eq!(r.tier, Tier::Simple);
    assert!(!r.bumped);

    let mock = Mock::new([probs([0.0, 0.1, 0.4, 0.5], 0.2)]);
    let r = classifier(&mock)
        .build()
        .unwrap()
        .classify("x")
        .await
        .unwrap();
    assert_eq!(r.tier, Tier::VeryHard);
    assert!(!r.bumped);
}

#[tokio::test(start_paused = true)]
async fn retries_retryable_errors_then_succeeds() {
    let mock = Mock::new([http(503), http(429), probs([1.0, 0.0, 0.0, 0.0], 1.0)]);
    let r = classifier(&mock)
        .retries(2)
        .build()
        .unwrap()
        .classify("x")
        .await
        .unwrap();
    assert_eq!(r.tier, Tier::Simple);
    assert_eq!(mock.calls(), 3);
}

#[tokio::test(start_paused = true)]
async fn non_retryable_error_is_not_retried() {
    let mock = Mock::new([http(401)]);
    let r = classifier(&mock)
        .retries(3)
        .build()
        .unwrap()
        .classify("x")
        .await
        .unwrap();
    assert!(r.degraded);
    assert_eq!(mock.calls(), 1);
    assert!(r.error.unwrap().contains("401"));
}

#[tokio::test(start_paused = true)]
async fn exhausted_retries_fall_back_to_medium() {
    let mock = Mock::new([http(500), http(500), http(500)]);
    let r = classifier(&mock)
        .build()
        .unwrap()
        .classify("x")
        .await
        .unwrap();
    assert_eq!(mock.calls(), 3);
    assert_eq!(r.tier, Tier::Medium);
    assert!(r.degraded);
    assert_eq!(r.probabilities, [0.0; 4]);
}

#[tokio::test(start_paused = true)]
async fn custom_fallback_tier() {
    let mock = Mock::new([http(500)]);
    let c = classifier(&mock)
        .retries(0)
        .on_error(OnError::Fallback(Tier::Hard))
        .build()
        .unwrap();
    assert_eq!(c.classify("x").await.unwrap().tier, Tier::Hard);
}

#[tokio::test(start_paused = true)]
async fn fail_mode_returns_error() {
    let mock = Mock::new([http(500)]);
    let c = classifier(&mock)
        .retries(0)
        .on_error(OnError::Fail)
        .build()
        .unwrap();
    assert!(matches!(
        c.classify("x").await,
        Err(Error::Http { status: 500, .. })
    ));
}

#[tokio::test(start_paused = true)]
async fn timeout_is_retried_then_degrades() {
    let slow = || (Duration::from_secs(10), probs([1.0, 0.0, 0.0, 0.0], 1.0));
    let mock = Mock::delayed([slow(), slow()]);
    let c = classifier(&mock)
        .timeout(Duration::from_millis(500))
        .retries(1)
        .build()
        .unwrap();
    let r = c.classify("x").await.unwrap();
    assert!(r.degraded);
    assert_eq!(mock.calls(), 2);
    assert!(r.error.unwrap().contains("timed out"));
}

#[tokio::test(start_paused = true)]
async fn slow_first_attempt_then_fast_success() {
    let mock = Mock::delayed([
        (Duration::from_secs(10), probs([1.0, 0.0, 0.0, 0.0], 1.0)),
        (Duration::from_millis(50), probs([0.0, 0.0, 1.0, 0.0], 1.0)),
    ]);
    let c = classifier(&mock)
        .timeout(Duration::from_millis(500))
        .build()
        .unwrap();
    let r = c.classify("x").await.unwrap();
    assert!(!r.degraded);
    assert_eq!(r.tier, Tier::Hard);
}

#[tokio::test]
async fn invalid_distribution_is_not_retried() {
    let mock = Mock::new([probs([0.0; 4], 1.0)]);
    let c = classifier(&mock).on_error(OnError::Fail).build().unwrap();
    assert!(matches!(
        c.classify("x").await,
        Err(Error::InvalidResponse(_))
    ));
    assert_eq!(mock.calls(), 1);
}

#[tokio::test]
async fn long_input_is_truncated_head_and_tail() {
    let mock = Mock::new([probs([1.0, 0.0, 0.0, 0.0], 1.0)]);
    let text = format!("HEAD{}TAIL", "x".repeat(10_000));
    classifier(&mock)
        .max_input_chars(100)
        .build()
        .unwrap()
        .classify(&text)
        .await
        .unwrap();
    let sent = &mock.texts()[0];
    assert!(sent.starts_with("HEAD") && sent.ends_with("TAIL"));
    assert!(sent.contains(TRUNCATION_MARKER));
    assert_eq!(
        sent.chars().count(),
        100 + TRUNCATION_MARKER.chars().count()
    );
}

#[tokio::test]
async fn input_is_trimmed_and_empty_input_rejected() {
    let mock = Mock::new([probs([1.0, 0.0, 0.0, 0.0], 1.0)]);
    let c = classifier(&mock).build().unwrap();
    c.classify("  fix typo \n").await.unwrap();
    assert_eq!(mock.texts(), ["fix typo"]);
    assert!(matches!(c.classify(" \n\t").await, Err(Error::EmptyInput)));
    assert_eq!(mock.calls(), 1);
}

#[tokio::test]
async fn classify_many_keeps_order() {
    let mock = Mock::delayed([
        (Duration::from_millis(30), probs([1.0, 0.0, 0.0, 0.0], 1.0)),
        (Duration::from_millis(10), probs([0.0, 1.0, 0.0, 0.0], 1.0)),
        (Duration::ZERO, probs([0.0, 0.0, 0.0, 1.0], 1.0)),
    ]);
    let c = classifier(&mock).build().unwrap();
    let tiers: Vec<Tier> = c
        .classify_many(["a", "b", "c"])
        .await
        .into_iter()
        .map(|r| r.unwrap().tier)
        .collect();
    assert_eq!(tiers, [Tier::Simple, Tier::Medium, Tier::VeryHard]);
}

#[test]
fn build_validates_config() {
    assert!(matches!(
        Classifier::builder().build(),
        Err(Error::Config(_))
    ));
    let mock = Mock::default();
    assert!(classifier(&mock).min_confidence(1.5).build().is_err());
    assert!(classifier(&mock).timeout(Duration::ZERO).build().is_err());
    assert!(classifier(&mock).concurrency(0).build().is_err());
}

#[tokio::test]
async fn result_carries_model_for_lineup() {
    let mock = Mock::new([probs([0.0, 0.0, 0.0, 1.0], 0.9)]);
    let r = classifier(&mock)
        .build()
        .unwrap()
        .classify("x")
        .await
        .unwrap();
    assert_eq!(r.model, "claude-fable-5-1");

    let mock = Mock::new([probs([0.9, 0.1, 0.0, 0.0], 0.9)]);
    let c = classifier(&mock)
        .models(ModelMap::openai())
        .build()
        .unwrap();
    let r = c.classify("x").await.unwrap();
    assert_eq!(r.model, "gpt-6-luna");
    assert_eq!(c.model_for(Tier::Hard), "gpt-6.1-sol");
}

#[tokio::test]
async fn bumped_and_degraded_results_use_final_tier_model() {
    let mock = Mock::new([probs([0.45, 0.4, 0.1, 0.05], 0.3)]);
    let c = classifier(&mock)
        .models(ModelMap::google())
        .build()
        .unwrap();
    let r = c.classify("x").await.unwrap();
    assert_eq!((r.raw_tier, r.tier), (Tier::Simple, Tier::Medium));
    assert_eq!(r.model, "gemini-3.6-flash");

    let mock = Mock::new([http(401)]);
    let models = ModelMap::new("small", "mid", "large", "huge");
    let r = classifier(&mock)
        .models(models)
        .build()
        .unwrap()
        .classify("x")
        .await
        .unwrap();
    assert!(r.degraded);
    assert_eq!(r.model, "mid");
}

#[cfg(feature = "cache")]
#[tokio::test]
async fn cache_serves_repeats_by_normalized_text() {
    let mock = Mock::new([probs([0.0, 0.0, 1.0, 0.0], 0.9)]);
    let c = classifier(&mock).cache(16).build().unwrap();
    let first = c.classify("Refactor  the billing module").await.unwrap();
    let second = c.classify("refactor the BILLING module ").await.unwrap();
    assert_eq!(mock.calls(), 1);
    assert!(!first.cached && second.cached);
    assert_eq!(second.tier, Tier::Hard);
}

#[cfg(feature = "cache")]
#[tokio::test(start_paused = true)]
async fn cache_skips_degraded_results() {
    let mock = Mock::new([http(500), probs([0.0, 0.0, 1.0, 0.0], 0.9)]);
    let c = classifier(&mock).retries(0).cache(16).build().unwrap();
    assert!(c.classify("x").await.unwrap().degraded);
    let r = c.classify("x").await.unwrap();
    assert!(!r.degraded && !r.cached);
    assert_eq!(mock.calls(), 2);
}

#[cfg(feature = "blocking")]
#[test]
fn blocking_wrapper_classifies() {
    let mock = Mock::new([probs([0.0, 1.0, 0.0, 0.0], 1.0)]);
    let c = sortana::blocking::Classifier::new(classifier(&mock).build().unwrap()).unwrap();
    assert_eq!(c.classify("x").unwrap().tier, Tier::Medium);
}
