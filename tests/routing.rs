//! Effort, needs, context, tier limits, cost routing and escalation against
//! a mock backend that answers the extra questions (no network).

use std::sync::{Arc, Mutex};

use sortana::{
    Assessment, Backend, BoxFuture, Classifier, ClassifyRequest, Context, Decision, Effort, Error,
    Lineup, ModelMap, Result, Tier, TierProbs,
};

/// Answers every request with `answer` and records the requests.
#[derive(Clone)]
struct Mock {
    answer: Result<Assessment, u16>,
    seen: Arc<Mutex<Vec<ClassifyRequest>>>,
}

impl Mock {
    fn new(tiers: [f32; 4], effort: Option<[f32; 3]>, vision: f32, tools: f32) -> Self {
        Mock {
            answer: Ok(Assessment {
                tiers: TierProbs {
                    probabilities: tiers,
                    confidence: 0.9,
                },
                effort,
                vision: Some(vision),
                tools: Some(tools),
            }),
            seen: Arc::default(),
        }
    }

    fn tier(t: Tier) -> Self {
        let mut p = [0.0; 4];
        p[t.index()] = 1.0;
        Self::new(p, None, 0.0, 0.0)
    }

    fn failing(status: u16) -> Self {
        Mock {
            answer: Err(status),
            seen: Arc::default(),
        }
    }

    fn requests(&self) -> Vec<ClassifyRequest> {
        self.seen.lock().unwrap().clone()
    }
}

impl Backend for Mock {
    fn classify(&self, _req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
        unreachable!("the classifier calls assess")
    }

    fn assess(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<Assessment>> {
        self.seen.lock().unwrap().push(req);
        let answer = self.answer.map_err(|status| Error::Http {
            status,
            body: String::new(),
        });
        Box::pin(async move { answer })
    }
}

fn builder(mock: &Mock) -> sortana::ClassifierBuilder {
    Classifier::builder().backend(mock.clone()).retries(0)
}

#[tokio::test]
async fn effort_maps_to_the_models_setting() {
    let mock = Mock::new([0.0, 0.1, 0.9, 0.0], Some([0.1, 0.2, 0.7]), 0.1, 0.9);
    let c = builder(&mock).build().unwrap();
    let r = c.classify("Find the race in the scheduler").await.unwrap();
    assert_eq!(r.tier, Tier::Hard);
    assert_eq!(r.effort, Some(Effort::High));
    assert_eq!(r.reasoning_effort, Some("high"));
    assert_eq!(r.info.unwrap().id, "anthropic/claude-opus-5-5");
    assert!(r.needs.tools && !r.needs.vision);
    assert_eq!(r.fits, Some(true));
    // DeepSeek's effort setting has low/high/max: medium rounds up.
    let mock = Mock::new([0.0, 0.0, 1.0, 0.0], Some([0.0, 1.0, 0.0]), 0.0, 0.0);
    let c = builder(&mock)
        .models(ModelMap::preset("deepseek").unwrap())
        .build()
        .unwrap();
    let r = c.classify("x").await.unwrap();
    assert_eq!(r.reasoning_effort, Some("high"));
}

#[tokio::test]
async fn questions_and_context_reach_the_backend() {
    let mock = Mock::tier(Tier::Medium);
    let c = builder(&mock).ask_effort(false).build().unwrap();
    let ctx = Context {
        files: Some(12),
        turns: Some(3),
        ..Default::default()
    };
    c.classify_with("Add pagination", &ctx).await.unwrap();
    c.classify("Add pagination").await.unwrap();
    let reqs = mock.requests();
    assert!(!reqs[0].ask.effort && reqs[0].ask.needs);
    assert_eq!(
        reqs[0].state(),
        "Add pagination\n\nContext:\n- files attached or in scope: 12\n- conversation so far: 3 turns"
    );
    assert_eq!(reqs[1].context, None);
    assert_eq!(reqs[1].state(), "Add pagination");
}

#[tokio::test]
async fn images_move_to_a_model_with_vision() {
    // DeepSeek: v4.1 flash (vision) for simple/medium, v4 pro (text only) above.
    let mock = Mock::tier(Tier::Hard);
    let c = builder(&mock)
        .models(ModelMap::preset("deepseek").unwrap())
        .build()
        .unwrap();
    let ctx = Context {
        images: 1,
        ..Default::default()
    };
    let r = c
        .classify_with("Why does this chart dip?", &ctx)
        .await
        .unwrap();
    assert_eq!(r.raw_tier, Tier::Hard);
    assert_eq!(r.tier, Tier::Medium);
    assert_eq!(r.model, "deepseek-flash");
    assert!(r.refit && r.needs.vision);
    assert_eq!(r.fits, Some(true));
}

#[tokio::test]
async fn backend_vision_answer_counts_too() {
    let mock = Mock::new([0.0, 0.0, 1.0, 0.0], None, 0.8, 0.0);
    let c = builder(&mock)
        .models(ModelMap::preset("deepseek").unwrap())
        .build()
        .unwrap();
    let r = c.classify("Look at the screenshot").await.unwrap();
    assert!(r.needs.vision);
    assert_eq!(r.tier, Tier::Medium);
}

#[tokio::test]
async fn nothing_fits_keeps_the_tier() {
    let mock = Mock::tier(Tier::Medium);
    let c = builder(&mock).build().unwrap();
    let ctx = Context {
        prompt_tokens: Some(5_000_000),
        ..Default::default()
    };
    let r = c.classify_with("Summarize the logs", &ctx).await.unwrap();
    assert_eq!(r.tier, Tier::Medium);
    assert_eq!(r.fits, Some(false));
    assert!(!r.refit);
    // Unknown models count as fitting, with unknown fit.
    let c = builder(&mock)
        .models(ModelMap::new("a", "b", "c", "d"))
        .build()
        .unwrap();
    let r = c.classify_with("Summarize the logs", &ctx).await.unwrap();
    assert_eq!((r.tier, r.fits, r.info), (Tier::Medium, None, None));
}

#[tokio::test]
async fn tier_limits_clamp() {
    let mock = Mock::tier(Tier::VeryHard);
    let c = builder(&mock).max_tier(Tier::Hard).build().unwrap();
    let r = c.classify("Design a database").await.unwrap();
    assert_eq!((r.tier, r.raw_tier), (Tier::Hard, Tier::VeryHard));
    assert!(r.clamped && !r.refit);
    assert_eq!(r.model, "claude-opus-5-5");

    let mock = Mock::tier(Tier::Simple);
    let c = builder(&mock).min_tier(Tier::Medium).build().unwrap();
    let r = c.classify("Fix a typo").await.unwrap();
    assert_eq!(r.tier, Tier::Medium);
    assert!(r.clamped);

    // The fallback tier is clamped too.
    let mock = Mock::failing(503);
    let c = builder(&mock).max_tier(Tier::Simple).build().unwrap();
    let r = c.classify("anything").await.unwrap();
    assert!(r.degraded && r.clamped);
    assert_eq!(r.tier, Tier::Simple);

    assert!(matches!(
        Classifier::builder()
            .backend(Mock::tier(Tier::Simple))
            .min_tier(Tier::Hard)
            .max_tier(Tier::Medium)
            .build(),
        Err(Error::Config(_))
    ));
}

#[tokio::test]
async fn degraded_results_still_honour_context_needs() {
    let mock = Mock::failing(500);
    let c = builder(&mock)
        .models(ModelMap::preset("deepseek").unwrap())
        .on_error(sortana::OnError::Fallback(Tier::Hard))
        .build()
        .unwrap();
    let ctx = Context {
        images: 2,
        ..Default::default()
    };
    let r = c.classify_with("x", &ctx).await.unwrap();
    assert!(r.degraded && r.refit);
    assert_eq!(r.tier, Tier::Medium);
    assert_eq!(r.effort, None);
}

#[tokio::test]
async fn cost_decision_weighs_prices() {
    // Anthropic blended prices: 0.2, 4, 8, 20 per Mtok.
    // E(simple) = 0.2 + 2 * (0.3 * 4 + 0.1 * 8) = 4.2 beats E(medium) = 5.6.
    let mock = Mock::new([0.6, 0.3, 0.1, 0.0], None, 0.0, 0.0);
    let c = builder(&mock).decision(Decision::COST).build().unwrap();
    let r = c.classify("x").await.unwrap();
    assert_eq!(r.tier, Tier::Simple);
    // A likely hard task is cheapest on the hard model.
    let mock = Mock::new([0.1, 0.2, 0.7, 0.0], None, 0.0, 0.0);
    let c = builder(&mock).decision(Decision::COST).build().unwrap();
    assert_eq!(c.classify("x").await.unwrap().tier, Tier::Hard);
    // A large miss penalty moves a lopsided decision up:
    // E(simple) = 0.2 + 2 * 0.2 * 4 = 1.8 < 4, but 0.2 + 5 * 0.2 * 4 = 4.2 > 4.
    let mock = Mock::new([0.8, 0.2, 0.0, 0.0], None, 0.0, 0.0);
    let cheap = builder(&mock).decision(Decision::COST).build().unwrap();
    let careful = builder(&mock)
        .decision(Decision::Cost { miss_penalty: 4.0 })
        .build()
        .unwrap();
    assert_eq!(cheap.classify("x").await.unwrap().tier, Tier::Simple);
    assert_eq!(careful.classify("x").await.unwrap().tier, Tier::Medium);
    assert!(
        builder(&mock)
            .decision(Decision::Cost { miss_penalty: -1.0 })
            .build()
            .is_err()
    );
}

#[tokio::test]
async fn cost_decision_without_prices_uses_confidence() {
    let mock = Mock::new([0.6, 0.3, 0.1, 0.0], None, 0.0, 0.0);
    let c = builder(&mock)
        .decision(Decision::COST)
        .models(ModelMap::new("a", "b", "c", "d"))
        .build()
        .unwrap();
    assert_eq!(c.classify("x").await.unwrap().tier, Tier::Simple);
    let mock = Mock::new([0.0, 0.2, 0.8, 0.0], None, 0.0, 0.0);
    let c = builder(&mock)
        .decision(Decision::COST)
        .models(ModelMap::new("a", "b", "c", "d"))
        .build()
        .unwrap();
    assert_eq!(c.classify("x").await.unwrap().tier, Tier::Hard);
}

#[tokio::test]
async fn cost_decision_respects_needs_and_limits() {
    let mock = Mock::new([0.0, 0.0, 0.0, 1.0], None, 0.0, 0.0);
    let c = builder(&mock)
        .decision(Decision::COST)
        .max_tier(Tier::Hard)
        .build()
        .unwrap();
    let r = c.classify("x").await.unwrap();
    assert_eq!(r.tier, Tier::Hard);
    assert!(!r.clamped, "cost picks within the limits");
}

#[tokio::test]
async fn escalation_skips_repeated_models_and_stops_at_the_top() {
    let mock = Mock::new([1.0, 0.0, 0.0, 0.0], Some([0.0, 1.0, 0.0]), 0.0, 0.0);
    let c = builder(&mock).build().unwrap();
    let r = c.classify("x").await.unwrap();
    let up = c.escalate(&r).unwrap();
    assert_eq!(
        (up.tier, up.model.as_str()),
        (Tier::Medium, "claude-sonnet-5-5")
    );
    assert_eq!(up.raw_tier, Tier::Simple);
    assert_eq!(up.reasoning_effort, Some("medium"));
    let top = c.escalate(&c.escalate(&up).unwrap()).unwrap();
    assert_eq!(top.tier, Tier::VeryHard);
    assert!(c.escalate(&top).is_none());

    // xAI repeats grok-4.3 for simple and medium.
    let c = builder(&mock)
        .models(ModelMap::preset("xai").unwrap())
        .build()
        .unwrap();
    let r = c.classify("x").await.unwrap();
    let up = c.escalate(&r).unwrap();
    assert_eq!(up.tier, Tier::Hard);
    assert_ne!(up.model, r.model);

    // Never above max_tier.
    let c = builder(&mock).max_tier(Tier::Simple).build().unwrap();
    let r = c.classify("x").await.unwrap();
    assert!(c.escalate(&r).is_none());
}

#[tokio::test]
async fn escalation_prefers_models_that_fit() {
    // With an image, the hard tier's model (DeepSeek v4 pro) lacks vision.
    let mock = Mock::tier(Tier::Medium);
    let models = ModelMap::new("small", "mid", "big-text", "big-vision")
        .with_info(
            Tier::Medium,
            sortana::ModelInfo::get("deepseek/deepseek-v4.1-flash"),
        )
        .with_info(
            Tier::Hard,
            sortana::ModelInfo::get("deepseek/deepseek-v4-pro"),
        )
        .with_info(
            Tier::VeryHard,
            sortana::ModelInfo::get("anthropic/claude-fable-5-1"),
        );
    let c = builder(&mock).models(models).build().unwrap();
    let ctx = Context {
        images: 1,
        ..Default::default()
    };
    let r = c.classify_with("x", &ctx).await.unwrap();
    assert_eq!(r.tier, Tier::Medium);
    let up = c.escalate(&r).unwrap();
    assert_eq!(up.model, "big-vision");
    assert_eq!(up.fits, Some(true));
}

#[tokio::test]
async fn classify_for_routes_to_app_models() {
    let mock = Mock::tier(Tier::VeryHard);
    let c = builder(&mock).build().unwrap();
    let a = Lineup::get("anthropic").unwrap();
    let models = a.models_among(&["opus", "sonnet", "haiku"]).unwrap();
    let r = c
        .classify_for("x", &Context::default(), &models)
        .await
        .unwrap();
    // No `fable` entry: the nearest tier's model, with opus's info.
    assert_eq!(r.model, "opus");
    assert_eq!(r.info, a.info(Tier::Hard));
    assert!(c.escalate_for(&r, &models).is_none());
}

#[tokio::test]
async fn classify_only_backends_still_work() {
    struct Plain;
    impl Backend for Plain {
        fn classify(&self, _req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
            Box::pin(async {
                Ok(TierProbs {
                    probabilities: [0.0, 1.0, 0.0, 0.0],
                    confidence: 1.0,
                })
            })
        }
    }
    let c = Classifier::builder().backend(Plain).build().unwrap();
    let ctx = Context {
        tools: Some(true),
        ..Default::default()
    };
    let r = c.classify_with("x", &ctx).await.unwrap();
    assert_eq!((r.tier, r.effort), (Tier::Medium, None));
    assert!(r.needs.tools && !r.needs.vision);
}
