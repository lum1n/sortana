//! Live test against Jev. Run with `TYPESAFE_API_KEY=... cargo test -- --ignored`.

use sortana::{Classifier, OnError, Tier};

#[tokio::test]
#[ignore = "needs TYPESAFE_API_KEY and network"]
async fn live_jev_classifies_extremes() {
    let key = std::env::var("TYPESAFE_API_KEY").expect("TYPESAFE_API_KEY is not set");
    let c = Classifier::builder()
        .jev(key)
        .on_error(OnError::Fail)
        .build()
        .unwrap();

    let easy = c.classify("fix typo in README").await.unwrap();
    assert!(!easy.degraded);
    assert!(easy.tier <= Tier::Medium, "{easy:?}");
    let sum: f32 = easy.probabilities.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3, "{easy:?}");

    let hard = c
        .classify(
            "Design a multi-region distributed event-processing platform spanning five \
             existing systems, with exactly-once delivery, schema evolution and a failure analysis",
        )
        .await
        .unwrap();
    assert!(hard.tier >= Tier::Hard, "{hard:?}");
}
