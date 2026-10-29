//! `Suggester` against a local one-shot HTTP server (no network).

use sortana_ffi::{
    Decision, Effort, SortanaError, Suggester, SuggesterOptions, TaskContext, Tier,
    lineup_for_harness, model_info,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Answer one request with `status` and `body`; returns the base URL.
async fn serve_once(status: &'static str, body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let n = sock.read(&mut chunk).await.unwrap();
            buf.extend_from_slice(&chunk[..n]);
            let text = String::from_utf8_lossy(&buf);
            if let Some(end) = text.find("\r\n\r\n") {
                let len = text[..end]
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                if buf.len() >= end + 4 + len {
                    break;
                }
            }
            if n == 0 {
                break;
            }
        }
        let resp = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        sock.write_all(resp.as_bytes()).await.unwrap();
    });
    url
}

const HARD: &str = r#"{"answers":{"complexity":{"type":"choice","choice":"hard","confidence":0.9,
    "probabilities":{"simple":0.0,"medium":0.1,"hard":0.9,"very_hard":0.0}}}}"#;

#[tokio::test]
async fn suggests_from_the_apps_models() {
    let url = serve_once("200 OK", HARD).await;
    let s = Suggester::new("k".into(), "claude".into(), Some(url)).unwrap();
    assert_eq!(s.lineup(), "anthropic");
    let r = s
        .suggest(
            "Migrate billing to event sourcing".into(),
            vec!["opus".into(), "sonnet".into(), "haiku".into()],
        )
        .await
        .unwrap();
    assert_eq!(r.tier, Tier::Hard);
    assert_eq!(r.model.as_deref(), Some("opus"));
    assert!(!r.bumped);
    assert_eq!(r.probabilities.len(), 4);
}

const SIMPLE: &str = r#"{"answers":{"complexity":{"type":"choice","choice":"simple","confidence":0.9,
    "probabilities":{"simple":0.9,"medium":0.1,"hard":0.0,"very_hard":0.0}}}}"#;

const HARD_WITH_SIGNALS: &str = r#"{"answers":{
    "complexity":{"choice":"hard","confidence":0.9,"probabilities":{"medium":0.1,"hard":0.9}},
    "effort":{"choice":"high","probabilities":{"low":0.0,"medium":0.2,"high":0.8}},
    "vision":{"choice":"no","probabilities":{"yes":0.1,"no":0.9}},
    "tools":{"choice":"yes","probabilities":{"yes":0.9,"no":0.1}}}}"#;

#[tokio::test]
async fn suggestion_carries_effort_needs_and_info() {
    let url = serve_once("200 OK", HARD_WITH_SIGNALS).await;
    let s = Suggester::new("k".into(), "anthropic".into(), Some(url)).unwrap();
    let r = s
        .suggest(
            "Find the deadlock".into(),
            vec!["opus".into(), "sonnet".into()],
        )
        .await
        .unwrap();
    assert_eq!(r.effort, Some(Effort::High));
    assert_eq!(r.reasoning_effort.as_deref(), Some("high"));
    assert!(r.needs.tools && !r.needs.vision);
    let info = r.model_info.as_ref().unwrap();
    assert_eq!(info.id, "anthropic/claude-opus-5-5");
    assert_eq!(r.fits, Some(true));
    assert_eq!(r.raw_tier, Tier::Hard);

    // Escalating from opus among the app's models: nothing stronger listed.
    assert_eq!(
        s.escalate(r.clone(), vec!["opus".into(), "sonnet".into()]),
        None
    );
    // With fable listed, it is the next model up.
    let up = s.escalate(r, vec!["opus".into(), "fable".into()]).unwrap();
    assert_eq!(
        (up.tier, up.model.as_deref()),
        (Tier::VeryHard, Some("fable"))
    );
    assert_eq!(up.reasoning_effort.as_deref(), Some("high"));
}

#[tokio::test]
async fn context_and_options_shape_the_suggestion() {
    let url = serve_once("200 OK", SIMPLE).await;
    let s = Suggester::with_options(
        "k".into(),
        "amazon".into(),
        SuggesterOptions {
            base_url: Some(url),
            decision: Some(Decision::Confidence),
            ..Default::default()
        },
    )
    .unwrap();
    let ctx = TaskContext {
        images: 1,
        ..Default::default()
    };
    // Amazon's simple model (Nova Micro) has no vision; medium's does.
    let r = s
        .suggest_with("What is wrong in this screenshot?".into(), ctx, vec![])
        .await
        .unwrap();
    assert_eq!(r.tier, Tier::Medium);
    assert!(r.refit && r.needs.vision);
    assert_eq!(r.model.as_deref(), Some("nova-lite"));

    let url = serve_once("200 OK", HARD).await;
    let s = Suggester::with_options(
        "k".into(),
        "anthropic".into(),
        SuggesterOptions {
            base_url: Some(url),
            max_tier: Some(Tier::Medium),
            ..Default::default()
        },
    )
    .unwrap();
    let r = s.suggest("x".into(), vec![]).await.unwrap();
    assert_eq!(r.tier, Tier::Medium);
    assert!(r.clamped);
    assert_eq!(s.escalate(r, vec![]), None);

    assert!(matches!(
        Suggester::with_options(
            "k".into(),
            "anthropic".into(),
            SuggesterOptions {
                min_tier: Some(Tier::Hard),
                max_tier: Some(Tier::Simple),
                ..Default::default()
            },
        ),
        Err(SortanaError::Config { .. })
    ));
}

#[test]
fn looks_up_model_info() {
    let m = model_info("openrouter/openai/gpt-6-luna".into()).unwrap();
    assert_eq!(m.id, "openai/gpt-6-luna");
    assert!(m.input_cost.is_some() && m.vision);
    assert_eq!(model_info("nope".into()), None);
}

#[tokio::test]
async fn empty_list_gives_native_id() {
    let url = serve_once("200 OK", HARD).await;
    let s = Suggester::new("k".into(), "openai".into(), Some(url)).unwrap();
    let r = s.suggest("refactor".into(), vec![]).await.unwrap();
    assert_eq!(r.model.as_deref(), Some("gpt-6.1-sol"));
}

#[tokio::test]
async fn backend_failure_is_an_error() {
    let url = serve_once("401 Unauthorized", r#"{"error":"bad key"}"#).await;
    let s = Suggester::new("k".into(), "anthropic".into(), Some(url)).unwrap();
    let e = s.suggest("x".into(), vec![]).await.unwrap_err();
    assert!(matches!(e, SortanaError::Backend { .. }), "{e:?}");
    let e = s.suggest("  ".into(), vec![]).await.unwrap_err();
    assert!(matches!(e, SortanaError::EmptyInput));
}

#[test]
fn lineups() {
    assert!(matches!(
        Suggester::new("k".into(), "nope".into(), None),
        Err(SortanaError::UnknownLineup { .. })
    ));
    assert_eq!(
        lineup_for_harness("claude".into()).as_deref(),
        Some("anthropic")
    );
    assert_eq!(
        lineup_for_harness("Codex".into()).as_deref(),
        Some("openai")
    );
    assert_eq!(lineup_for_harness("cursor".into()), None);
}
