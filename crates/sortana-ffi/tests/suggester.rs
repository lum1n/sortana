//! `Suggester` against a local one-shot HTTP server (no network).

use sortana_ffi::{SortanaError, Suggester, Tier, lineup_for_harness};
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
