//! `JevBackend` against a local one-shot HTTP server (no network).

use sortana::{Backend, ClassifyRequest, Error, JevBackend, Tier};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve one request with `status` and `body`. Returns the base URL and a
/// handle that yields the raw request text.
async fn serve_once(
    status: &'static str,
    body: &'static str,
) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        // Read headers, then the body by Content-Length.
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
        String::from_utf8(buf).unwrap()
    });
    (url, handle)
}

fn request(text: &str) -> ClassifyRequest {
    ClassifyRequest {
        text: text.into(),
        context: None,
        rubric: &sortana::rubric::RUBRIC,
        ask: sortana::Ask {
            effort: true,
            needs: true,
        },
    }
}

#[tokio::test]
async fn posts_to_systemone_and_parses_answer() {
    let (url, server) = serve_once(
        "200 OK",
        r#"{"answers":{"complexity":{"type":"choice","choice":"medium","confidence":0.71,
            "probabilities":{"simple":0.2,"medium":0.71,"hard":0.08,"very_hard":0.01}}},
            "model":"jev-test","usage":{"input_tokens":180}}"#,
    )
    .await;
    let backend = JevBackend::with_client(reqwest::Client::new(), "secret-key").base_url(url + "/");
    let probs = backend.classify(request("add a flag")).await.unwrap();
    assert_eq!(probs.argmax(), Tier::Medium);
    assert_eq!(probs.confidence, 0.71);

    let raw = server.await.unwrap();
    assert!(raw.starts_with("POST /v1/systemone HTTP/1.1"), "{raw}");
    assert!(
        raw.to_ascii_lowercase()
            .contains("authorization: bearer secret-key")
    );
    let body: serde_json::Value =
        serde_json::from_str(raw.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["state"], "add a flag");
    assert_eq!(body["model"], "jev-latest");
}

#[tokio::test]
async fn maps_http_errors() {
    let (url, _server) = serve_once("429 Too Many Requests", r#"{"error":"rate limited"}"#).await;
    let backend = JevBackend::with_client(reqwest::Client::new(), "k").base_url(url);
    let err = backend.classify(request("x")).await.unwrap_err();
    assert!(matches!(&err, Error::Http { status: 429, body } if body.contains("rate limited")));
    assert!(err.is_retryable());

    let (url, _server) = serve_once("401 Unauthorized", "{}").await;
    let backend = JevBackend::with_client(reqwest::Client::new(), "k").base_url(url);
    let err = backend.classify(request("x")).await.unwrap_err();
    assert!(matches!(err, Error::Http { status: 401, .. }));
    assert!(!err.is_retryable());

    // Jev's documented overload status.
    let (url, _server) = serve_once("529 Overloaded", r#"{"detail":"overloaded"}"#).await;
    let backend = JevBackend::with_client(reqwest::Client::new(), "k").base_url(url);
    let err = backend.classify(request("x")).await.unwrap_err();
    assert!(matches!(err, Error::Http { status: 529, .. }));
    assert!(err.is_retryable());

    // Validation failures (400, 413, 415, 422) are permanent.
    let (url, _server) = serve_once("422 Unprocessable Entity", r#"{"detail":[]}"#).await;
    let backend = JevBackend::with_client(reqwest::Client::new(), "k").base_url(url);
    let err = backend.classify(request("x")).await.unwrap_err();
    assert!(!err.is_retryable());
}

#[tokio::test]
async fn connection_refused_is_retryable_transport_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let backend = JevBackend::with_client(reqwest::Client::new(), "k").base_url(url);
    let err = backend.classify(request("x")).await.unwrap_err();
    assert!(matches!(err, Error::Transport(_)), "{err:?}");
    assert!(err.is_retryable());
}
