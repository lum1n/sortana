//! Backend for TypeSafe AI's Jev ("System One") model.
//! API docs: <https://docs.typesafe.ai>.

use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::backend::{Backend, BoxFuture, ClassifyRequest};
use crate::{Error, Result, Tier, TierProbs};

pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
pub const DEFAULT_MODEL: &str = "jev-latest";

/// Id of the single Choice question in each request.
const QUESTION_ID: &str = "complexity";
/// Cap on response body text kept in `Error::Http`.
const MAX_ERROR_BODY: usize = 512;

#[derive(Clone)]
pub struct JevBackend {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl std::fmt::Debug for JevBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevBackend")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl JevBackend {
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| Error::Config(format!("building HTTP client: {e}")))?;
        Ok(Self::with_client(client, api_key))
    }

    /// Use an existing `reqwest::Client`, e.g. to share a connection pool.
    pub fn with_client(client: reqwest::Client, api_key: impl Into<String>) -> Self {
        JevBackend {
            client,
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            model: DEFAULT_MODEL.to_owned(),
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_owned();
        self
    }

    /// JSON body for a request.
    pub fn request_body(&self, req: &ClassifyRequest) -> Value {
        let criteria: Map<String, Value> = Tier::ALL
            .iter()
            .map(|t| (t.as_str().to_owned(), Value::from(req.rubric.criterion(*t))))
            .collect();
        json!({
            "state": req.text,
            "model": self.model,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": req.rubric.instructions,
                    "criteria": criteria,
                }
            }
        })
    }

    async fn call(&self, req: ClassifyRequest) -> Result<TierProbs> {
        let resp = self
            .client
            .post(format!("{}/v1/systemone", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&self.request_body(&req))
            .send()
            .await
            .map_err(transport)?;
        let status = resp.status();
        let body = resp.text().await.map_err(transport)?;
        if !status.is_success() {
            let mut body = body;
            if body.len() > MAX_ERROR_BODY {
                let cut = (0..=MAX_ERROR_BODY)
                    .rev()
                    .find(|&i| body.is_char_boundary(i))
                    .unwrap_or(0);
                body.truncate(cut);
            }
            return Err(Error::Http {
                status: status.as_u16(),
                body,
            });
        }
        parse_response(&body)
    }
}

impl Backend for JevBackend {
    fn classify(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
        Box::pin(self.call(req))
    }
}

fn transport(e: reqwest::Error) -> Error {
    if e.is_timeout() {
        Error::Transport(format!("timed out: {e}"))
    } else {
        Error::Transport(e.to_string())
    }
}

/// Parse a Jev response body into tier probabilities.
///
/// Uses `answers.complexity.probabilities` when present (missing options count
/// as 0). Without probabilities, it falls back to a one-hot on `choice`.
/// Without `confidence`, it uses the top probability.
pub fn parse_response(body: &str) -> Result<TierProbs> {
    let v: Value =
        serde_json::from_str(body).map_err(|e| Error::InvalidResponse(format!("not JSON: {e}")))?;
    let answer = v
        .get("answers")
        .and_then(|a| a.get(QUESTION_ID))
        .ok_or_else(|| Error::InvalidResponse(format!("missing answers.{QUESTION_ID}")))?;

    let mut probabilities = [0.0f32; 4];
    let mut any = false;
    if let Some(map) = answer.get("probabilities").and_then(Value::as_object) {
        for (k, p) in map {
            let (Ok(tier), Some(p)) = (k.parse::<Tier>(), p.as_f64()) else {
                continue;
            };
            probabilities[tier.index()] = p as f32;
            any = true;
        }
    }
    if !any {
        let choice = answer
            .get("choice")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::InvalidResponse("no probabilities and no choice".into()))?;
        let tier = choice
            .parse::<Tier>()
            .map_err(|e| Error::InvalidResponse(e.to_string()))?;
        probabilities[tier.index()] = 1.0;
    }

    let confidence = match answer.get("confidence").and_then(Value::as_f64) {
        Some(c) => c as f32,
        None => probabilities.iter().copied().fold(0.0, f32::max),
    };
    Ok(TierProbs {
        probabilities,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rubric::RUBRIC;

    #[test]
    fn parses_documented_response() {
        let body = r#"{
            "answers": {
                "complexity": {
                    "type": "choice",
                    "choice": "hard",
                    "confidence": 0.82,
                    "probabilities": {"simple": 0.01, "medium": 0.12, "hard": 0.8, "very_hard": 0.07}
                }
            },
            "model": "jev-2026-09",
            "usage": {"input_tokens": 210}
        }"#;
        let p = parse_response(body).unwrap();
        assert_eq!(p.probabilities, [0.01, 0.12, 0.8, 0.07]);
        assert_eq!(p.confidence, 0.82);
        assert_eq!(p.argmax(), Tier::Hard);
    }

    #[test]
    fn missing_probabilities_fall_back_to_choice() {
        let body = r#"{"answers":{"complexity":{"type":"choice","choice":"very_hard"}}}"#;
        let p = parse_response(body).unwrap();
        assert_eq!(p.probabilities, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(p.confidence, 1.0);
    }

    #[test]
    fn missing_options_count_as_zero() {
        let body = r#"{"answers":{"complexity":{"confidence":0.6,"probabilities":{"medium":0.7,"hard":0.3}}}}"#;
        let p = parse_response(body).unwrap();
        assert_eq!(p.probabilities, [0.0, 0.7, 0.3, 0.0]);
    }

    #[test]
    fn rejects_bad_responses() {
        for body in [
            "not json",
            r#"{"answers":{}}"#,
            r#"{"answers":{"complexity":{"choice":"extreme"}}}"#,
            r#"{"answers":{"complexity":{}}}"#,
        ] {
            assert!(
                matches!(parse_response(body), Err(Error::InvalidResponse(_))),
                "{body}"
            );
        }
    }

    #[test]
    fn request_body_shape() {
        let b = JevBackend::with_client(reqwest::Client::new(), "k").model("jev-test");
        let req = ClassifyRequest {
            text: "fix typo".into(),
            rubric: &RUBRIC,
        };
        let v = b.request_body(&req);
        assert_eq!(v["state"], "fix typo");
        assert_eq!(v["model"], "jev-test");
        let q = &v["questions"]["complexity"];
        assert_eq!(q["type"], "choice");
        assert_eq!(q["instructions"], RUBRIC.instructions);
        let criteria = q["criteria"].as_object().unwrap();
        assert_eq!(criteria.len(), 4);
        assert_eq!(criteria["very_hard"], RUBRIC.criteria[3]);
    }
}
