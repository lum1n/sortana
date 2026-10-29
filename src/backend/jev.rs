//! Backend for TypeSafe AI's Jev ("System One") model.
//! API docs: <https://docs.typesafe.ai>.

use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::backend::{Backend, BoxFuture, ClassifyRequest};
use crate::rubric::{EFFORT, Question, TOOLS, VISION};
use crate::{Assessment, Effort, Error, Result, Tier, TierProbs};

pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
pub const DEFAULT_MODEL: &str = "jev-latest";

/// Id of the tier Choice question in each request.
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

    /// JSON body for a request: the tier question, plus the effort, vision
    /// and tools questions when `req.ask` wants them.
    pub fn request_body(&self, req: &ClassifyRequest) -> Value {
        let criteria: Map<String, Value> = Tier::ALL
            .iter()
            .map(|t| (t.as_str().to_owned(), Value::from(req.rubric.criterion(*t))))
            .collect();
        let mut questions = Map::new();
        questions.insert(
            QUESTION_ID.to_owned(),
            json!({
                "type": "choice",
                "instructions": req.rubric.instructions,
                "criteria": criteria,
            }),
        );
        let mut extra = Vec::new();
        if req.ask.effort {
            extra.push(&EFFORT);
        }
        if req.ask.needs {
            extra.extend([&VISION, &TOOLS]);
        }
        for q in extra {
            questions.insert(q.id.to_owned(), choice(q));
        }
        json!({
            "state": req.state(),
            "model": self.model,
            "questions": questions,
        })
    }

    async fn call(&self, req: ClassifyRequest) -> Result<Assessment> {
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
        parse_assessment(&body)
    }
}

impl Backend for JevBackend {
    fn classify(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
        Box::pin(async move { self.call(req).await.map(|a| a.tiers) })
    }

    fn assess(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<Assessment>> {
        Box::pin(self.call(req))
    }
}

fn choice(q: &Question) -> Value {
    let criteria: Map<String, Value> = q
        .options
        .iter()
        .map(|(k, v)| ((*k).to_owned(), Value::from(*v)))
        .collect();
    json!({ "type": "choice", "instructions": q.instructions, "criteria": criteria })
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
    parse_assessment(body).map(|a| a.tiers)
}

/// Parse a Jev response body: the tier answer as in [`parse_response`], plus
/// the effort, vision and tools answers when present. A missing or
/// unreadable extra answer is `None`, not an error.
pub fn parse_assessment(body: &str) -> Result<Assessment> {
    let v: Value =
        serde_json::from_str(body).map_err(|e| Error::InvalidResponse(format!("not JSON: {e}")))?;
    let answers = v.get("answers");
    let answer = answers
        .and_then(|a| a.get(QUESTION_ID))
        .ok_or_else(|| Error::InvalidResponse(format!("missing answers.{QUESTION_ID}")))?;
    let extra = |id: &str| answers.and_then(|a| a.get(id));
    Ok(Assessment {
        tiers: parse_tiers(answer)?,
        effort: extra(EFFORT.id).and_then(|a| {
            let p = options(a, &Effort::ALL.map(Effort::as_str))?;
            Some([p[0], p[1], p[2]])
        }),
        vision: extra(VISION.id).and_then(yes),
        tools: extra(TOOLS.id).and_then(yes),
    })
}

/// Probability of each option in `names`, from `probabilities` or, without
/// them, a one-hot on `choice`. `None` when neither names an option.
fn options(answer: &Value, names: &[&str]) -> Option<Vec<f32>> {
    let mut p = vec![0.0f32; names.len()];
    let mut any = false;
    if let Some(map) = answer.get("probabilities").and_then(Value::as_object) {
        for (i, name) in names.iter().enumerate() {
            if let Some(x) = map.get(*name).and_then(Value::as_f64) {
                p[i] = x as f32;
                any = true;
            }
        }
    }
    if !any {
        let choice = answer.get("choice").and_then(Value::as_str)?;
        let i = names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(choice.trim()))?;
        p[i] = 1.0;
    }
    let sum: f32 = p.iter().sum();
    (sum > 0.0 && sum.is_finite()).then(|| p.iter().map(|x| x / sum).collect())
}

/// Probability of `yes` in a yes/no answer.
fn yes(answer: &Value) -> Option<f32> {
    options(answer, &["yes", "no"]).map(|p| p[0])
}

fn parse_tiers(answer: &Value) -> Result<TierProbs> {
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
    use crate::backend::Ask;
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
    fn parses_extra_answers() {
        let body = r#"{"answers":{
            "complexity":{"choice":"hard","confidence":0.8,"probabilities":{"hard":0.8,"medium":0.2}},
            "effort":{"choice":"high","probabilities":{"low":0.1,"medium":0.3,"high":0.6}},
            "vision":{"choice":"no","probabilities":{"yes":0.25,"no":0.75}},
            "tools":{"choice":"yes"}
        }}"#;
        let a = parse_assessment(body).unwrap();
        assert_eq!(a.tiers.argmax(), Tier::Hard);
        assert_eq!(a.effort, Some([0.1, 0.3, 0.6]));
        assert_eq!(a.vision, Some(0.25));
        assert_eq!(a.tools, Some(1.0));
        // Missing or unusable extras are None.
        let body =
            r#"{"answers":{"complexity":{"choice":"simple"},"effort":{"choice":"extreme"}}}"#;
        let a = parse_assessment(body).unwrap();
        assert_eq!((a.effort, a.vision, a.tools), (None, None, None));
    }

    #[test]
    fn request_body_shape() {
        let b = JevBackend::with_client(reqwest::Client::new(), "k").model("jev-test");
        let req = ClassifyRequest {
            text: "fix typo".into(),
            context: None,
            rubric: &RUBRIC,
            ask: Ask {
                effort: false,
                needs: false,
            },
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
        assert_eq!(v["questions"].as_object().unwrap().len(), 1);
    }

    #[test]
    fn request_body_with_context_and_extras() {
        let b = JevBackend::with_client(reqwest::Client::new(), "k");
        let req = ClassifyRequest {
            text: "fix typo".into(),
            context: Some("Context:\n- images attached: 1".into()),
            rubric: &RUBRIC,
            ask: Ask {
                effort: true,
                needs: true,
            },
        };
        let v = b.request_body(&req);
        assert_eq!(v["state"], "fix typo\n\nContext:\n- images attached: 1");
        let q = v["questions"].as_object().unwrap();
        assert_eq!(q.len(), 4);
        assert_eq!(q["effort"]["criteria"].as_object().unwrap().len(), 3);
        assert_eq!(q["vision"]["criteria"]["yes"], VISION.options[0].1);
        assert_eq!(q["tools"]["type"], "choice");
    }
}
