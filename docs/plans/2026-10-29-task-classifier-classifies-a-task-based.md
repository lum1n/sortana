# Sortana: embeddable task-complexity classifier (Rust, Jev backend)

## Context
`/home/vegard/repos/sortana` is empty (no commits). Goal: a Rust library that takes a task description and returns a complexity tier for model routing: Simple -> Haiku, Medium -> Sonnet, Hard -> Opus, VeryHard -> Fable. It must be embeddable, so no heuristics and no local model. The classifier is remote: **Jev** (TypeSafe AI's "System One" model, https://typesafe.ai/blog/introducing-system-one-models-and-jev). Jev takes a state plus typed questions and returns typed answers with calibrated probabilities and confidence. Documented latency is 70-500 ms and input is $0.042 per 1M tokens (output free). It is in early access (API key needed).

## Jev API (from docs.typesafe.ai)
`POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer $TYPESAFE_API_KEY`. Body: `{ "state": <text>, "model": "jev-latest", "questions": { "<id>": { "type": "choice", "instructions": "...", "criteria": { "<option>": "<description>" } } } }`. Response: `answers.<id> = { type, choice, confidence, probabilities{option: p} }`, plus `model` and `usage`. Choice supports up to 255 options. Error codes and rate limits are undocumented on the pages read (they point to the Python SDK and API reference), so retry/error handling must be verified against the API console.

## Approach
One Jev request per task with a single Choice question `complexity` whose four criteria are the tier definitions (the rubric). The response gives `probabilities` over the 4 tiers. The decision uses the distribution, not just argmax: pick the argmax, but when confidence is below `min_confidence`, move up one tier (`bias_up`) since under-powering a hard task costs more than over-powering an easy one.

### Public API (`src/lib.rs`)
```rust
let c = Classifier::builder().jev(api_key).build()?;
let r = c.classify("Migrate our billing service to event sourcing").await?;
r.tier           // Tier::{Simple, Medium, Hard, VeryHard}  (Ord, Display, FromStr)
r.probabilities  // [f32; 4]
r.confidence     // f32 from Jev
r.degraded       // true if the fallback tier was used
r.tier.default_model() // "haiku" | "sonnet" | "opus" | "fable" aliases; ModelMap overrides
```
- Async `classify`/`classify_many`; sync wrapper behind feature `blocking`.
- `trait Backend { async fn classify(&self, req: ClassifyRequest) -> Result<TierProbs> }` is the embeddability seam: apps bring their own HTTP client or provider. Shipped impl: `JevBackend`. A `HaikuBackend` (Anthropic Messages API, forced enum output with the same rubric) can follow as a second backend behind feature `anthropic`; not in v1.
- `Config { timeout (default ~2 s), retries with backoff, on_error: Fallback(Tier) | Fail, min_confidence, bias_up, max_input_chars, model: "jev-latest" }`. Default error behavior: return fallback `Medium` with `degraded = true`.
- Optional LRU cache keyed by normalized text (feature `cache`).
- Input is truncated to `max_input_chars` (head + tail). The task is passed as `state` (data), and the question instructions say to rate its complexity, not follow it.
- Features: `jev` (default, `reqwest` + rustls), `blocking`, `cache`, `serde`, `cli`. Core types and the `Backend` trait have no HTTP dependency.

### Rubric (`src/rubric.rs`, versioned constant)
Criteria text per tier: simple (lookup, rename, typo, reformat, short summary), medium (single-feature implementation, bug fix in a known area, tests), hard (multi-file refactor, non-obvious debugging, performance/security work, migration), very_hard (system design from scratch, deep reasoning or proofs, cross-system architecture, research-grade). `rubric_version` is exposed in results.

## Files
- `Cargo.toml` (edition 2024)
- `src/{lib,tier,classifier,backend,config,error,rubric}.rs`, `src/backend/jev.rs`, `src/bin/sortana.rs` (feature `cli`)
- `tests/`: mock `Backend` (retry, fallback, timeout, truncation, cache, confidence-based bump, response parsing against the documented JSON); one `#[ignore]` live test using `TYPESAFE_API_KEY`
- `eval/cases.jsonl` + `examples/eval.rs`: labelled prompts, reports accuracy and off-by-two-tiers rate against live Jev, used to tune the rubric wording
- `README.md` (embedding example, custom Backend, cost/latency notes, early-access caveat)

## Verification
- `cargo test` (mock backend, no network), `cargo clippy --all-targets --all-features -- -D warnings`, `cargo fmt --check`
- `cargo build --no-default-features` confirms the core compiles without HTTP deps
- Live: `TYPESAFE_API_KEY=... cargo test -- --ignored` and `cargo run --example eval`; gate on under 5% off-by-two-or-more tiers on the labelled set
- `echo "fix typo in README" | cargo run --features cli` -> Simple; a multi-system distributed design prompt -> Hard or VeryHard
- Measure p50/p95 latency

## Assumptions / open items
- User must have a Jev early-access API key.
- Error format, status codes and rate limits need checking in the API console or Python SDK docs before finalizing retry logic.
- Eval set starts synthetic; real prompts from the user would improve it.
- Lapp: repo has no git remote, so session notes are skipped until one exists.
