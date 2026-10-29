# sortana

Embeddable task-complexity classifier for model routing. Give it a task
description and it returns a tier:

| Tier       | Default model |
|------------|---------------|
| `Simple`   | `haiku`       |
| `Medium`   | `sonnet`      |
| `Hard`     | `opus`        |
| `VeryHard` | `fable`       |

Classification is remote, with no heuristics and no local model. The default
backend is [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One" model. Each task becomes one Jev request with a single
Choice question. Its four criteria are the tier definitions in
[`src/rubric.rs`](src/rubric.rs). Jev returns a calibrated probability per tier.

> Jev is in early access and needs an API key (`TYPESAFE_API_KEY`).

## Usage

```rust
let c = sortana::Classifier::builder()
    .jev(std::env::var("TYPESAFE_API_KEY")?)
    .build()?;

let r = c.classify("Migrate our billing service to event sourcing").await?;
r.tier;              // Tier::Hard
r.probabilities;     // [f32; 4], indexed like Tier::ALL
r.confidence;        // Jev's confidence
r.degraded;          // true if the backend failed and the fallback tier was used
c.model_for(r.tier); // "opus", or your ModelMap override
```

Map tiers to concrete model ids:

```rust
use sortana::{ModelMap, Tier};

let c = sortana::Classifier::builder()
    .jev(key)
    .models(ModelMap::default()
        .with(Tier::Simple, "claude-haiku-5-5")
        .with(Tier::Medium, "claude-sonnet-5-5")
        .with(Tier::Hard, "claude-opus-5-5")
        .with(Tier::VeryHard, "claude-fable-5-1"))
    .build()?;
```

`classify_many` classifies a batch concurrently and keeps input order. With
feature `blocking`, `sortana::blocking::Classifier` wraps the async classifier
for sync code.

### How the tier is chosen

1. The input is trimmed. If it is longer than `max_input_chars` (default 8000),
   its head and tail are kept.
2. The task goes to Jev as the `state`, which is data, not a prompt. The question
   tells Jev to rate the task, not to follow it.
3. The tier is the argmax of the probabilities. If Jev's confidence is below
   `min_confidence` (default 0.5) and `bias_up` is on (default), the tier moves up
   one. Under-powering a hard task costs more than over-powering an easy one.
4. Timeouts, transport errors, HTTP 408/429/5xx and backend errors marked
   retryable are retried `retries` times (default 2), with exponential backoff.
   If every attempt fails, `on_error` decides the result. The default,
   `OnError::Fallback(Tier::Medium)`, returns `Medium` with `degraded = true`.
   `OnError::Fail` returns the error instead.

`timeout` (default 2 s) applies to each attempt. Every result carries
`rubric_version` so you can tell which rubric produced it.

## Custom backends

The `Backend` trait is the embedding seam. Use it to bring your own HTTP client
or provider. The classifier still handles timeouts, retries, fallback, caching
and the final decision.

```rust
use sortana::{Backend, BoxFuture, ClassifyRequest, Result, TierProbs};

struct MyBackend;

impl Backend for MyBackend {
    fn classify(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
        Box::pin(async move {
            // req.text is the (truncated) task; req.rubric has the instructions
            // and one criterion per tier.
            Ok(TierProbs { probabilities: [0.1, 0.7, 0.2, 0.0], confidence: 0.7 })
        })
    }
}

let c = sortana::Classifier::builder().backend(MyBackend).build()?;
```

To reuse a connection pool or point at another endpoint, use
`JevBackend::with_client(client, key).base_url(url)`.

## Features

| Feature    | Default | Adds |
|------------|---------|------|
| `jev`      | yes     | `JevBackend` (`reqwest` + rustls) |
| `blocking` |         | `sortana::blocking::Classifier` |
| `cache`    |         | LRU cache keyed by normalized text: `.cache(capacity)` |
| `serde`    |         | `Serialize` for `Classification`, `Serialize`/`Deserialize` for `Tier` and `ModelMap` |
| `cli`      |         | the `sortana` binary |

With `--no-default-features`, the core types and the `Backend` trait build with
no HTTP dependencies. The classifier uses Tokio's timer, so call it from within a
Tokio runtime.

## CLI

```sh
export TYPESAFE_API_KEY=...
echo "fix typo in README" | cargo run --features cli
cargo run --features cli -- --json "design a multi-region event platform"
```

## Cost and latency

According to TypeSafe AI, Jev responds in 70-500 ms and costs $0.042 per million
input tokens, with free output. The rubric adds a few hundred input tokens per request.

## Evaluation

`eval/cases.jsonl` holds labelled tasks. Run them against live Jev with:

```sh
TYPESAFE_API_KEY=... cargo run --example eval
```

The eval reports accuracy, the share of tasks that are off by two or more tiers
(the target is under 5%), p50/p95 latency and a confusion matrix. Re-run it after
any rubric change and bump `RUBRIC_VERSION`.

## Development

```sh
cargo test --all-features                       # mock backend + local HTTP, no network
TYPESAFE_API_KEY=... cargo test -- --ignored    # live Jev
cargo clippy --all-targets --all-features -- -D warnings
cargo build --no-default-features
```
