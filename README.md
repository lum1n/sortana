# sortana

Embeddable, provider-agnostic task-complexity classifier for model routing.
Give it a task description and it returns a complexity tier plus the model that
fits that tier, for whichever model maker and host you route to.

19 curated **lineups** pick four models per maker, one per tier: Anthropic
(default), OpenAI, Google, Mistral, xAI, DeepSeek, Alibaba (Qwen), Moonshot
(Kimi), Zhipu (GLM), MiniMax, Meta (Muse), Nvidia (Nemotron), Xiaomi (MiMo),
ByteDance (Seed), StepFun, Tencent (Hy), Cohere, Amazon (Nova) and Upstage
(Solar). Some makers have fewer than four distinct tiers, so a model can repeat.

| Tier       | `anthropic` (default) | `openai`        | `google`                |
|------------|-----------------------|-----------------|-------------------------|
| `Simple`   | `claude-haiku-5-5`    | `gpt-6-luna`    | `gemini-3.5-flash-lite` |
| `Medium`   | `claude-sonnet-5-5`   | `gpt-6.1-terra` | `gemini-3.6-flash`      |
| `Hard`     | `claude-opus-5-5`     | `gpt-6.1-sol`   | `gemini-3.8-flash`      |
| `VeryHard` | `claude-fable-5-1`    | `gpt-6-astra`   | `gemini-4-argon`        |

Each lineup resolves to model ids on any of 170 **hosts**: the maker's own API,
clouds (Amazon Bedrock, Google Vertex, Azure), gateways (OpenRouter, Vercel,
Cloudflare AI Gateway, OpenCode Zen, OpenCode Go) and inference providers. Ids
can also come in [opencode](https://opencode.ai)'s `host/model` format. The
mapping is a snapshot of [models.dev](https://models.dev), the catalogue opencode
itself uses. Run `sortana --list` to see every lineup, or `sortana --list <host>`
for what a host serves.

Classification is remote, with no heuristics and no local model. The default
backend is [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One" model. Each task becomes one Jev request with a single
Choice question. Its four criteria are the tier definitions in
[`src/rubric.rs`](src/rubric.rs). Jev returns a calibrated probability per tier.

> Jev is in early access and needs an API key (`TYPESAFE_API_KEY`).

## Usage

```rust
use sortana::{Classifier, ModelMap};

let c = Classifier::builder()
    .jev(std::env::var("TYPESAFE_API_KEY")?)
    .models(ModelMap::preset("openai")?)
    .build()?;

let r = c.classify("Migrate our billing service to event sourcing").await?;
r.tier;          // Tier::Hard
r.model;         // "gpt-6.1-sol": the model to send the task to
r.probabilities; // [f32; 4], indexed like Tier::ALL
r.confidence;    // Jev's confidence
r.degraded;      // true if the backend failed and the fallback tier was used
```

### Choosing models

```rust
use sortana::{ModelMap, Tier};

// A lineup on the maker's own API. Aliases work too: "claude", "gemini", "qwen", "kimi", ...
ModelMap::preset("mistral")?;              // "ministral-14b-2512", ..., "mistral-large-4"

// The same lineup on another host.
ModelMap::preset_on("anthropic", "amazon-bedrock")?; // "anthropic.claude-haiku-5-5", ...
ModelMap::preset_on("openai", "azure")?;

// opencode's host/model format, ready for opencode's `model` setting.
ModelMap::opencode("anthropic", "opencode")?;  // "opencode/claude-haiku-5-5", ...
ModelMap::opencode("qwen", "openrouter")?;     // "openrouter/qwen/qwen3.8-flash", ...
ModelMap::opencode("anthropic", "anthropic")?; // "anthropic/claude-haiku-5-5", ...

// Override single tiers, mix makers, or list every id yourself.
ModelMap::openai().with(Tier::Hard, "ft:gpt-6.1-sol:acme:billing");
ModelMap::google().with(Tier::VeryHard, "claude-fable-5-1");
ModelMap::new("my-small", "my-medium", "my-large", "my-large");
```

Hosts often carry only part of a lineup. When a host lacks a tier's model,
`preset_on` and `opencode` use the nearest model from the same lineup that the
host does serve, trying stronger tiers first. `Lineup::coverage(host)` shows
which tiers were filled this way. The call fails with
`PresetError::NotServed` only when the host serves none of the lineup's
models.

The tier does not depend on the lineup, so one classification can route to
several places: `ModelMap::preset_on("google", "openrouter")?.get(r.tier)`.

The catalogue is a snapshot (see `sortana::CATALOG_DATE`). Model makers and
hosts rename and retire models often. To refresh it, edit
[`presets/lineups.toml`](presets/lineups.toml) if needed and run
`python3 scripts/update-catalog.py`. The script fails if a lineup names a
model that models.dev no longer lists. It skips deprecated models and free
promotional variants.

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

The `Backend` trait is the embedding seam. Use it to bring your own HTTP client,
or to classify with something other than Jev. That choice is independent of the
`Provider` you route to: you could classify with an OpenAI model and route to
Anthropic models, or the reverse. The classifier still handles timeouts,
retries, fallback, caching and the final decision.

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
cargo run --features cli -- --lineup openai --json "design a multi-region event platform"
cargo run --features cli -- --lineup qwen --opencode --host openrouter "add a --verbose flag"
cargo run --features cli -- --list opencode   # what a host serves; * marks substituted tiers
cargo run --features cli -- --hosts
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
