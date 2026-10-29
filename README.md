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

Each lineup resolves to model ids on any of 171 **hosts**: the maker's own API,
clouds (Amazon Bedrock, Google Vertex, Azure), gateways (OpenRouter, Vercel,
Cloudflare AI Gateway, OpenCode Zen, OpenCode Go) and inference providers. Ids
can also come in [opencode](https://opencode.ai)'s `host/model` format. The
mapping is a snapshot of [models.dev](https://models.dev), the catalogue opencode
itself uses. Run `sortana --list` to see every lineup, or `sortana --list <host>`
for what a host serves.

Classification is remote, with no heuristics and no local model. The default
backend is [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One" model. Each task becomes one Jev request. Its main
Choice question has four criteria, the tier definitions in
[`src/rubric.rs`](src/rubric.rs), and Jev returns a calibrated probability per
tier. The same request also asks how much reasoning effort the task needs and
whether it needs vision or tools. Jev's output is free, so the extra answers
cost almost nothing.

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
r.effort;        // Some(Effort::High): how much the model should think
r.reasoning_effort; // Some("high"): that effort as the model's own setting
r.needs;         // Needs { vision, tools, context_tokens }
r.info;          // Some(&ModelInfo): limits, prices, capabilities of r.model
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
`python3 scripts/update-catalog.py`. It also refreshes each lineup model's
limits, prices and capabilities. The script fails if a lineup names a
model that models.dev no longer lists. It skips deprecated models and free
promotional variants.

### Effort, needs and context

Pass what the app knows about a task with `classify_with`. The context's
counts go to Jev as a `Context:` section after the task.

```rust
use sortana::Context;

let ctx = Context {
    files: Some(40),              // files attached or in scope
    images: 1,                    // any image: the model must accept images
    prompt_tokens: Some(180_000), // the model's input limit must fit this
    turns: Some(12),
    tools: Some(true),            // tools will be offered; overrides Jev's guess
    notes: Some("Rust monorepo, 400k lines".into()),
};
let r = c.classify_with("Why does the chart in this screenshot dip?", &ctx).await?;
```

`r.needs` combines the context with Jev's vision and tools answers. When the
tier's model can't meet the needs (no image input, a context window that is
too small, no tool calling), the tier moves up to the lowest stronger tier
whose model can, up to `max_tier`, and `r.refit` is set. Needs never move the
tier down: a hard task stays on the hard model even if only a weaker model
accepts images. In that case `r.fits` is `Some(false)`, so the app can
decide, for example by describing the images in text. `r.fits` is `None` when
the model is not one sortana has metadata for.

`r.effort` (low, medium, high) is separate from the tier: a short proof can
need a strong model *and* a lot of thinking, while a big refactor might need a
strong model and little thinking. `r.reasoning_effort` is that level as a value
the model's effort setting takes, from models.dev (for example `"high"` for
Claude, or DeepSeek's `"high"` for medium, since DeepSeek only has
low/high/max). Pass it as `reasoning_effort` or the provider's equivalent.

Turn the extra questions off with `.ask_effort(false)` / `.ask_needs(false)`.
A custom `Backend` that only implements `classify` gives no extra answers, and
`needs` then comes from the context alone.

### Tier limits and escalation

```rust
let c = Classifier::builder()
    .jev(key)
    .min_tier(Tier::Medium)   // never the smallest model
    .max_tier(Tier::Hard)     // never the most expensive one
    .build()?;

let r = c.classify(task).await?;   // r.clamped when a limit moved the tier
// The model failed, or the user asked for a better answer:
if let Some(up) = c.escalate(&r) { /* up.tier, up.model */ }
```

`escalate` returns the lowest tier above the current one, up to `max_tier`,
whose model differs from the current model. Lineups that repeat a model skip
the repeat. Models that meet `r.needs` come first. It returns `None` at the top.
The fallback tier of a degraded result is clamped too.

### Routing by cost

`Decision::Cost` picks the tier with the lowest expected cost instead of the
most likely tier. It uses Jev's calibrated probabilities and the models' list
prices from models.dev:

```rust
use sortana::Decision;

let c = Classifier::builder().jev(key).decision(Decision::COST).build()?;
let c = Classifier::builder().jev(key).decision(Decision::Cost { miss_penalty: 3.0 }).build()?;
```

Choosing a tier costs its model's blended price (input weighted 3:1 over
output). If the task turns out to need a stronger model, it is re-run there at
that model's price times `1 + miss_penalty`. The penalty stands for the time
lost and the bad answer. Tasks harder than `max_tier` count as needing
`max_tier`. With `Decision::COST` (`miss_penalty = 1`) and Anthropic's prices,
a task that is 60% simple, 30% medium and 10% hard goes to Haiku. Raise the
penalty when wrong answers are expensive. When a tier's model has no known
price (custom ids), it falls back to the confidence rule. The cost rule picks
among tiers within the limits; needs apply afterwards and can only move the
tier up.

### Model metadata

`ModelInfo` holds models.dev's data for every lineup model: context window,
input and output limits, prices, image and PDF input, tool calling, and the
values of its reasoning effort setting. `ModelInfo::find` accepts any id a
lineup model goes by (canonical, native, a host's id, `host/id`). Every
`ModelMap` keeps the info of each tier's model: presets set it directly,
`ModelMap::new` and `.with()` look it up. Use `.with_info()` to give a
fine-tune its base model's info. `Classifier::classify_for(text, &ctx,
&models)` routes one call to another map, for example
`lineup.models_among(&app_ids)`, which also carries the matched models' infos.

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
   With `Decision::Cost`, the tier is the cheapest expected one instead.
4. The tier is clamped to `min_tier..=max_tier`, then moved up to the lowest
   tier (up to `max_tier`) whose model meets the task's needs. Needs never
   move it down.
5. Timeouts, transport errors, HTTP 408/429/5xx and backend errors marked
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
retries, fallback, caching and the final decision. To support the effort,
vision and tools questions too, override `Backend::assess` and return an
`Assessment`. The questions are in `sortana::rubric` (`EFFORT`, `VISION`,
`TOOLS`), and `req.ask` says which ones were asked. `req.state()` is the task
followed by its context section.

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

## Swift (UniFFI)

`crates/sortana-ffi` exposes sortana to Swift (and Kotlin) through
[UniFFI](https://mozilla.github.io/uniffi-rs/). A `Suggester` classifies a
task and picks its model from a lineup and, optionally, from the ids the
app's own model picker lists:

```swift
let s = try Suggester(apiKey: key, lineup: lineupForHarness(harness: "claude") ?? "anthropic", baseUrl: nil)
let r = try await s.suggest(task: "Migrate billing to event sourcing", available: ["opus", "sonnet", "haiku"])
r.tier            // .hard
r.model           // "opus"
r.effort          // .high
r.reasoningEffort // "high"
r.needs           // Needs(vision: false, tools: true, contextTokens: nil)
r.modelInfo       // ModelInfo(id: "anthropic/claude-opus-5-5", context: 1000000, inputCost: 4, ...)

// Limits, cost routing, context and escalation.
let s2 = try Suggester.withOptions(apiKey: key, lineup: "anthropic",
                                   options: SuggesterOptions(maxTier: .hard, decision: .cost(missPenalty: 1)))
let r2 = try await s2.suggestWith(task: "What is wrong in this screenshot?",
                                  context: TaskContext(files: 12, images: 1, promptTokens: 40_000),
                                  available: ["opus", "sonnet", "haiku"])
let up = s2.escalate(from: r2, available: ["opus", "sonnet", "haiku"])  // next model up, or nil
modelInfo(id: "openrouter/openai/gpt-6-luna")?.inputCost
```

`lineupForHarness` maps a coding harness to its lineup (`"claude"` and
`"opencode"` to `anthropic`, `"codex"` to `openai`) and returns `nil` for any
other. `available` entries match a lineup model by native or canonical id, with
or without a `host/` prefix, or by a word of the id that names one tier
(`opus`). A tier with no match takes the nearest matched tier, stronger
first (`Lineup::model_among`). Effort, needs, model info and capability fit
are judged on the lineup model each entry matched, and needs only move the
tier up. `SuggesterOptions` fields
left `nil` keep sortana's defaults. Backend failures throw
`SortanaError.Backend(detail:)`, and invalid options throw
`SortanaError.Config(detail:)`.

Two ways to ship it:

- **Standalone:** `scripts/build-swift.sh` (macOS) builds `swift/`, a Swift
  package with `SortanaFFI.xcframework` and the bindings.
- **Inside another Rust static library** (an app that already links Rust,
  like Sessh): depend on `crates/sortana-ffi` from that library and generate
  the bindings with `scripts/swift-bindings.sh OUT_DIR`. Two Rust static
  libraries in one binary would duplicate std.

## Kotlin (Android, JVM)

The same crate generates Kotlin bindings (package `io.github.lum1n.sortana`,
loaded through JNA, `suspend` functions on kotlinx-coroutines). The API
mirrors Swift's:

```kotlin
import io.github.lum1n.sortana.*

val s = Suggester.withOptions(key, lineupForHarness("codex") ?: "openai",
    SuggesterOptions(maxTier = Tier.HARD, decision = Decision.Cost(1.0f)))
val r = s.suggestWith("Find the deadlock in the job queue",
    TaskContext(files = 12u, images = 1u), listOf())
r.tier; r.model; r.effort; r.reasoningEffort; r.needs; r.modelInfo
s.escalate(r, listOf())
```

Errors are `SortanaException.Backend` / `.Config`, with the text in `detail`.

- **Android:** `scripts/build-android.sh` builds `libsortana_ffi.so` for
  arm64-v8a, armeabi-v7a and x86_64 with cargo-ndk, writes the bindings into
  `kotlin/`, and assembles `kotlin/build/outputs/aar/sortana-release.aar`
  (minSdk 26, depends on JNA and kotlinx-coroutines). Use `--no-gradle` to
  only generate the sources and libraries.
- **JVM or other builds:** `scripts/kotlin-bindings.sh OUT_DIR` writes the
  bindings and prints the host library's path. Load it with
  `-Djna.library.path=<its directory>`.

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
cargo run --features cli -- --max-tier hard --cost 2 --images 1 --tokens 120000 "why does this chart dip"
cargo run --features cli -- --list opencode   # what a host serves; * marks substituted tiers
cargo run --features cli -- --hosts
```

## Cost and latency

According to TypeSafe AI, Jev responds in 70-500 ms and costs $0.042 per million
input tokens, with free output. The tier rubric adds a few hundred input tokens
per request, and the effort, vision and tools questions add a few hundred more.

## Evaluation

`eval/cases.jsonl` holds labelled tasks. Run them against live Jev with:

```sh
TYPESAFE_API_KEY=... cargo run --example eval
```

The eval reports accuracy, the share of tasks that are off by two or more tiers
(the target is under 5%), p50/p95 latency and a confusion matrix. Cases may
also carry an `effort` label (`low`, `medium`, `high`). The eval then reports
how often Jev answered the effort question, effort accuracy with its own
confusion matrix, and how many tasks were flagged as needing vision or tools.
Effort is reported, not gated. Re-run it after
any rubric change and bump `RUBRIC_VERSION`.

## Development

```sh
cargo test --all-features                       # mock backend + local HTTP, no network
TYPESAFE_API_KEY=... cargo test -- --ignored    # live Jev
cargo clippy --all-targets --all-features -- -D warnings
cargo build --no-default-features
```
