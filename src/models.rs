//! Mapping from tiers to concrete model ids.
//!
//! A [`Lineup`] is a curated choice of four models from one model maker, one
//! per tier. A [`Host`] is anywhere those models are served: the maker's own
//! API, a cloud (Bedrock, Vertex, Azure), a gateway (OpenRouter, OpenCode Zen)
//! and so on. The same model has different ids on different hosts; the bundled
//! catalogue, generated from <https://models.dev> on [`CATALOG_DATE`], resolves
//! them. Regenerate it with `scripts/update-catalog.py`.

use std::fmt;

use crate::catalog::{HOST_MODELS, HOSTS, LINEUPS, MODELS};
use crate::{Effort, Needs, Tier};

pub use crate::catalog::CATALOG_DATE;

/// Four models from one maker, indexed like [`Tier::ALL`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lineup {
    /// models.dev provider id of the maker, e.g. `"anthropic"`.
    pub id: &'static str,
    pub name: &'static str,
    /// models.dev canonical ids, e.g. `"anthropic/claude-haiku-5-5"`.
    pub canonical: [&'static str; 4],
    /// Ids on the maker's own API.
    pub native: [&'static str; 4],
}

/// A place that serves models, identified by its models.dev provider id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Host {
    pub id: &'static str,
    pub name: &'static str,
}

/// What models.dev knows about a lineup model: limits, list prices and
/// capabilities. Prices are the maker's own, in USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ModelInfo {
    /// models.dev canonical id, e.g. `"anthropic/claude-opus-5-5"`.
    pub id: &'static str,
    pub name: &'static str,
    /// Context window in tokens; `0` when unknown.
    pub context: u32,
    /// Largest prompt in tokens (the context window unless the model caps input lower).
    pub max_input: u32,
    pub max_output: u32,
    pub input_cost: Option<f32>,
    pub output_cost: Option<f32>,
    /// Accepts image input.
    pub vision: bool,
    /// Accepts PDF input.
    pub pdf: bool,
    pub tools: bool,
    pub reasoning: bool,
    /// Values the model's reasoning effort setting takes, e.g. `["low", "medium", "high"]`.
    pub efforts: &'static [&'static str],
}

/// Reasoning effort levels as models.dev names them, weakest first.
const EFFORT_SCALE: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

impl ModelInfo {
    /// Every lineup model, sorted by canonical id.
    pub fn all() -> &'static [ModelInfo] {
        MODELS
    }

    /// Look up by canonical id.
    pub fn get(canonical: &str) -> Option<&'static ModelInfo> {
        MODELS
            .binary_search_by(|m| m.id.cmp(canonical))
            .ok()
            .map(|i| &MODELS[i])
    }

    /// Look up by any id a lineup model goes by: canonical, a host's id, the
    /// same in opencode's `host/model` form, or the maker's native id.
    pub fn find(id: &str) -> Option<&'static ModelInfo> {
        let id = id.trim();
        if let Some(m) = Self::get(id) {
            return Some(m);
        }
        let prefixed = || {
            let (host, rest) = id.split_once('/')?;
            HOST_MODELS
                .iter()
                .find(|(h, _, m)| *h == host && *m == rest)
                .map(|r| r.1)
        };
        let bare = || HOST_MODELS.iter().find(|r| r.2 == id).map(|r| r.1);
        let native = || {
            LINEUPS.iter().find_map(|l| {
                l.native
                    .iter()
                    .position(|n| *n == id)
                    .map(|i| l.canonical[i])
            })
        };
        Self::get(prefixed().or_else(bare).or_else(native)?)
    }

    /// Blended list price per million tokens, weighting input 3:1 over
    /// output as agent and chat traffic does. `None` without both prices.
    pub fn price(&self) -> Option<f32> {
        Some((3.0 * self.input_cost? + self.output_cost?) / 4.0)
    }

    /// Whether the model meets `needs`: image input, tool calling and a
    /// prompt of `needs.context_tokens`.
    pub fn meets(&self, needs: &Needs) -> bool {
        (!needs.vision || self.vision)
            && (!needs.tools || self.tools)
            && needs
                .context_tokens
                .is_none_or(|t| self.max_input == 0 || t <= u64::from(self.max_input))
    }

    /// The value of this model's reasoning effort setting closest to
    /// `effort` (ties go to the stronger value). `None` when the model has
    /// no effort setting.
    pub fn effort_value(&self, effort: Effort) -> Option<&'static str> {
        if !self.reasoning {
            return None;
        }
        let rank = |v: &str| EFFORT_SCALE.iter().position(|s| *s == v);
        let want = rank(effort.as_str())? as isize;
        self.efforts
            .iter()
            .filter_map(|v| Some((rank(v)? as isize, *v)))
            .min_by_key(|(r, _)| ((r - want).abs(), -r))
            .map(|(_, v)| v)
    }
}

/// Short names accepted by [`Lineup::get`] besides the ids.
const LINEUP_ALIASES: &[(&str, &str)] = &[
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("gemini", "google"),
    ("grok", "xai"),
    ("qwen", "alibaba"),
    ("kimi", "moonshotai"),
    ("moonshot", "moonshotai"),
    ("glm", "zhipuai"),
    ("zhipu", "zhipuai"),
    ("zai", "zhipuai"),
    ("muse", "meta"),
    ("nemotron", "nvidia"),
    ("mimo", "xiaomi"),
    ("seed", "bytedance-seed"),
    ("bytedance", "bytedance-seed"),
    ("step", "stepfun"),
    ("hy", "tencent"),
    ("nova", "amazon"),
    ("solar", "upstage"),
];

impl Lineup {
    pub fn all() -> &'static [Lineup] {
        LINEUPS
    }

    /// Look up by id or alias (`"claude"`, `"gemini"`, `"qwen"`, ...), ignoring case.
    pub fn get(id: &str) -> Option<&'static Lineup> {
        let id = id.trim().to_ascii_lowercase();
        let id = LINEUP_ALIASES
            .iter()
            .find(|(alias, _)| *alias == id)
            .map_or(id.as_str(), |(_, target)| target);
        LINEUPS.iter().find(|l| l.id == id)
    }

    /// Metadata of the tier's model.
    pub fn info(&self, tier: Tier) -> Option<&'static ModelInfo> {
        ModelInfo::get(self.canonical[tier.index()])
    }

    /// Which tiers' models `host` serves.
    pub fn coverage(&self, host: &str) -> [bool; 4] {
        self.canonical.map(|c| host_model_id(host, c).is_some())
    }

    /// The entry of `available` that serves `tier`, for apps whose model
    /// picker lists its own ids (a CLI's `opus` alias, a full id, or
    /// opencode's `host/model`). An entry matches a lineup model when, after
    /// any `host/` prefix, it equals the native or canonical id, or is one of
    /// the native id's words (`opus` in `claude-opus-5-5`). A tier with no
    /// match takes the nearest matched tier, preferring stronger ones, as in
    /// [`ModelMap::preset_on`]. With an empty `available` it is the native id.
    pub fn model_among<S: AsRef<str>>(&self, tier: Tier, available: &[S]) -> Option<String> {
        if available.is_empty() {
            return Some(self.native[tier.index()].to_owned());
        }
        self.entry_among(tier.index(), available).map(|(_, e)| e)
    }

    /// [`model_among`](Self::model_among) for every tier, as a [`ModelMap`]
    /// whose infos are those of the lineup models the entries matched.
    /// `None` when no entry matches any tier.
    pub fn models_among<S: AsRef<str>>(&self, available: &[S]) -> Option<ModelMap> {
        if available.is_empty() {
            return Some(ModelMap::from_lineup(self));
        }
        let mut ids: [String; 4] = Default::default();
        let mut infos = [None; 4];
        for t in 0..4 {
            let (j, entry) = self.entry_among(t, available)?;
            ids[t] = entry;
            infos[t] = ModelInfo::get(self.canonical[j]);
        }
        Some(ModelMap { ids, infos })
    }

    /// The entry of `available` for tier `t` and the lineup tier it names.
    fn entry_among<S: AsRef<str>>(&self, t: usize, available: &[S]) -> Option<(usize, String)> {
        (t..4).chain((0..t).rev()).find_map(|i| {
            available
                .iter()
                .map(AsRef::as_ref)
                .find(|a| self.serves(i, a))
                .map(|a| (i, a.to_owned()))
        })
    }

    /// Whether the picker entry `entry` names the tier-`i` model.
    fn serves(&self, i: usize, entry: &str) -> bool {
        let entry = entry.trim().to_ascii_lowercase();
        let native = self.native[i];
        let canonical = self.canonical[i];
        let bare = canonical.rsplit('/').next().unwrap_or(canonical);
        let last = entry.rsplit('/').next().unwrap_or(&entry);
        if entry == canonical || last == native || last == bare {
            return true;
        }
        // A bare alias: an alphabetic word of the native id that no other
        // tier's model shares (`claude` names every tier, `opus` one).
        entry == last
            && entry.chars().all(|c| c.is_ascii_alphabetic())
            && native.split('-').any(|w| w == entry)
            && (0..4)
                .filter(|&j| self.native[j] != native)
                .all(|j| !self.native[j].split('-').any(|w| w == entry))
    }
}

impl Host {
    /// All hosts serving at least one lineup model, sorted by id.
    pub fn all() -> &'static [Host] {
        HOSTS
    }

    pub fn get(id: &str) -> Option<&'static Host> {
        HOSTS
            .binary_search_by(|h| h.id.cmp(id))
            .ok()
            .map(|i| &HOSTS[i])
    }
}

/// The id `host` uses for a lineup model, by canonical id. Only covers models
/// that appear in a lineup.
pub fn host_model_id(host: &str, canonical: &str) -> Option<&'static str> {
    HOST_MODELS
        .binary_search_by(|(h, c, _)| (*h, *c).cmp(&(host, canonical)))
        .ok()
        .map(|i| HOST_MODELS[i].2)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PresetError {
    #[error("unknown lineup {0:?}")]
    UnknownLineup(String),
    #[error("unknown host {0:?}")]
    UnknownHost(String),
    #[error("host {host:?} serves none of the {lineup:?} lineup's models")]
    NotServed { lineup: String, host: String },
}

/// Maps tiers to model ids. Build one from a [`Lineup`] with
/// [`preset`](Self::preset) / [`preset_on`](Self::preset_on) /
/// [`opencode`](Self::opencode), or list ids yourself with [`new`](Self::new).
///
/// Each tier also carries the [`ModelInfo`] of its model when it is a lineup
/// model, which the classifier uses to check capabilities and prices. Two
/// maps are equal when their ids are; with `serde` a map is its four ids.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(from = "[String; 4]", into = "[String; 4]"))]
pub struct ModelMap {
    ids: [String; 4],
    infos: [Option<&'static ModelInfo>; 4],
}

impl ModelMap {
    /// Infos are looked up with [`ModelInfo::find`].
    pub fn new(
        simple: impl Into<String>,
        medium: impl Into<String>,
        hard: impl Into<String>,
        very_hard: impl Into<String>,
    ) -> Self {
        [simple.into(), medium.into(), hard.into(), very_hard.into()].into()
    }

    fn from_lineup(l: &Lineup) -> Self {
        ModelMap {
            ids: l.native.map(str::to_owned),
            infos: Tier::ALL.map(|t| l.info(t)),
        }
    }

    /// A lineup with ids for the maker's own API.
    pub fn preset(lineup: &str) -> Result<Self, PresetError> {
        let l = Lineup::get(lineup).ok_or_else(|| PresetError::UnknownLineup(lineup.to_owned()))?;
        Ok(Self::from_lineup(l))
    }

    /// A lineup with ids for `host`. A tier whose model the host lacks gets the
    /// nearest model the host has from the same lineup, preferring stronger
    /// tiers. Check [`Lineup::coverage`] to see which tiers were substituted.
    pub fn preset_on(lineup: &str, host: &str) -> Result<Self, PresetError> {
        let l = Lineup::get(lineup).ok_or_else(|| PresetError::UnknownLineup(lineup.to_owned()))?;
        let h = Host::get(host).ok_or_else(|| PresetError::UnknownHost(host.to_owned()))?;
        let ids = l.canonical.map(|c| host_model_id(h.id, c));
        let pick = |tier: usize| (tier..4).chain((0..tier).rev()).find(|&i| ids[i].is_some());
        let not_served = || PresetError::NotServed {
            lineup: l.id.to_owned(),
            host: h.id.to_owned(),
        };
        let mut out = ModelMap {
            ids: Default::default(),
            infos: [None; 4],
        };
        for t in 0..4 {
            let j = pick(t).ok_or_else(not_served)?;
            out.ids[t] = ids[j].unwrap_or_default().to_owned();
            out.infos[t] = l.info(Tier::ALL[j]);
        }
        Ok(out)
    }

    /// Like [`preset_on`](Self::preset_on), with ids in opencode's
    /// `host/model` format, e.g. `opencode/claude-haiku-5-5` for OpenCode Zen
    /// or `anthropic/claude-haiku-5-5` for Anthropic's API.
    pub fn opencode(lineup: &str, host: &str) -> Result<Self, PresetError> {
        let h = Host::get(host).ok_or_else(|| PresetError::UnknownHost(host.to_owned()))?;
        Ok(Self::preset_on(lineup, host)?.prefixed(&format!("{}/", h.id)))
    }

    pub fn anthropic() -> Self {
        Self::preset("anthropic").expect("bundled anthropic lineup")
    }

    pub fn openai() -> Self {
        Self::preset("openai").expect("bundled openai lineup")
    }

    pub fn google() -> Self {
        Self::preset("google").expect("bundled google lineup")
    }

    /// Prepend `prefix` to every id.
    pub fn prefixed(self, prefix: &str) -> Self {
        ModelMap {
            ids: self.ids.map(|id| format!("{prefix}{id}")),
            infos: self.infos,
        }
    }

    /// Replace a tier's model. Its info is looked up with [`ModelInfo::find`];
    /// use [`with_info`](Self::with_info) to set it yourself.
    pub fn with(mut self, tier: Tier, model: impl Into<String>) -> Self {
        let model = model.into();
        self.infos[tier.index()] = ModelInfo::find(&model);
        self.ids[tier.index()] = model;
        self
    }

    /// Set a tier's model info, e.g. a fine-tune's base model.
    pub fn with_info(mut self, tier: Tier, info: Option<&'static ModelInfo>) -> Self {
        self.infos[tier.index()] = info;
        self
    }

    pub fn get(&self, tier: Tier) -> &str {
        &self.ids[tier.index()]
    }

    /// Metadata of the tier's model, when it is a known lineup model.
    pub fn info(&self, tier: Tier) -> Option<&'static ModelInfo> {
        self.infos[tier.index()]
    }
}

impl From<[String; 4]> for ModelMap {
    fn from(ids: [String; 4]) -> Self {
        let infos = [0, 1, 2, 3].map(|i| ModelInfo::find(&ids[i]));
        ModelMap { ids, infos }
    }
}

impl From<ModelMap> for [String; 4] {
    fn from(m: ModelMap) -> Self {
        m.ids
    }
}

impl PartialEq for ModelMap {
    fn eq(&self, other: &Self) -> bool {
        self.ids == other.ids
    }
}

impl Eq for ModelMap {}

/// The Anthropic lineup.
impl Default for ModelMap {
    fn default() -> Self {
        Self::anthropic()
    }
}

impl fmt::Display for Lineup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_is_sorted_and_consistent() {
        assert!(HOSTS.windows(2).all(|w| w[0].id < w[1].id));
        assert!(
            HOST_MODELS
                .windows(2)
                .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1))
        );
        for l in LINEUPS {
            assert!(
                l.canonical
                    .iter()
                    .all(|c| c.starts_with(&format!("{}/", l.id)))
            );
            // Every bundled lineup resolves somewhere, and the infallible
            // constructors below rely on these three.
            assert!(
                HOSTS.iter().any(|h| l.coverage(h.id).iter().any(|&b| b)),
                "{}",
                l.id
            );
        }
        for id in ["anthropic", "openai", "google"] {
            assert!(Lineup::get(id).is_some());
        }
    }

    #[test]
    fn presets_and_aliases() {
        assert_eq!(ModelMap::default().get(Tier::Simple), "claude-haiku-5-5");
        assert_eq!(ModelMap::openai().get(Tier::VeryHard), "gpt-6-astra");
        assert_eq!(
            ModelMap::google().get(Tier::Simple),
            "gemini-3.5-flash-lite"
        );
        assert_eq!(ModelMap::preset("Gemini").unwrap(), ModelMap::google());
        assert_eq!(Lineup::get("kimi").unwrap().id, "moonshotai");
        assert_eq!(
            ModelMap::preset("nope"),
            Err(PresetError::UnknownLineup("nope".into()))
        );
    }

    #[test]
    fn resolves_host_ids() {
        let m = ModelMap::preset_on("anthropic", "amazon-bedrock").unwrap();
        assert_eq!(m.get(Tier::Simple), "anthropic.claude-haiku-5-5");
        let m = ModelMap::opencode("anthropic", "opencode").unwrap();
        assert_eq!(m.get(Tier::Hard), "opencode/claude-opus-5-5");
        let m = ModelMap::opencode("openai", "openrouter").unwrap();
        assert_eq!(m.get(Tier::Simple), "openrouter/openai/gpt-6-luna");
        assert!(matches!(
            ModelMap::preset_on("anthropic", "nope"),
            Err(PresetError::UnknownHost(_))
        ));
    }

    #[test]
    fn missing_tiers_take_nearest_stronger_model() {
        let mut partial = 0;
        for l in LINEUPS {
            for h in HOSTS {
                let cov = l.coverage(h.id);
                if cov.iter().all(|&b| b) || !cov.iter().any(|&b| b) {
                    continue;
                }
                partial += 1;
                let m = ModelMap::preset_on(l.id, h.id).unwrap();
                for (i, tier) in Tier::ALL.into_iter().enumerate() {
                    let j = (i..4).chain((0..i).rev()).find(|&j| cov[j]).unwrap();
                    let want = host_model_id(h.id, l.canonical[j]).unwrap();
                    assert_eq!(m.get(tier), want, "{} on {}", l.id, h.id);
                }
            }
        }
        assert!(
            partial > 0,
            "snapshot has no partially served lineup to test"
        );
    }

    #[test]
    fn native_ids_match_the_makers_own_host() {
        for l in LINEUPS {
            if Host::get(l.id).is_none() {
                continue;
            }
            for (c, n) in l.canonical.iter().zip(l.native) {
                if let Some(id) = host_model_id(l.id, c) {
                    assert_eq!(id, n, "{c}");
                }
            }
        }
        assert_eq!(
            ModelMap::preset("deepseek").unwrap(),
            ModelMap::preset_on("deepseek", "deepseek").unwrap()
        );
        assert_eq!(
            ModelMap::preset("deepseek").unwrap().get(Tier::Simple),
            "deepseek-flash"
        );
    }

    #[test]
    fn model_among_matches_aliases_ids_and_prefixes() {
        let a = Lineup::get("anthropic").unwrap();
        let aliases = ["opus", "sonnet", "haiku"];
        assert_eq!(
            a.model_among(Tier::Simple, &aliases).as_deref(),
            Some("haiku")
        );
        assert_eq!(a.model_among(Tier::Hard, &aliases).as_deref(), Some("opus"));
        // No `fable` alias listed: the nearest tier (weaker, as none is stronger).
        assert_eq!(
            a.model_among(Tier::VeryHard, &aliases).as_deref(),
            Some("opus")
        );
        let ids = ["claude-sonnet-5-5", "claude-fable-5-1"];
        assert_eq!(
            a.model_among(Tier::Simple, &ids).as_deref(),
            Some("claude-sonnet-5-5")
        );
        assert_eq!(
            a.model_among(Tier::VeryHard, &ids).as_deref(),
            Some("claude-fable-5-1")
        );
        let prefixed = ["opencode/claude-opus-5-5", "anthropic/claude-haiku-5-5"];
        assert_eq!(
            a.model_among(Tier::Medium, &prefixed).as_deref(),
            Some("opencode/claude-opus-5-5")
        );
        // `claude` names every tier, so it is not an alias for one.
        assert_eq!(a.model_among(Tier::Hard, &["claude", "default"]), None);
        assert_eq!(
            a.model_among::<&str>(Tier::Hard, &[]).as_deref(),
            Some("claude-opus-5-5")
        );
        // Repeated models (xAI uses grok-4.3 twice) still resolve.
        let x = Lineup::get("xai").unwrap();
        assert_eq!(
            x.model_among(Tier::Medium, &["grok-4.3"]).as_deref(),
            Some("grok-4.3")
        );
    }

    #[test]
    fn model_infos() {
        assert!(MODELS.windows(2).all(|w| w[0].id < w[1].id));
        for l in LINEUPS {
            for t in Tier::ALL {
                assert!(l.info(t).is_some(), "{} {t}", l.id);
            }
        }
        let opus = ModelInfo::get("anthropic/claude-opus-5-5").unwrap();
        for id in [
            "claude-opus-5-5",
            "opencode/claude-opus-5-5",
            "anthropic.claude-opus-5-5",
            "amazon-bedrock/anthropic.claude-opus-5-5",
        ] {
            assert_eq!(ModelInfo::find(id), Some(opus), "{id}");
        }
        assert_eq!(ModelInfo::find("ft:my-model"), None);
        assert_eq!(opus.price(), Some((3.0 * 4.0 + 20.0) / 4.0));
        // Bedrock ids still carry the lineup model's info.
        let m = ModelMap::preset_on("anthropic", "amazon-bedrock").unwrap();
        assert_eq!(m.info(Tier::Hard), Some(opus));
        assert_eq!(
            ModelMap::anthropic().with(Tier::Hard, "x").info(Tier::Hard),
            None
        );
    }

    #[test]
    fn effort_values_round_to_the_nearest_setting() {
        let info = |id| ModelInfo::get(id).unwrap();
        let opus = info("anthropic/claude-opus-5-5");
        assert_eq!(opus.effort_value(Effort::Medium), Some("medium"));
        // Mistral Small only has `none` and `high`: low is equally far from both.
        let small = info("mistral/mistral-small-2603");
        assert_eq!(small.effort_value(Effort::Low), Some("high"));
        // DeepSeek has low/high/max: medium rounds up.
        let ds = info("deepseek/deepseek-v4-pro");
        assert_eq!(ds.effort_value(Effort::Medium), Some("high"));
        assert_eq!(
            info("cohere/command-r7b-12-2024").effort_value(Effort::High),
            None
        );
    }

    #[test]
    fn needs_check_capabilities() {
        let ds = ModelInfo::get("deepseek/deepseek-v4-pro").unwrap();
        assert!(ds.meets(&Needs::default()));
        assert!(!ds.meets(&Needs {
            vision: true,
            ..Needs::default()
        }));
        let glm = ModelInfo::get("zhipuai/glm-5.1").unwrap();
        assert!(glm.meets(&Needs {
            context_tokens: Some(200_000),
            ..Needs::default()
        }));
        assert!(!glm.meets(&Needs {
            context_tokens: Some(200_001),
            ..Needs::default()
        }));
    }

    #[test]
    fn models_among_carries_matched_infos() {
        let a = Lineup::get("anthropic").unwrap();
        let m = a.models_among(&["opus", "haiku"]).unwrap();
        assert_eq!(m.get(Tier::Medium), "opus");
        assert_eq!(m.info(Tier::Medium), a.info(Tier::Hard));
        assert_eq!(m.info(Tier::Simple), a.info(Tier::Simple));
        assert_eq!(m.get(Tier::VeryHard), "opus");
        assert!(a.models_among(&["default"]).is_none());
        assert_eq!(a.models_among::<&str>(&[]).unwrap(), ModelMap::anthropic());
    }

    #[test]
    fn host_without_lineup_models_errors() {
        // Amazon's Nova models are only on a few hosts; Anthropic's API is not one.
        assert!(matches!(
            ModelMap::preset_on("amazon", "anthropic"),
            Err(PresetError::NotServed { .. })
        ));
    }
}
