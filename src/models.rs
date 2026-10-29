//! Mapping from tiers to concrete model ids.
//!
//! A [`Lineup`] is a curated choice of four models from one model maker, one
//! per tier. A [`Host`] is anywhere those models are served: the maker's own
//! API, a cloud (Bedrock, Vertex, Azure), a gateway (OpenRouter, OpenCode Zen)
//! and so on. The same model has different ids on different hosts; the bundled
//! catalogue, generated from <https://models.dev> on [`CATALOG_DATE`], resolves
//! them. Regenerate it with `scripts/update-catalog.py`.

use std::fmt;

use crate::Tier;
use crate::catalog::{HOST_MODELS, HOSTS, LINEUPS};

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

    /// Which tiers' models `host` serves.
    pub fn coverage(&self, host: &str) -> [bool; 4] {
        self.canonical.map(|c| host_model_id(host, c).is_some())
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelMap([String; 4]);

impl ModelMap {
    pub fn new(
        simple: impl Into<String>,
        medium: impl Into<String>,
        hard: impl Into<String>,
        very_hard: impl Into<String>,
    ) -> Self {
        ModelMap([simple.into(), medium.into(), hard.into(), very_hard.into()])
    }

    /// A lineup with ids for the maker's own API.
    pub fn preset(lineup: &str) -> Result<Self, PresetError> {
        let l = Lineup::get(lineup).ok_or_else(|| PresetError::UnknownLineup(lineup.to_owned()))?;
        Ok(ModelMap(l.native.map(str::to_owned)))
    }

    /// A lineup with ids for `host`. A tier whose model the host lacks gets the
    /// nearest model the host has from the same lineup, preferring stronger
    /// tiers. Check [`Lineup::coverage`] to see which tiers were substituted.
    pub fn preset_on(lineup: &str, host: &str) -> Result<Self, PresetError> {
        let l = Lineup::get(lineup).ok_or_else(|| PresetError::UnknownLineup(lineup.to_owned()))?;
        let h = Host::get(host).ok_or_else(|| PresetError::UnknownHost(host.to_owned()))?;
        let ids = l.canonical.map(|c| host_model_id(h.id, c));
        let pick =
            |tier: usize| -> Option<&str> { (tier..4).chain((0..tier).rev()).find_map(|i| ids[i]) };
        let not_served = || PresetError::NotServed {
            lineup: l.id.to_owned(),
            host: h.id.to_owned(),
        };
        let mut out: [String; 4] = Default::default();
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = pick(i).ok_or_else(not_served)?.to_owned();
        }
        Ok(ModelMap(out))
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
        ModelMap(self.0.map(|id| format!("{prefix}{id}")))
    }

    pub fn with(mut self, tier: Tier, model: impl Into<String>) -> Self {
        self.0[tier.index()] = model.into();
        self
    }

    pub fn get(&self, tier: Tier) -> &str {
        &self.0[tier.index()]
    }
}

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
    fn host_without_lineup_models_errors() {
        // Amazon's Nova models are only on a few hosts; Anthropic's API is not one.
        assert!(matches!(
            ModelMap::preset_on("amazon", "anthropic"),
            Err(PresetError::NotServed { .. })
        ));
    }
}
