//! What a task needs besides a tier: reasoning effort, capabilities, and the
//! context it comes with.

use std::fmt;
use std::str::FromStr;

use crate::TierProbs;

/// How much step-by-step reasoning a task needs, separate from how capable
/// the model must be. Maps onto a model's reasoning effort setting with
/// [`ModelInfo::effort_value`](crate::ModelInfo::effort_value).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Effort {
    Low,
    Medium,
    High,
}

impl Effort {
    /// All levels in ascending order, indexed like [`Assessment::effort`].
    pub const ALL: [Effort; 3] = [Effort::Low, Effort::Medium, Effort::High];

    /// Stable lowercase identifier, also used as the Jev option name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        }
    }

    /// The most likely level. Ties go to the higher one.
    pub fn argmax(probabilities: &[f32; 3]) -> Effort {
        let mut best = 0;
        for i in 1..3 {
            if probabilities[i] >= probabilities[best] {
                best = i;
            }
        }
        Effort::ALL[best]
    }
}

impl fmt::Display for Effort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown effort {0:?}")]
pub struct ParseEffortError(pub String);

impl FromStr for Effort {
    type Err = ParseEffortError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Ok(Effort::Low),
            "medium" => Ok(Effort::Medium),
            "high" => Ok(Effort::High),
            _ => Err(ParseEffortError(s.to_owned())),
        }
    }
}

/// Capabilities the model has to have for a task.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Needs {
    /// Image input: the task comes with images or asks to look at some.
    pub vision: bool,
    /// Tool calling: the task acts on files, commands, the web or APIs.
    pub tools: bool,
    /// Prompt size in tokens, from [`Context::prompt_tokens`].
    pub context_tokens: Option<u64>,
}

/// What the app knows about a task beyond its text. Every field is optional.
/// Counts are passed to the backend as a `Context:` section after the task;
/// `images`, `tools` and `prompt_tokens` also decide [`Needs`].
///
/// ```
/// let ctx = sortana::Context { files: Some(40), images: 1, ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Context {
    /// Files attached or in scope for the task.
    pub files: Option<u32>,
    /// Images attached to the task. Any image means the model needs vision.
    pub images: u32,
    /// Size of the prompt the model will get (conversation, files), in tokens.
    /// Models whose input limit is smaller are skipped.
    pub prompt_tokens: Option<u64>,
    /// Conversation turns so far.
    pub turns: Option<u32>,
    /// Whether the request will offer tools. Overrides the backend's guess.
    pub tools: Option<bool>,
    /// Anything else worth knowing, as short text ("Rust monorepo, 400k lines").
    pub notes: Option<String>,
}

impl Context {
    /// The `Context:` section sent after the task, or `None` when there is
    /// nothing to say.
    pub fn render(&self) -> Option<String> {
        let mut lines = Vec::new();
        if let Some(n) = self.files {
            lines.push(format!("- files attached or in scope: {n}"));
        }
        if self.images > 0 {
            lines.push(format!("- images attached: {}", self.images));
        }
        if let Some(n) = self.prompt_tokens {
            lines.push(format!("- prompt size: about {n} tokens"));
        }
        if let Some(n) = self.turns {
            lines.push(format!("- conversation so far: {n} turns"));
        }
        if let Some(t) = self.tools {
            let s = if t { "available" } else { "not available" };
            lines.push(format!("- tools (files, commands, web): {s}"));
        }
        if let Some(n) = self
            .notes
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty())
        {
            lines.push(format!("- notes: {n}"));
        }
        (!lines.is_empty()).then(|| format!("Context:\n{}", lines.join("\n")))
    }
}

/// Everything a backend answers about one task: tier probabilities plus the
/// optional answers asked for in [`Ask`](crate::Ask). Missing answers are
/// `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Assessment {
    pub tiers: TierProbs,
    /// Probabilities indexed like [`Effort::ALL`].
    pub effort: Option<[f32; 3]>,
    /// Probability that the task needs image input.
    pub vision: Option<f32>,
    /// Probability that the task needs tools.
    pub tools: Option<f32>,
}

impl From<TierProbs> for Assessment {
    fn from(tiers: TierProbs) -> Self {
        Assessment {
            tiers,
            effort: None,
            vision: None,
            tools: None,
        }
    }
}

impl Assessment {
    /// Needs implied by this assessment and the task's context.
    pub fn needs(&self, ctx: &Context) -> Needs {
        Needs::from_parts(Some(self), ctx)
    }
}

impl Needs {
    /// Needs from the context alone, when the backend gave no answer.
    pub fn from_context(ctx: &Context) -> Needs {
        Needs::from_parts(None, ctx)
    }

    fn from_parts(a: Option<&Assessment>, ctx: &Context) -> Needs {
        let yes = |p: Option<f32>| p.is_some_and(|p| p >= 0.5);
        Needs {
            vision: ctx.images > 0 || yes(a.and_then(|a| a.vision)),
            tools: ctx.tools.unwrap_or_else(|| yes(a.and_then(|a| a.tools))),
            context_tokens: ctx.prompt_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_context() {
        assert_eq!(Context::default().render(), None);
        let ctx = Context {
            files: Some(40),
            images: 2,
            tools: Some(true),
            notes: Some("  ".into()),
            ..Default::default()
        };
        assert_eq!(
            ctx.render().unwrap(),
            "Context:\n- files attached or in scope: 40\n- images attached: 2\n\
             - tools (files, commands, web): available"
        );
    }

    #[test]
    fn needs_combine_context_and_answers() {
        let a = Assessment {
            vision: Some(0.2),
            tools: Some(0.9),
            ..Assessment::from(TierProbs {
                probabilities: [1.0, 0.0, 0.0, 0.0],
                confidence: 1.0,
            })
        };
        let n = a.needs(&Context::default());
        assert!(!n.vision && n.tools);
        let ctx = Context {
            images: 1,
            tools: Some(false),
            prompt_tokens: Some(9),
            ..Default::default()
        };
        assert_eq!(
            a.needs(&ctx),
            Needs {
                vision: true,
                tools: false,
                context_tokens: Some(9)
            }
        );
        assert_eq!(Needs::from_context(&Context::default()), Needs::default());
    }

    #[test]
    fn effort_argmax_and_parse() {
        assert_eq!(Effort::argmax(&[0.4, 0.4, 0.2]), Effort::Medium);
        assert_eq!("HIGH".parse::<Effort>().unwrap(), Effort::High);
        assert!("max".parse::<Effort>().is_err());
    }
}
