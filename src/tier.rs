use std::fmt;
use std::str::FromStr;

/// Complexity tier of a task, ordered from easiest to hardest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Tier {
    Simple,
    Medium,
    Hard,
    VeryHard,
}

impl Tier {
    /// All tiers in ascending order. Index `i` matches `Classification::probabilities[i]`.
    pub const ALL: [Tier; 4] = [Tier::Simple, Tier::Medium, Tier::Hard, Tier::VeryHard];

    /// Position in [`Tier::ALL`].
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_index(i: usize) -> Option<Tier> {
        match i {
            0 => Some(Tier::Simple),
            1 => Some(Tier::Medium),
            2 => Some(Tier::Hard),
            3 => Some(Tier::VeryHard),
            _ => None,
        }
    }

    /// Stable snake_case identifier, also used as the Jev option name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Tier::Simple => "simple",
            Tier::Medium => "medium",
            Tier::Hard => "hard",
            Tier::VeryHard => "very_hard",
        }
    }

    /// The next tier up, saturating at [`Tier::VeryHard`].
    pub const fn up(self) -> Tier {
        match self {
            Tier::Simple => Tier::Medium,
            Tier::Medium => Tier::Hard,
            Tier::Hard | Tier::VeryHard => Tier::VeryHard,
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown tier {0:?}")]
pub struct ParseTierError(pub String);

impl FromStr for Tier {
    type Err = ParseTierError;

    /// Accepts `simple`, `medium`, `hard`, `very_hard` case-insensitively,
    /// with `-`, ` ` or nothing in place of `_`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let norm: String = s
            .trim()
            .chars()
            .filter(|c| !matches!(c, '_' | '-' | ' '))
            .flat_map(char::to_lowercase)
            .collect();
        match norm.as_str() {
            "simple" => Ok(Tier::Simple),
            "medium" => Ok(Tier::Medium),
            "hard" => Ok(Tier::Hard),
            "veryhard" => Ok(Tier::VeryHard),
            _ => Err(ParseTierError(s.to_owned())),
        }
    }
}

/// Raw backend output: a probability per tier (indexed like [`Tier::ALL`]) and
/// the backend's confidence in its answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierProbs {
    pub probabilities: [f32; 4],
    pub confidence: f32,
}

impl TierProbs {
    /// Tier with the highest probability. Ties go to the harder tier.
    pub fn argmax(&self) -> Tier {
        let mut best = 0;
        for i in 1..4 {
            if self.probabilities[i] >= self.probabilities[best] {
                best = i;
            }
        }
        Tier::ALL[best]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_variants() {
        for s in [
            "very_hard",
            "VeryHard",
            "very-hard",
            "Very Hard",
            "VERYHARD",
        ] {
            assert_eq!(s.parse::<Tier>().unwrap(), Tier::VeryHard);
        }
        assert_eq!(" Simple ".parse::<Tier>().unwrap(), Tier::Simple);
        assert!("extreme".parse::<Tier>().is_err());
    }

    #[test]
    fn roundtrip_and_order() {
        for t in Tier::ALL {
            assert_eq!(t.to_string().parse::<Tier>().unwrap(), t);
            assert_eq!(Tier::from_index(t.index()), Some(t));
        }
        assert!(Tier::Simple < Tier::VeryHard);
        assert_eq!(Tier::VeryHard.up(), Tier::VeryHard);
    }

    #[test]
    fn argmax_ties_go_up() {
        let p = TierProbs {
            probabilities: [0.4, 0.4, 0.1, 0.1],
            confidence: 0.5,
        };
        assert_eq!(p.argmax(), Tier::Medium);
    }
}
