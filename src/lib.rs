//! Sortana classifies a task description into a complexity tier
//! ([`Tier::Simple`], [`Tier::Medium`], [`Tier::Hard`], [`Tier::VeryHard`]) and
//! picks the model for that tier from a [`ModelMap`]. Bundled [`Lineup`]s cover
//! the major model makers, resolved to ids on any [`Host`] in the bundled
//! models.dev catalogue (including opencode's `host/model` format). Anthropic's
//! lineup is the default; any other models work through a custom map.
//!
//! Classification is remote. The default backend is TypeSafe AI's Jev
//! (feature `jev`), and any provider can be plugged in through [`Backend`].
//!
//! ```no_run
//! # #[cfg(feature = "jev")]
//! # async fn run() -> sortana::Result<()> {
//! let c = sortana::Classifier::builder()
//!     .jev(std::env::var("TYPESAFE_API_KEY").unwrap())
//!     .models(sortana::ModelMap::opencode("openai", "openrouter")?)
//!     .build()?;
//! let r = c.classify("Migrate our billing service to event sourcing").await?;
//! println!("{} -> {}", r.tier, r.model);
//! # Ok(()) }
//! ```

pub mod backend;
#[cfg(feature = "blocking")]
pub mod blocking;
mod catalog;
mod classifier;
mod config;
mod error;
mod models;
pub mod rubric;
mod tier;

#[cfg(feature = "jev")]
pub use backend::jev::JevBackend;
pub use backend::{Backend, BoxFuture, ClassifyRequest};
pub use classifier::{Classification, Classifier, ClassifierBuilder, TRUNCATION_MARKER, truncate};
pub use config::{Config, OnError};
pub use error::{Error, Result};
pub use models::{CATALOG_DATE, Host, Lineup, ModelMap, PresetError, host_model_id};
pub use tier::{ParseTierError, Tier, TierProbs};
