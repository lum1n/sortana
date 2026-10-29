//! Sortana classifies a task description into a complexity tier for model
//! routing: [`Tier::Simple`] → Haiku, [`Tier::Medium`] → Sonnet,
//! [`Tier::Hard`] → Opus, [`Tier::VeryHard`] → Fable.
//!
//! Classification is remote. The default backend is TypeSafe AI's Jev
//! (feature `jev`), and any provider can be plugged in through [`Backend`].
//!
//! ```no_run
//! # #[cfg(feature = "jev")]
//! # async fn run() -> sortana::Result<()> {
//! let c = sortana::Classifier::builder()
//!     .jev(std::env::var("TYPESAFE_API_KEY").unwrap())
//!     .build()?;
//! let r = c.classify("Migrate our billing service to event sourcing").await?;
//! println!("{} -> {}", r.tier, c.model_for(r.tier));
//! # Ok(()) }
//! ```

pub mod backend;
#[cfg(feature = "blocking")]
pub mod blocking;
mod classifier;
mod config;
mod error;
pub mod rubric;
mod tier;

#[cfg(feature = "jev")]
pub use backend::jev::JevBackend;
pub use backend::{Backend, BoxFuture, ClassifyRequest};
pub use classifier::{Classification, Classifier, ClassifierBuilder, TRUNCATION_MARKER, truncate};
pub use config::{Config, OnError};
pub use error::{Error, Result};
pub use tier::{ModelMap, ParseTierError, Tier, TierProbs};
