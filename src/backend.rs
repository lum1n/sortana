use std::future::Future;
use std::pin::Pin;

use crate::rubric::Rubric;
use crate::{Result, TierProbs};

#[cfg(feature = "jev")]
pub mod jev;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One classification request, already truncated.
#[derive(Debug, Clone)]
pub struct ClassifyRequest {
    pub text: String,
    pub rubric: &'static Rubric,
}

/// A source of tier probabilities. Implement this to bring your own HTTP
/// client or provider; the classifier handles timeouts, retries, fallback,
/// caching and the final tier decision.
///
/// ```
/// use sortana::{Backend, BoxFuture, ClassifyRequest, Result, TierProbs};
///
/// struct AlwaysSimple;
///
/// impl Backend for AlwaysSimple {
///     fn classify(&self, _req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
///         Box::pin(async {
///             Ok(TierProbs { probabilities: [1.0, 0.0, 0.0, 0.0], confidence: 1.0 })
///         })
///     }
/// }
/// ```
pub trait Backend: Send + Sync {
    /// Return one probability per tier, indexed like `Tier::ALL`. They need not
    /// sum to 1; the classifier normalizes them.
    fn classify(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>>;
}

impl<B: Backend + ?Sized> Backend for std::sync::Arc<B> {
    fn classify(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
        (**self).classify(req)
    }
}

impl<B: Backend + ?Sized> Backend for Box<B> {
    fn classify(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
        (**self).classify(req)
    }
}
