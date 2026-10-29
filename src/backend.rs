use std::future::Future;
use std::pin::Pin;

use crate::rubric::Rubric;
use crate::{Assessment, Result, TierProbs};

#[cfg(feature = "jev")]
pub mod jev;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One classification request, already truncated.
#[derive(Debug, Clone)]
pub struct ClassifyRequest {
    /// The task text.
    pub text: String,
    /// The rendered [`Context`](crate::Context) section, if any.
    pub context: Option<String>,
    pub rubric: &'static Rubric,
    /// Extra answers wanted from [`Backend::assess`].
    pub ask: Ask,
}

impl ClassifyRequest {
    /// The task followed by its context section: what a backend should rate.
    pub fn state(&self) -> String {
        match &self.context {
            Some(c) => format!("{}\n\n{c}", self.text),
            None => self.text.clone(),
        }
    }
}

/// Which answers besides the tier a request wants. Questions are in
/// [`rubric`](crate::rubric): [`EFFORT`](crate::rubric::EFFORT) for `effort`,
/// [`VISION`](crate::rubric::VISION) and [`TOOLS`](crate::rubric::TOOLS) for `needs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ask {
    pub effort: bool,
    pub needs: bool,
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

    /// Tier probabilities plus the answers `req.ask` wants. The classifier
    /// calls this. The default calls [`classify`](Self::classify) and gives
    /// no extra answers; override it to support them.
    fn assess(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<Assessment>> {
        let probs = self.classify(req);
        Box::pin(async move { probs.await.map(Assessment::from) })
    }
}

impl<B: Backend + ?Sized> Backend for std::sync::Arc<B> {
    fn classify(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
        (**self).classify(req)
    }

    fn assess(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<Assessment>> {
        (**self).assess(req)
    }
}

impl<B: Backend + ?Sized> Backend for Box<B> {
    fn classify(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<TierProbs>> {
        (**self).classify(req)
    }

    fn assess(&self, req: ClassifyRequest) -> BoxFuture<'_, Result<Assessment>> {
        (**self).assess(req)
    }
}
