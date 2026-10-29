//! Synchronous wrapper around [`crate::Classifier`].

use crate::{Classification, Context, Error, ModelMap, Result};

/// Runs a [`crate::Classifier`] on its own single-threaded Tokio runtime.
///
/// Do not call it from inside an async runtime; Tokio panics on nested
/// `block_on`. Use the async classifier there.
#[derive(Debug)]
pub struct Classifier {
    inner: crate::Classifier,
    rt: tokio::runtime::Runtime,
}

impl Classifier {
    pub fn new(inner: crate::Classifier) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::Config(format!("building Tokio runtime: {e}")))?;
        Ok(Classifier { inner, rt })
    }

    pub fn classify(&self, text: &str) -> Result<Classification> {
        self.rt.block_on(self.inner.classify(text))
    }

    pub fn classify_with(&self, text: &str, ctx: &Context) -> Result<Classification> {
        self.rt.block_on(self.inner.classify_with(text, ctx))
    }

    pub fn classify_for(
        &self,
        text: &str,
        ctx: &Context,
        models: &ModelMap,
    ) -> Result<Classification> {
        self.rt.block_on(self.inner.classify_for(text, ctx, models))
    }

    pub fn classify_many<I, S>(&self, texts: I) -> Vec<Result<Classification>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.rt.block_on(self.inner.classify_many(texts))
    }

    /// The wrapped async classifier.
    pub fn inner(&self) -> &crate::Classifier {
        &self.inner
    }
}
