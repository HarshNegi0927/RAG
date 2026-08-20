"""
Embedding backends behind one interface: `Embedder.embed(texts) -> np.ndarray`.

Why two backends instead of just calling OpenAI:
This project needs to run and be testable without anyone (including a
recruiter cloning the repo) needing to pay for API calls just to see it
work. SpacyLocalEmbedder gives real, legitimate dense vectors (averaged
300-dim GloVe-style word vectors, trained on a large web/news corpus) with
zero external dependency beyond the one-time model download. It's not
transformer-quality, but it's genuinely semantic -- synonyms and related
concepts land close together in vector space, which is the property
hybrid retrieval actually needs to demonstrate (see scripts/compare_retrieval.py
for a query where this beats keyword-only search).

OpenAIEmbedder is the "swap this in for production" path -- same
interface, so nothing else in the codebase changes.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from config import EMBEDDING_PROVIDER, OPENAI_EMBEDDING_MODEL, SPACY_MODEL_NAME


class Embedder(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 array, L2-normalized rows."""
        ...

    def fit(self, corpus_texts: list[str]) -> None:
        """Optional: learn any corpus-level statistics before embedding.
        Default is a no-op (e.g. OpenAIEmbedder doesn't need this -- a
        transformer was already trained on a huge, diverse corpus, so a
        single project's ~60 chunks isn't going to shift its geometry).
        build_index.py always calls this before embed() so backends that
        DO need it (see SpacyLocalEmbedder) get the chance."""
        return

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class SpacyLocalEmbedder(Embedder):
    """Free, offline, deterministic. Default embedder for this repo.

    IMPORTANT LIMITATION, found and fixed while building this: naively
    averaging every token's word vector (what spaCy's `doc.vector` does
    out of the box) produces a collapsed embedding space for multi-
    sentence chunks -- measured on this corpus, unrelated chunk pairs
    averaged ~0.79 cosine similarity, which means almost everything looks
    similar to almost everything, and a chunk full of generic connective
    language (e.g. "does this do what the PR says, are there tests...")
    becomes a "hub" that dense search returns for nearly any query. This
    is a known failure mode of naive mean-pooled word vectors, not a bug
    specific to spaCy's vectors.

    Two standard, lightweight fixes are applied here:
      1. IDF-weighted pooling instead of a uniform average, so common
         words contribute less and corpus-specific terms (env var names,
         service names, policy-specific vocabulary) contribute more.
      2. Mean-centering: subtract the corpus's average embedding before
         normalizing, which removes the dominant shared direction that
         was causing the collapse (a simplified version of the
         top-component-removal trick from the SIF sentence-embedding
         paper / embedding "whitening" methods).

    Both are fit once on the chunk corpus (see fit()) and reused for
    every later query embedding, the same way a fitted TF-IDF vectorizer
    is reused at inference time.

    This is a real, citable limitation of static word-vector averaging
    versus transformer-based sentence embeddings (which model word order
    and context, not just a bag of words) -- swapping EMBEDDING_PROVIDER
    to "openai" removes this class of problem entirely, at the cost of
    needing an API key.
    """

    name = "spacy_local"

    def __init__(self, model_name: str = SPACY_MODEL_NAME):
        import spacy  # imported lazily so OpenAIEmbedder doesn't need spacy installed

        self._nlp = spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])
        self.dim = self._nlp.vocab.vectors_length
        self._idf: dict[str, float] = {}
        self._default_idf: float = 1.0
        self._centroid: np.ndarray = np.zeros(self.dim, dtype=np.float32)
        self._fitted = False

    def fit(self, corpus_texts: list[str]) -> None:
        # Document frequency per lowercase token across the chunk corpus.
        df: dict[str, int] = {}
        n = len(corpus_texts)
        for text in corpus_texts:
            tokens = {
                t.lower_ for t in self._nlp.tokenizer(text) if t.is_alpha and not t.is_stop
            }
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1

        # Smoothed IDF, sklearn-style: log((1+n)/(1+df)) + 1 -- always positive.
        self._idf = {tok: float(np.log((1 + n) / (1 + d)) + 1.0) for tok, d in df.items()}
        self._default_idf = float(np.mean(list(self._idf.values()))) if self._idf else 1.0

        # First pass with IDF weights but zero centroid, purely to compute
        # what the centroid *is* -- then embed() below applies it for real.
        self._fitted = True
        raw_vectors = self._pooled_vectors(corpus_texts)
        self._centroid = raw_vectors.mean(axis=0)

    def _pooled_vectors(self, texts: list[str]) -> np.ndarray:
        """IDF-weighted mean pooling, no centering / normalization yet."""
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, doc in enumerate(self._nlp.pipe(texts, batch_size=32)):
            weighted_sum = np.zeros(self.dim, dtype=np.float32)
            weight_total = 0.0
            for token in doc:
                if not token.has_vector or token.is_stop or not token.is_alpha:
                    continue
                w = self._idf.get(token.lower_, self._default_idf) if self._fitted else 1.0
                weighted_sum += token.vector * w
                weight_total += w
            if weight_total > 0:
                vectors[i] = weighted_sum / weight_total
        return vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        pooled = self._pooled_vectors(texts)
        if self._fitted:
            pooled = pooled - self._centroid
        return self._l2_normalize(pooled)


class OpenAIEmbedder(Embedder):
    """Production backend. Requires OPENAI_API_KEY. Not exercised in this
    sandbox (no network path to api.openai.com here + no key configured),
    but this is the real integration code -- point OPENAI_BASE_URL /
    OPENAI_API_KEY at a live account and EMBEDDING_PROVIDER=openai to use it."""

    name = "openai"

    def __init__(self, model: str = OPENAI_EMBEDDING_MODEL):
        from openai import OpenAI  # imported lazily

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Export it, or switch back to EMBEDDING_PROVIDER=spacy_local."
            )
        self._client = OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        # OpenAI's embeddings endpoint accepts batches directly; chunk
        # defensively in case the caller passes a very large batch.
        out: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return self._l2_normalize(np.array(out, dtype=np.float32))


_REGISTRY = {
    "spacy_local": SpacyLocalEmbedder,
    "openai": OpenAIEmbedder,
}


def get_embedder(provider: str | None = None) -> Embedder:
    provider = provider or EMBEDDING_PROVIDER
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown embedding provider '{provider}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[provider]()


if __name__ == "__main__":
    embedder = get_embedder()
    print(f"Using embedder: {embedder.name}")
    sample = ["rotate an API key", "how do I renew credentials", "chocolate chip cookie recipe"]
    vecs = embedder.embed(sample)
    print("shape:", vecs.shape)
    sims = vecs @ vecs.T
    print("cosine sim(rotate api key, renew credentials) =", round(float(sims[0, 1]), 3))
    print("cosine sim(rotate api key, cookie recipe)      =", round(float(sims[0, 2]), 3))
