"""
Builds a HybridRetriever end to end: load raw docs -> chunk -> embed ->
build dense + sparse indexes. Also persists the chunk metadata + dense
vectors to index_store/ so other scripts (eval runner, API server) don't
have to re-embed the whole corpus on every run.

Usage:
    python3 build_index.py                     # structure_aware, default embedder
    python3 build_index.py --strategy fixed     # build the fixed-chunking variant instead
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from config import INDEX_STORE_DIR, CHUNKING_STRATEGY
from ingest import build_chunks, Chunk
from embeddings import get_embedder
from sparse_index import SparseIndex
from retrieval import DenseIndex, HybridRetriever


def build_retriever(strategy: str = CHUNKING_STRATEGY) -> tuple[HybridRetriever, list[Chunk]]:
    chunks = build_chunks(strategy)
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    embedder = get_embedder()
    embedder.fit(texts)  # no-op for API-based embedders; fits IDF+centroid for spacy_local
    vectors = embedder.embed(texts)

    dense_index = DenseIndex(chunk_ids, vectors)
    sparse_index = SparseIndex(chunk_ids, texts)
    retriever = HybridRetriever(chunks, dense_index, sparse_index, embedder)
    return retriever, chunks


def persist(strategy: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
    INDEX_STORE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_STORE_DIR / f"vectors_{strategy}.npy", vectors)
    meta = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_file": c.source_file,
            "section_heading": c.section_heading,
            "chunk_index": c.chunk_index,
            "chunking_strategy": c.chunking_strategy,
            "char_count": c.char_count,
        }
        for c in chunks
    ]
    with open(INDEX_STORE_DIR / f"chunks_{strategy}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=CHUNKING_STRATEGY, choices=["fixed", "structure_aware"])
    args = parser.parse_args()

    chunks = build_chunks(args.strategy)
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    embedder = get_embedder()
    embedder.fit(texts)
    print(f"Embedding {len(texts)} chunks with '{embedder.name}' ({args.strategy} chunking)...")
    vectors = embedder.embed(texts)

    persist(args.strategy, chunks, vectors)
    print(f"Saved {len(chunks)} chunks + vectors (dim={vectors.shape[1]}) to {INDEX_STORE_DIR}/")
