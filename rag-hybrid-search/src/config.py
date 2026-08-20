"""
Central configuration for the RAG pipeline.

Design note: EMBEDDING_PROVIDER is the one thing you'll almost certainly
change before using this "for real". It defaults to "spacy_local" because
that runs fully offline with zero API cost -- useful for development,
CI, and for anyone cloning this repo without an API key. Flip it to
"openai" (see src/embeddings.py) once you have an OPENAI_API_KEY and want
production-quality embeddings. The rest of the pipeline (chunking, BM25,
RRF fusion, generation) doesn't care which one you pick -- that's the
point of hiding it behind one interface.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_STORE_DIR = PROJECT_ROOT / "index_store"
EVAL_DIR = PROJECT_ROOT / "eval"

# Load PROJECT_ROOT/.env into os.environ (GROQ_API_KEY etc.) if it exists.
# Explicit path so this works regardless of which directory you run scripts
# from -- doesn't matter if you're in scripts/, src/, or the repo root.
# Safe to call even if .env doesn't exist (just does nothing).
load_dotenv(PROJECT_ROOT / ".env")

# --- Chunking ---
# Which strategy build_index.py uses by default. Both are implemented in
# ingest.py so you can A/B them later (that comparison is its own portfolio
# artifact -- see Phase 4 of the plan).
CHUNKING_STRATEGY = os.environ.get("CHUNKING_STRATEGY", "structure_aware")

FIXED_CHUNK_SIZE_CHARS = 800
FIXED_CHUNK_OVERLAP_CHARS = 150

# For structure_aware chunking: if a section under one heading is longer
# than this, it gets split further with the same overlap as fixed chunking.
MAX_SECTION_CHARS = 1200

# --- Embeddings ---
# "spacy_local"  -> free, offline, 300-dim GloVe-style vectors (what this
#                   repo runs with out of the box, no API key needed)
# "openai"       -> text-embedding-3-small via the OpenAI API, requires
#                   OPENAI_API_KEY. Swap this in before you claim
#                   "production embeddings" in an interview.
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "spacy_local")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
SPACY_MODEL_NAME = "en_core_web_md"

# --- Retrieval ---
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
RRF_K = 60  # standard RRF damping constant from the original paper
HYBRID_TOP_K = 5  # how many fused chunks get passed to generation

# Fusion weighting, TUNED against eval/golden_qa.json (see scripts/tune_rrf_weight.py)
# for the *local spacy embedder specifically*. With that embedder, equal
# weighting (0.5/0.5) measurably underperformed sparse-only on this corpus
# (88.6% vs 95.5% Hit@5) -- the dense signal was too weak to be trusted
# equally. 0.3/0.7 keeps Hit@5 tied with sparse-only while still exercising
# the fusion path. RE-RUN THE SWEEP after switching EMBEDDING_PROVIDER to
# "openai" -- a real sentence embedder is expected to earn more weight, not
# less, and defaulting to a stale weight tuned for a weak embedder would be
# the same mistake in the other direction.
DENSE_WEIGHT = 0.3
SPARSE_WEIGHT = 0.7

# --- Generation ---
# "groq"      -> GroqCloud's fast inference API (serves open models like
#                Llama / gpt-oss / Qwen on custom LPU hardware), via its
#                OpenAI-compatible endpoint, requires GROQ_API_KEY.
#                DEFAULT -- this is what was actually asked for.
# "grok"      -> xAI's Grok (different company, similar-sounding name --
#                easy mix-up). Kept as an option since it's already built
#                and tested the same way, but NOT the default anymore.
#                Requires XAI_API_KEY.
# "anthropic" -> Claude via the Messages API, requires ANTHROPIC_API_KEY
# "openai"    -> GPT-4o / GPT-4o-mini, requires OPENAI_API_KEY
#
# None of these are reachable from THIS build sandbox -- verified for
# api.x.ai and api.groq.com specifically (both return
# x-deny-reason: host_not_allowed from the egress proxy here). Every
# backend below is written correctly against each provider's documented
# API but none has had a live network round-trip inside this environment.
GENERATION_PROVIDER = os.environ.get("GENERATION_PROVIDER", "groq")

# Model IDs move fast, especially on Groq -- they host other companies'
# open models and add/deprecate frequently (e.g. llama-3.3-70b-versatile
# and llama-3.1-8b-instant were both deprecated as of mid-2026, in favor
# of the gpt-oss family). Don't trust this hardcoded value for long --
# check what's actually live yourself:
#   curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GROK_MODEL = "grok-4.6"
XAI_BASE_URL = "https://api.x.ai/v1"

ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_GENERATION_MODEL = "gpt-4o-mini"

# Below what combined-signal confidence does generation refuse to answer
# and say so instead of guessing (see src/generate.py). Tuned later against
# eval/golden_qa.json's 5 unanswerable cases once this is live-tested.
MIN_CONTEXT_RELEVANCE = 0.15
