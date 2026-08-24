"""
FastAPI service. Wraps the existing pipeline (confidence gate ->
retrieval -> generation -> optional citation verification) behind HTTP
endpoints. Builds the index ONCE at startup (via lifespan), not per
request -- embedding 63 chunks takes a couple seconds with the local
embedder; doing that on every /v1/ask call would be wasteful and slow.

Run (from the project root):
    uvicorn api:app --app-dir src --reload --port 8000

Then:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/v1/ask -H "Content-Type: application/json" \
         -d '{"question": "What command do I use to roll back a bad deploy?"}'
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from build_index import build_retriever
from confidence import answer_with_gate
from verify import verify_citations, citation_accuracy
from config import HYBRID_TOP_K

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever, chunks = build_retriever()
    _state["retriever"] = retriever
    _state["chunks"] = chunks
    yield
    _state.clear()


app = FastAPI(title="RAG Hybrid Search API", version="0.1.0", lifespan=lifespan)


# --- Request/response models ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=HYBRID_TOP_K, ge=1, le=20)
    verify: bool = Field(default=False, description="Run citation verification too (extra LLM calls, one per cited claim).")
    rerank: bool = Field(default=False, description="LLM-judge rerank the top-20 candidates before generating (one extra LLM call).")


class CitationOut(BaseModel):
    marker: str
    source_file: str
    section_heading: str | None


class VerificationOut(BaseModel):
    claim_text: str
    markers: list[str]
    sources: list[str]
    verdict: str
    reasoning: str


class ConfidenceOut(BaseModel):
    top_sparse_score: float
    threshold: float
    is_confident: bool


class AskResponse(BaseModel):
    query: str
    answer: str
    gated: bool
    confidence: ConfidenceOut
    citations: list[CitationOut]
    invalid_citation_markers: list[str]
    provider: str
    model: str
    verification: list[VerificationOut] | None = None
    citation_accuracy: dict | None = None


class DocumentOut(BaseModel):
    source_file: str
    chunk_count: int


# --- Routes ---

@app.get("/health")
def health():
    chunks = _state.get("chunks")
    if chunks is None:
        raise HTTPException(status_code=503, detail="Index not built yet")
    return {"status": "ok", "chunks_indexed": len(chunks)}


@app.get("/v1/documents", response_model=list[DocumentOut])
def list_documents():
    chunks = _state.get("chunks", [])
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.source_file] = counts.get(c.source_file, 0) + 1
    return [DocumentOut(source_file=k, chunk_count=v) for k, v in sorted(counts.items())]


@app.get("/v1/stats")
def stats():
    chunks = _state.get("chunks", [])
    return {
        "total_chunks": len(chunks),
        "total_documents": len({c.source_file for c in chunks}),
        "chunking_strategy": chunks[0].chunking_strategy if chunks else None,
    }


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retriever = _state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Index not ready yet")

    try:
        answer, assessment, retrieved = answer_with_gate(retriever, req.question, top_k=req.top_k, use_reranker=req.rerank)
    except RuntimeError as e:
        # Missing/misconfigured API key -- our own clean error from generate.py, not a crash
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream LLM call failed: {e}")

    was_gated = answer.provider == "confidence_gate"

    verification_out, accuracy_out = None, None
    if req.verify and not was_gated:
        try:
            results = verify_citations(answer.answer_text, retrieved)
            verification_out = [
                VerificationOut(claim_text=r.claim_text, markers=r.markers, sources=r.sources,
                                 verdict=r.verdict, reasoning=r.reasoning)
                for r in results
            ]
            accuracy_out = citation_accuracy(results)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Citation verification failed: {e}")

    return AskResponse(
        query=req.question,
        answer=answer.answer_text,
        gated=was_gated,
        confidence=ConfidenceOut(
            top_sparse_score=assessment.top_sparse_score,
            threshold=assessment.threshold,
            is_confident=assessment.is_confident,
        ),
        citations=[
            CitationOut(marker=c.marker, source_file=c.source_file, section_heading=c.section_heading)
            for c in answer.citations
        ],
        invalid_citation_markers=answer.invalid_citation_markers,
        provider=answer.provider,
        model=answer.model,
        verification=verification_out,
        citation_accuracy=accuracy_out,
    )
