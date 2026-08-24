"""
Demo dashboard. Imports the pipeline directly (same pattern as the
scripts/ test files) rather than calling the FastAPI service over HTTP --
one process, one command, nothing to coordinate. The FastAPI service in
src/api.py remains the separate "real backend" artifact for when you want
to show a production-style API instead of a demo UI.

Run (from the project root):
    export PYTHONPATH=src        # Windows: $env:PYTHONPATH = "src"
    streamlit run dashboard.py

Needs GROQ_API_KEY in .env for anything beyond the confidence gate itself
(the gate's own verdict -- confident/not -- works with no key at all).
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from build_index import build_retriever  # noqa: E402
from confidence import answer_with_gate  # noqa: E402
from verify import verify_citations, citation_accuracy  # noqa: E402

st.set_page_config(page_title="Internal Docs — Hybrid RAG", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Building index (one-time, ~15-20s)...")
def load_retriever():
    return build_retriever()


retriever, chunks = load_retriever()
doc_counts: dict[str, int] = {}
for c in chunks:
    doc_counts[c.source_file] = doc_counts.get(c.source_file, 0) + 1

# --- Sidebar ---
with st.sidebar:
    st.subheader("Indexed documents")
    for doc, count in sorted(doc_counts.items()):
        st.caption(f"{doc} — {count} chunks")

    st.divider()
    st.subheader("Options")
    top_k = st.slider("Chunks retrieved", 1, 10, 5)
    use_rerank = st.checkbox("Rerank (LLM judge)", value=False, help="One extra LLM call; re-scores the top-20 candidates for better ranking precision.")
    do_verify = st.checkbox("Verify citations", value=False, help="One extra LLM call per cited claim; checks each citation actually supports what it's attached to, not just that it resolves to a real chunk.")

    st.divider()
    st.caption(f"{len(chunks)} chunks · {len(doc_counts)} documents · structure-aware chunking")

st.title("🔍 Internal Docs — Hybrid RAG Search")
st.caption("Hybrid (dense + BM25) retrieval, confidence-gated generation, citation verification. Groq-backed.")

example_questions = [
    "What environment variable controls the database migration timeout?",
    "How do I immediately disable a broken feature flag?",
    "What's the company's total revenue last quarter?",
]

if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "trigger_ask" not in st.session_state:
    st.session_state.trigger_ask = False

st.write("Try one:")
cols = st.columns(len(example_questions))
for col, q in zip(cols, example_questions):
    if col.button(q, use_container_width=True):
        st.session_state.question_input = q  # setting session_state BEFORE the
        st.session_state.trigger_ask = True   # widget below is created is what
                                                # actually works in Streamlit --
                                                # passing value= after the fact
                                                # (once a key exists) is ignored.

question = st.text_input("Or ask your own:", key="question_input")
ask_clicked = st.button("Ask", type="primary")

should_ask = ask_clicked or st.session_state.trigger_ask
st.session_state.trigger_ask = False  # consume the flag so it doesn't fire again next rerun

if should_ask and question:
    with st.spinner("Thinking..."):
        try:
            answer, assessment, retrieved = answer_with_gate(retriever, question, top_k=top_k, use_reranker=use_rerank)
        except RuntimeError as e:
            st.error(f"Setup issue: {e}")
            st.stop()
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    was_gated = answer.provider == "confidence_gate"
    if was_gated:
        st.warning(
            f"⚠️ **Gated before calling the LLM** — top relevance score "
            f"{assessment.top_sparse_score:.2f} was below the {assessment.threshold:.2f} confidence threshold."
        )
    else:
        st.success(
            f"✅ Confident — top relevance score {assessment.top_sparse_score:.2f} "
            f"≥ {assessment.threshold:.2f} threshold. ({answer.provider} / {answer.model})"
        )

    st.markdown("#### Answer")
    st.write(answer.answer_text)

    if answer.citations:
        st.markdown("#### Citations")
        for cit in answer.citations:
            loc = f" › {cit.section_heading}" if cit.section_heading else ""
            st.write(f"**{cit.marker}** {cit.source_file}{loc}")

    if answer.invalid_citation_markers:
        st.error(f"Model cited markers that weren't offered as context: {answer.invalid_citation_markers}")

    if retrieved:
        with st.expander(f"Raw retrieved context ({len(retrieved)} chunks passed to the LLM)"):
            for rc in retrieved:
                loc = f" › {rc.chunk.section_heading}" if rc.chunk.section_heading else ""
                st.markdown(f"**{rc.chunk.source_file}{loc}**  _(score {rc.score:.3f}, {rc.method})_")
                st.text(rc.chunk.text[:400] + ("..." if len(rc.chunk.text) > 400 else ""))
                st.divider()

    if do_verify and not was_gated:
        with st.spinner("Verifying citations..."):
            try:
                results = verify_citations(answer.answer_text, retrieved)
            except Exception as e:
                st.error(f"Verification failed: {e}")
                results = []
        if results:
            st.markdown("#### Citation verification")
            icon = {"supported": "✅", "partial": "⚠️", "unsupported": "❌", "error": "❓"}
            for v in results:
                st.write(f"{icon.get(v.verdict, '❓')} **{v.verdict.upper()}** — {v.claim_text[:90]}...")
                st.caption(v.reasoning)
            acc = citation_accuracy(results)
            st.caption(f"Accuracy: {acc}")
        else:
            st.caption("No cited claims found to verify.")
