"""
Ingestion: load raw markdown docs and split them into chunks.

Two chunking strategies are implemented on purpose, both switchable from
config.py:

  fixed        -- naive fixed-size character windows with overlap. This is
                  the baseline every RAG tutorial uses. It's dumb about
                  document structure (can slice a sentence, or a code
                  block, in half) but it's a fair baseline to compare
                  against.

  structure_aware -- splits on markdown headings first (so a chunk never
                  straddles two unrelated sections), then further splits
                  any section that's still too long using the same
                  fixed-size logic. This is closer to what you'd actually
                  ship, and is expected to win on retrieval precision for
                  documents that are already organized into sections --
                  which most internal docs are.

Each Chunk carries enough metadata (source file, section heading, chunk
index, strategy used) to make retrieved results explainable later, and to
let build_index.py index the *same* corpus twice under different
strategies for the eventual side-by-side comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from config import RAW_DOCS_DIR, FIXED_CHUNK_SIZE_CHARS, FIXED_CHUNK_OVERLAP_CHARS, MAX_SECTION_CHARS


@dataclass
class Chunk:
    chunk_id: str          # stable id: "<source_file>::<index>"
    text: str
    source_file: str
    section_heading: str | None
    chunk_index: int
    chunking_strategy: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


def load_raw_documents(raw_dir: Path = RAW_DOCS_DIR) -> dict[str, str]:
    """Return {filename: raw_text} for every .md file in raw_dir."""
    docs = {}
    for path in sorted(raw_dir.glob("*.md")):
        docs[path.name] = path.read_text(encoding="utf-8")
    if not docs:
        raise FileNotFoundError(f"No .md files found in {raw_dir}")
    return docs


def _fixed_size_split(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window over raw characters. Tries not to cut mid-word by
    snapping the window boundary to the nearest preceding whitespace, but
    otherwise doesn't know or care about sentence/paragraph structure."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
    return chunks


def chunk_fixed(filename: str, raw_text: str) -> list[Chunk]:
    pieces = _fixed_size_split(raw_text, FIXED_CHUNK_SIZE_CHARS, FIXED_CHUNK_OVERLAP_CHARS)
    return [
        Chunk(
            chunk_id=f"{filename}::fixed::{i}",
            text=piece,
            source_file=filename,
            section_heading=None,  # fixed chunking doesn't track this
            chunk_index=i,
            chunking_strategy="fixed",
        )
        for i, piece in enumerate(pieces)
    ]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_structure_aware(filename: str, raw_text: str) -> list[Chunk]:
    """Split on markdown headings, then re-split any section that's still
    too long. Each resulting chunk knows which heading it came from."""
    matches = list(_HEADING_RE.finditer(raw_text))

    sections: list[tuple[str | None, str]] = []
    if not matches:
        sections.append((None, raw_text))
    else:
        # Anything before the first heading (rare, but handle it)
        if matches[0].start() > 0:
            preamble = raw_text[: matches[0].start()].strip()
            if preamble:
                sections.append((None, preamble))
        for i, m in enumerate(matches):
            heading = m.group(2).strip()
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            body = raw_text[body_start:body_end].strip()
            sections.append((heading, f"{heading}\n{body}" if body else heading))

    chunks: list[Chunk] = []
    idx = 0
    for heading, section_text in sections:
        if len(section_text) <= MAX_SECTION_CHARS:
            pieces = [section_text]
        else:
            pieces = _fixed_size_split(section_text, MAX_SECTION_CHARS, FIXED_CHUNK_OVERLAP_CHARS)
        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{filename}::structaware::{idx}",
                    text=piece,
                    source_file=filename,
                    section_heading=heading,
                    chunk_index=idx,
                    chunking_strategy="structure_aware",
                )
            )
            idx += 1
    return chunks


CHUNKERS = {
    "fixed": chunk_fixed,
    "structure_aware": chunk_structure_aware,
}


def build_chunks(strategy: str, raw_dir: Path = RAW_DOCS_DIR) -> list[Chunk]:
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Options: {list(CHUNKERS)}")
    chunker = CHUNKERS[strategy]
    docs = load_raw_documents(raw_dir)
    all_chunks: list[Chunk] = []
    for filename, raw_text in docs.items():
        all_chunks.extend(chunker(filename, raw_text))
    return all_chunks


if __name__ == "__main__":
    for strategy in CHUNKERS:
        chunks = build_chunks(strategy)
        sizes = [c.char_count for c in chunks]
        print(f"[{strategy}] {len(chunks)} chunks | avg {sum(sizes)/len(sizes):.0f} chars "
              f"| min {min(sizes)} | max {max(sizes)}")
