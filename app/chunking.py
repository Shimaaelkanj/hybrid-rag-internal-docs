
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from app.config import (CHUNK_OVERLAP, CHUNK_SIZE, SEMANTIC_BREAK_PERCENTILE,
                        SEMANTIC_MIN_CHARS)
from app.loaders import LoadedDocument

STRATEGIES = ("fixed", "recursive", "semantic")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    heading: str
    page: int | None
    strategy: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def metadata(self) -> dict:
        """Chroma metadata must not contain None values."""
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "heading": self.heading,
            "page": self.page if self.page is not None else -1,
            "strategy": self.strategy,
            "char_count": self.char_count,
        }


# ---- strategy 1: fixed-size with overlap (baseline) ----
def _fixed_pieces(text: str) -> list[str]:
    splitter = CharacterTextSplitter(separator="", chunk_size=CHUNK_SIZE,
                                     chunk_overlap=CHUNK_OVERLAP, length_function=len)
    return splitter.split_text(text)


# ---- strategy 2: recursive, structure-aware ----
# The loaders already cut on section headers; this splits inside a section
# on paragraph > line > sentence > word boundaries.
def _recursive_pieces(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""])
    return splitter.split_text(text)


# ---- strategy 3: semantic (split where adjacent-sentence similarity drops) ----
def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def _semantic_pieces(text: str, embed_fn: Callable[[list[str]], np.ndarray]) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        groups = [text]
    else:
        vecs = embed_fn(sentences)                      # unit vectors -> dot = cosine
        distances = 1 - np.sum(vecs[:-1] * vecs[1:], axis=1)
        cutoff = np.percentile(distances, SEMANTIC_BREAK_PERCENTILE)

        groups, current = [], [sentences[0]]
        for sentence, dist in zip(sentences[1:], distances):
            if dist > cutoff and sum(len(s) for s in current) >= SEMANTIC_MIN_CHARS:
                groups.append(" ".join(current))
                current = [sentence]
            else:
                current.append(sentence)
        groups.append(" ".join(current))

    # safety net: no semantic chunk may exceed the size limit
    final = []
    for g in groups:
        final.extend(_recursive_pieces(g) if len(g) > CHUNK_SIZE else [g])
    return final


def build_chunks(doc: LoadedDocument, strategy: str,
                 embed_fn: Callable[[list[str]], np.ndarray] | None = None) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if strategy == "semantic" and embed_fn is None:
        raise ValueError("semantic chunking needs an embed_fn")

    chunks: list[Chunk] = []
    for section in doc.sections:
        if strategy == "fixed":
            pieces = _fixed_pieces(section.text)
        elif strategy == "recursive":
            pieces = _recursive_pieces(section.text)
        else:
            pieces = _semantic_pieces(section.text, embed_fn)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            idx = len(chunks)
            cid = hashlib.sha1(f"{doc.source}|{strategy}|{idx}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, piece, doc.source, idx, section.heading,
                                section.page, strategy))
    return chunks