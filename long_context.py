"""Utilities for summarizing long documents with sliding windows and hierarchical strategies."""

from __future__ import annotations

from typing import Callable, List

import tiktoken
from transformers import pipeline


ENCODING = tiktoken.get_encoding("cl100k_base")


def sliding_window_chunks(text: str, chunk_size: int = 1024, overlap: int = 128) -> List[str]:
    """Split ``text`` into token chunks using a sliding window."""
    tokens = ENCODING.encode(text)
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(ENCODING.decode(chunk))
    return chunks


def _summarizer_fn(model_name: str):
    try:
        return pipeline("summarization", model=model_name, local_files_only=True)
    except Exception:
        return pipeline("summarization", model=model_name)


def tree_of_thought(summaries: List[str], summarizer: Callable[[str], str]) -> str:
    """Recursively summarize pairs of summaries until one remains."""
    current = summaries
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            merged = " ".join(current[i : i + 2])
            next_level.append(summarizer(merged))
        current = next_level
    return current[0] if current else ""


def long_context_summary(
    text: str,
    model_name: str | None = None,
    chunk_size: int = 1024,
    overlap: int = 128,
    strategy: str = "vote",
) -> str:
    """Summarize ``text`` with optional long-context model or hierarchical stitching.

    If ``model_name`` corresponds to a locally available long-context model (e.g.,
    "allenai/led-base-16384" or "mistralai/Mistral-7B-32k"), it is used directly.
    Otherwise, the document is chunked with a sliding window and summarized chunk-wise
    using a shorter-context model, then recombined via ``strategy`` which may be
    "vote" (summarize concatenated chunk summaries) or "tree" (tree-of-thought).
    """

    default_model = "google/pegasus-xsum"
    chosen = model_name or default_model
    try:
        summarizer = _summarizer_fn(chosen)
        return summarizer(text, truncation=True)[0]["summary_text"]
    except Exception:
        summarizer = _summarizer_fn(default_model)

    summarize = lambda t: summarizer(t, truncation=True)[0]["summary_text"]
    chunks = sliding_window_chunks(text, chunk_size, overlap)
    chunk_summaries = [summarize(c) for c in chunks]

    if strategy == "vote":
        combined = "\n".join(chunk_summaries)
        return summarize(combined)
    if strategy == "tree":
        return tree_of_thought(chunk_summaries, summarize)
    raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    sample = "FACTS: The quick brown fox jumps over the lazy dog. ISSUE: Speed? HELD: Yes."
    print(long_context_summary(sample, chunk_size=16, overlap=4))
