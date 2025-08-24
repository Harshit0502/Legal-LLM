"""Retrieval-augmented generation utilities.

This module builds a FAISS index over chunks of ``text_clean`` and retrieves
relevant contexts for a legal question.  Retrieved chunks are formatted into a
prompt with citations and passed to a text-generation model to produce an
answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - pandas may be unavailable
    pd = None  # type: ignore

try:
    import tiktoken
except Exception as exc:  # pragma: no cover
    tiktoken = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    import faiss
except Exception as exc:  # pragma: no cover
    faiss = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as exc:  # pragma: no cover
    AutoTokenizer = AutoModelForCausalLM = None  # type: ignore


@dataclass
class Chunk:
    """Simple container for a text chunk."""

    doc_id: str
    chunk_id: int
    text: str


def _split_text(text: str, enc: "tiktoken.Encoding", chunk_size: int = 1000,
                overlap: int = 200) -> Iterable[str]:
    """Yield ``text`` split into token chunks with the given overlap."""

    if enc is None:
        # Fallback to naive split on whitespace
        tokens = text.split()
        size = chunk_size
        step = chunk_size - overlap
        for i in range(0, len(tokens), step):
            yield " ".join(tokens[i:i + size])
        return

    ids = enc.encode(text)
    step = chunk_size - overlap
    for i in range(0, len(ids), step):
        chunk_ids = ids[i:i + chunk_size]
        yield enc.decode(chunk_ids)


def chunk_dataframe(df: "pd.DataFrame", chunk_size: int = 1000,
                    overlap: int = 200,
                    tokenizer_name: str = "cl100k_base") -> List[Chunk]:
    """Convert a DataFrame of documents into overlapped chunks.

    Parameters
    ----------
    df: DataFrame with ``doc_id`` and ``text_clean`` columns.
    chunk_size: maximum tokens per chunk.
    overlap: number of tokens to overlap between chunks.
    tokenizer_name: tiktoken encoding to use.
    """

    if pd is None:
        raise ImportError("pandas is required for chunk_dataframe")

    enc = tiktoken.get_encoding(tokenizer_name) if tiktoken else None
    chunks: List[Chunk] = []
    for _, row in df.iterrows():
        doc_id = row["doc_id"]
        text = row["text_clean"]
        for idx, chunk in enumerate(_split_text(text, enc, chunk_size, overlap)):
            chunks.append(Chunk(doc_id=doc_id, chunk_id=idx, text=chunk))
    return chunks


class FaissRetriever:
    """Encode text chunks and perform nearest-neighbour search with FAISS."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_type: str = "HNSW"):
        if SentenceTransformer is None or faiss is None:
            raise ImportError("sentence-transformers and faiss are required")
        self.encoder = SentenceTransformer(model_name)
        self.index_type = index_type.upper()
        self.index = None
        self.meta: List[Chunk] = []

    def build(self, chunks: List[Chunk]):
        self.meta = chunks
        embeddings = self.encoder.encode([c.text for c in chunks])
        dim = embeddings.shape[1]
        if self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dim, 32)
            self.index.hnsw.efConstruction = 200
        else:  # IVF
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 100)
            self.index.train(embeddings)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            raise ValueError("index has not been built")
        q_emb = self.encoder.encode([query])
        scores, idxs = self.index.search(q_emb, top_k)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append((self.meta[idx], float(score)))
        return results


def build_prompt(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    system_prompt: str = "You are a careful legal analyst. Answer the question using the provided context.",
) -> str:
    """Construct a prompt with citations for the generator."""

    lines = [f"SYSTEM: {system_prompt}", f"USER: {query}", "CONTEXTS:"]
    for chunk, _score in retrieved:
        citation = f"{chunk.doc_id}:{chunk.chunk_id}"
        lines.append(f"[{citation}] {chunk.text}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


class RAGPipeline:
    """Simple retrieval-augmented generation pipeline."""

    def __init__(self, retriever: FaissRetriever,
                 generator_name: str = "google/flan-t5-base"):
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers is required for generation")
        self.retriever = retriever
        self.tokenizer = AutoTokenizer.from_pretrained(generator_name)
        self.model = AutoModelForCausalLM.from_pretrained(generator_name)

    def generate(self, query: str, top_k: int = 5, max_new_tokens: int = 128) -> Dict[str, object]:
        retrieved = self.retriever.search(query, top_k=top_k)
        prompt = build_prompt(query, retrieved)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        citations = [f"{c.doc_id}:{c.chunk_id}" for c, _ in retrieved]
        return {"answer": answer, "citations": citations}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a FAISS index and run retrieval")
    parser.add_argument("--query", type=str, help="Legal question to answer", default="What is the holding?")
    args = parser.parse_args()

    print("This is a library module; see README for usage.")
