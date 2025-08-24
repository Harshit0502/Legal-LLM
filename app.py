"""FastAPI app exposing summarization and QA endpoints."""

from __future__ import annotations

import os
from typing import Dict, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover - FastAPI may be unavailable
    FastAPI = HTTPException = BaseModel = None  # type: ignore

if FastAPI is None or BaseModel is None:  # pragma: no cover - fail early if deps missing
    raise SystemExit("fastapi is required to run the API")

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover - transformers may be unavailable
    AutoModelForSeq2SeqLM = AutoTokenizer = None  # type: ignore

try:
    from rag import RAGPipeline, FaissRetriever, chunk_dataframe
    import pandas as pd
except Exception:  # pragma: no cover - optional deps may be missing
    RAGPipeline = FaissRetriever = chunk_dataframe = None  # type: ignore
    pd = None  # type: ignore

try:
    from policy import (
        DISCLAIMER,
        is_request_for_legal_advice,
        policy_check,
    )
except Exception:  # pragma: no cover - policy utilities may be missing
    DISCLAIMER = ""

    def is_request_for_legal_advice(_: str) -> bool:  # type: ignore
        return False

    def policy_check(_: str, __: list) -> list:  # type: ignore
        return []

MODEL_NAME = os.getenv("SFT_MODEL_NAME", "google/flan-t5-base")

# Load summarization model if possible
if AutoTokenizer and AutoModelForSeq2SeqLM:
    try:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    except Exception:  # pragma: no cover - model may be missing
        _tok = _model = None
else:  # pragma: no cover
    _tok = _model = None

# Build a tiny RAG pipeline for demo if dependencies are available
_rag: Optional[RAGPipeline]
if RAGPipeline and FaissRetriever and chunk_dataframe and pd is not None:
    try:
        _df = pd.DataFrame({"doc_id": ["demo"], "text_clean": ["Example legal passage for retrieval."]})
        _chunks = chunk_dataframe(_df)
        _retriever = FaissRetriever()
        _retriever.build(_chunks)
        _rag = RAGPipeline(_retriever)
    except Exception:  # pragma: no cover - build may fail
        _rag = None
else:  # pragma: no cover
    _rag = None

# Instantiate FastAPI app
app = FastAPI(title="Legal-LLM API") if FastAPI else None  # type: ignore


class SummarizeRequest(BaseModel):  # type: ignore[misc]
    text: str
    doc_id: Optional[str] = "unknown"

class QARequest(BaseModel):  # type: ignore[misc]
    question: str

@app.post("/summarize")  # type: ignore[misc]
def summarize(req: SummarizeRequest) -> Dict[str, object]:
    if _tok is None or _model is None:
        raise HTTPException(status_code=500, detail="summarization model unavailable")
    ids = _tok(req.text, return_tensors="pt").input_ids
    out = _model.generate(ids, max_new_tokens=256)
    summary = _tok.decode(out[0], skip_special_tokens=True)
    citations = [req.doc_id] if req.doc_id else []
    flags = policy_check(summary, citations)
    return {
        "summary": summary,
        "citations": citations,
        "policy_flags": flags,
        "disclaimer": DISCLAIMER,
    }

@app.post("/qa")  # type: ignore[misc]
def qa(req: QARequest) -> Dict[str, object]:
    if _rag is None:
        raise HTTPException(status_code=500, detail="RAG pipeline unavailable")
    if is_request_for_legal_advice(req.question):
        raise HTTPException(
            status_code=403,
            detail="I cannot provide legal advice. Responses are for educational purposes only.",
        )
    result = _rag.generate(req.question)
    flags = policy_check(result["answer"], result["citations"])
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "policy_flags": flags,
        "disclaimer": DISCLAIMER,
    }

if __name__ == "__main__":
    if FastAPI is None:
        raise SystemExit("fastapi is not installed")
    import uvicorn  # type: ignore

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
