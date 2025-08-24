"""Faithfulness and factuality evaluation utilities.

This module provides heuristics for measuring the faithfulness of model-generated
summaries relative to their source documents. The checks include question
answering based faithfulness (QAG), natural language inference for factual
consistency, a hallucination-rate proxy, and length-controlled ROUGE/BERTScore
metrics. Results are returned as a pandas DataFrame for easy inspection.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import evaluate
from transformers import pipeline


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter.

    Falls back to regex-based splitting to avoid requiring NLTK data downloads.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def qag_score(summary: str, source: str, qg, qa) -> float:
    """Compute question-answer generation faithfulness score.

    Questions are generated from the summary, answered from the source, and
    compared against the summary text. The score is the fraction of generated
    questions whose answers appear in the summary.
    """
    questions = qg(summary)
    if isinstance(questions, list):
        questions = [
            q.get("generated_question", q.get("question", "")) for q in questions
        ]
    supported = 0
    for q in questions:
        ans = qa({"question": q, "context": source}).get("answer", "")
        if ans and ans.lower() in summary.lower():
            supported += 1
    return supported / max(len(questions), 1)


def nli_consistency(summary: str, source: str, nli) -> float:
    """Fraction of summary sentences entailed by the source using NLI."""
    sentences = _split_sentences(summary)
    entail = 0
    for sent in sentences:
        pred = nli({"premise": source, "hypothesis": sent})[0]["label"]
        if pred.upper() == "ENTAILMENT":
            entail += 1
    return entail / max(len(sentences), 1)


def hallucination_rate(summary: str, contexts: Sequence[str]) -> float:
    """Percent of summary sentences not found in the retrieved contexts."""
    sentences = _split_sentences(summary)
    joined = " \n".join(contexts).lower()
    unsupported = 0
    for sent in sentences:
        if sent.lower() not in joined:
            unsupported += 1
    return unsupported / max(len(sentences), 1)


def length_controlled_metrics(
    references: Sequence[str], predictions: Sequence[str]
) -> dict:
    """Compute ROUGE and BERTScore along with length statistics."""
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bert_scores = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )
    result = {key: rouge_scores[key] for key in rouge_scores}
    result["bert_f1"] = float(np.mean(bert_scores["f1"]))
    result["pred_len"] = float(np.mean([len(p.split()) for p in predictions]))
    result["ref_len"] = float(np.mean([len(r.split()) for r in references]))
    return result


def evaluate_faithfulness(
    doc_ids: Sequence[str],
    sources: Sequence[str],
    summaries: Sequence[str],
    retrieved: Optional[Sequence[Sequence[str]]] = None,
    qg_model: str = "iarfmoose/t5-base-qg-hl",
    qa_model: str = "deepset/roberta-base-squad2",
    nli_model: str = "roberta-large-mnli",
) -> pd.DataFrame:
    """Run faithfulness checks and return a pandas DataFrame of scores."""
    qg = pipeline("text2text-generation", model=qg_model)
    qa = pipeline("question-answering", model=qa_model)
    nli = pipeline("text-classification", model=nli_model)

    records = []
    for idx, doc_id in enumerate(doc_ids):
        source = sources[idx]
        summary = summaries[idx]
        ctx = retrieved[idx] if retrieved is not None else [source]
        record = {
            "doc_id": doc_id,
            "qag": qag_score(summary, source, qg, qa),
            "nli": nli_consistency(summary, source, nli),
            "hallucination": hallucination_rate(summary, ctx),
        }
        records.append(record)

    df = pd.DataFrame(records)
    length_stats = length_controlled_metrics(sources, summaries)
    for key, val in length_stats.items():
        df[key] = val
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run faithfulness and factuality checks on summaries."
    )
    parser.add_argument("--demo", action="store_true", help="Run a small demo")
    args = parser.parse_args()

    if args.demo:
        ids = ["demo-1"]
        src = ["The court found the defendant liable for negligence."]
        summ = ["The defendant was held liable for negligence by the court."]
        df = evaluate_faithfulness(ids, src, summ)
        print(df)

