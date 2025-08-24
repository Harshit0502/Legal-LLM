"""Batch inference utility for summarizing the test split."""

from __future__ import annotations

import argparse
from typing import List, Dict

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas may be unavailable
    pd = None  # type: ignore

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover - transformers may be unavailable
    AutoModelForSeq2SeqLM = AutoTokenizer = None  # type: ignore

from data_utils import load_dataframes


def summarize(text: str, tokenizer, model, max_new_tokens: int = 256) -> str:
    """Generate a summary for ``text`` using the provided model."""
    ids = tokenizer(text, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the test split and write CSV")
    parser.add_argument("--model", type=str, default="google/flan-t5-base",
                        help="model name or path")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="where to save the summaries")
    args = parser.parse_args()

    if pd is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        raise SystemExit("pandas and transformers are required")

    _, _, df_test, _ = load_dataframes()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    rows: List[Dict[str, str]] = []
    for _, row in df_test.iterrows():
        summary = summarize(row["text_clean"], tok, model)
        rows.append({"doc_id": row["doc_id"], "summary": summary})

    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
