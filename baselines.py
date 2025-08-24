from typing import Dict, List
import os

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas may be missing
    pd = None  # type: ignore

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
except Exception:  # pragma: no cover - sumy may be missing
    PlaintextParser = Tokenizer = TextRankSummarizer = None  # type: ignore

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - transformers may be missing
    pipeline = None  # type: ignore

try:
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover - rouge_score may be missing
    rouge_scorer = None  # type: ignore

try:
    from bert_score import score as bert_score
except Exception:  # pragma: no cover - bert_score may be missing
    bert_score = None  # type: ignore



def textrank_summary(text: str, sentences: int = 3) -> str:
    """Return a TextRank summary of ``text`` using ``sentences`` sentences."""
    if PlaintextParser is None or Tokenizer is None or TextRankSummarizer is None:
        raise ImportError("sumy is required for TextRank summarization")
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join(str(s) for s in summary)


def pretrained_summary(text: str, model_name: str = "google/pegasus-xsum") -> str:
    """Generate an abstractive summary using a pretrained ``transformers`` model."""
    if pipeline is None:
        raise ImportError("transformers is required for pretrained_summary")
    summarizer = pipeline("summarization", model=model_name)
    return summarizer(text, truncation=True)[0]["summary_text"]


def evaluate_baselines(
    df_val: "pd.DataFrame",
    sample_size: int = 32,
    model_name: str = "google/pegasus-xsum",
    seed: int = 42,
    output_dir: str | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate extractive and abstractive baselines on a validation sample."""
    if pd is None or rouge_scorer is None or bert_score is None:
        raise ImportError("pandas, rouge_score, and bert_score are required")
    sample = df_val.sample(min(sample_size, len(df_val)), random_state=seed)
    texts = sample["text_clean"].tolist()
    refs = sample["summary_clean"].tolist()

    # Extractive baseline
    ext_summaries = [textrank_summary(t) for t in texts]

    # Abstractive baseline
    abs_summaries: List[str]
    if pipeline is None:
        raise ImportError("transformers is required for abstractive baseline")
    abstractive = pipeline("summarization", model=model_name)
    abs_summaries = [abstractive(t, truncation=True)[0]["summary_text"] for t in texts]

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def _avg_rouge(preds: List[str]):
        metrics = {k: 0.0 for k in ["rouge1", "rouge2", "rougeL"]}
        for pred, ref in zip(preds, refs):
            scores = rouge.score(ref, pred)
            for k in metrics:
                metrics[k] += scores[k].fmeasure
        n = len(refs)
        return {k: v / n for k, v in metrics.items()}

    ext_rouge = _avg_rouge(ext_summaries)
    abs_rouge = _avg_rouge(abs_summaries)

    P, R, F = bert_score(ext_summaries, refs, lang="en")
    ext_bert = {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F.mean()),
    }
    P, R, F = bert_score(abs_summaries, refs, lang="en")
    abs_bert = {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F.mean()),
    }

    results = {
        "extractive": {"rouge": ext_rouge, "bertscore": ext_bert},
        "abstractive": {"rouge": abs_rouge, "bertscore": abs_bert},
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        rows = []
        for name, metrics in results.items():
            row = {
                "model": name,
                **{f"rouge_{k}": v for k, v in metrics["rouge"].items()},
                "bert_f1": metrics["bertscore"]["f1"],
            }
            rows.append(row)
        csv_path = os.path.join(output_dir, "baseline_eval.csv")
        import csv as _csv

        with open(csv_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        try:
            import matplotlib.pyplot as plt

            df_plot = pd.DataFrame(rows)
            ax = df_plot.set_index("model").plot(kind="bar")
            ax.set_ylabel("score")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "baseline_eval.png"))
            plt.close(fig)
        except Exception:  # pragma: no cover - matplotlib may be missing
            pass

    return results


if __name__ == "__main__":
    if pd is None:
        raise SystemExit("pandas is required for the baseline demo")

    sample = {
        "doc_id": [1, 2, 3],
        "text": [
            "FACTS: The quick brown fox jumps over the lazy dog.",
            "ISSUE: Whether the fox was quick enough to jump.",
            "HELD: The fox succeeded.",
        ],
        "summary": [
            "A fox jumped over a dog.",
            "The speed of the fox was in question.",
            "The fox won.",
        ],
    }
    df = pd.DataFrame(sample)
    df["text_clean"] = df["text"]
    df["summary_clean"] = df["summary"]
    metrics = evaluate_baselines(df, sample_size=2)
    print(metrics)
