from typing import Dict

import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def textrank_summary(text: str, sentences: int = 3) -> str:
    """Return a TextRank summary of ``text`` using ``sentences`` sentences."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join(str(s) for s in summary)


def pretrained_summary(text: str, model_name: str = "google/pegasus-xsum") -> str:
    """Generate an abstractive summary using a pretrained ``transformers`` model."""
    summarizer = pipeline("summarization", model=model_name)
    return summarizer(text, truncation=True)[0]["summary_text"]


def evaluate_baselines(
    df_val: pd.DataFrame,
    sample_size: int = 32,
    model_name: str = "google/pegasus-xsum",
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate extractive and abstractive baselines on a validation sample."""
    sample = df_val.sample(min(sample_size, len(df_val)), random_state=seed)
    texts = sample["text_clean"].tolist()
    refs = sample["summary_clean"].tolist()

    # Extractive baseline
    ext_summaries = [textrank_summary(t) for t in texts]

    # Abstractive baseline
    abstractive = pipeline("summarization", model=model_name)
    abs_summaries = [abstractive(t, truncation=True)[0]["summary_text"] for t in texts]

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def _avg_rouge(preds):
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

    return {
        "extractive": {"rouge": ext_rouge, "bertscore": ext_bert},
        "abstractive": {"rouge": abs_rouge, "bertscore": abs_bert},
    }


if __name__ == "__main__":
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
