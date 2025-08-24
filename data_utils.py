import os
import re
import unicodedata
import hashlib
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import tiktoken

CONFIG = {
    "train_path": "train.csv",
    "val_path": "val.csv",
    "test_path": "test.csv",
}


def clean_text(x: str, anonymize: bool = True) -> Tuple[str, Dict[str, str]]:
    """Return normalized text and a mapping of anonymized names.

    Parameters
    ----------
    x : str
        Input string to clean.
    anonymize : bool, optional
        Whether to replace detected names with placeholders, by default True.

    Returns
    -------
    Tuple[str, Dict[str, str]]
        The cleaned text and a mapping from placeholder to original name.
    """

    if not isinstance(x, str):
        return x, {}

    text = unicodedata.normalize("NFKC", x)

    # Standardize quotes, dashes and bullet symbols
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "−": "-",
        "•": "-",
        "·": "-",
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    # Remove page numbers, line numbers, and simple header/footer patterns
    text = re.sub(r"\bPage\s+\d+(?:\s+of\s+\d+)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^(?:Header|Footer):.*$", "", text)

    # Normalize whitespace and collapse multiple newlines
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    name_map: Dict[str, str] = {}
    if anonymize:
        judge_idx = 1
        def_idx = 1

        def replace_judge(match: re.Match) -> str:
            nonlocal judge_idx
            name = match.group(1)
            placeholder = f"JUDGE_{judge_idx}"
            judge_idx += 1
            name_map[placeholder] = name
            return placeholder

        def replace_defendant(match: re.Match) -> str:
            nonlocal def_idx
            name = match.group(0)
            placeholder = f"DEFENDANT_{def_idx}"
            def_idx += 1
            name_map[placeholder] = name
            return placeholder

        # Replace judges/justices first to avoid double replacement
        text = re.sub(
            r"(?:Judge|Justice) ([A-Z][a-z]+ [A-Z][a-z]+)",
            replace_judge,
            text,
        )
        # Replace remaining capitalized first+last names
        text = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", replace_defendant, text)

    return text, name_map

def _read_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for {path}")


def _validate_dataframe(df: pd.DataFrame, name: str) -> None:
    required_cols = ["doc_id", "text", "summary"]
    if list(df.columns) != required_cols:
        raise AssertionError(
            f"{name} must have columns {required_cols}, got {list(df.columns)}"
        )

    # Ensure 'text' and 'summary' are non-empty strings after strip
    for col in ["text", "summary"]:
        if not df[col].map(lambda x: isinstance(x, str) and x.strip() != "").all():
            raise AssertionError(f"Column '{col}' in {name} contains empty strings or non-str values")

    print(f"{name} shape: {df.shape}")
    print(f"{name} null counts:\n{df.isna().sum()}")
    print(f"{name} examples:\n{df.head(2)}\n")


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text and summary columns to a copy of ``df``."""

    df = df.copy()
    df["text_clean"], df["text_map"] = zip(*df["text"].map(clean_text))
    df["summary_clean"], df["summary_map"] = zip(*df["summary"].map(clean_text))
    return df


def _simhash(text: str) -> int:
    """Return a 64-bit SimHash fingerprint of ``text``."""
    tokens = text.split()
    if not tokens:
        return 0
    shingles = (
        [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
        if len(tokens) >= 3
        else tokens
    )
    v = [0] * 64
    for sh in shingles:
        h = int(hashlib.md5(sh.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            bit = 1 << i
            v[i] += 1 if h & bit else -1
    fingerprint = 0
    for i, val in enumerate(v):
        if val >= 0:
            fingerprint |= 1 << i
    return fingerprint


def _simhash_similarity(a: int, b: int) -> float:
    """Similarity between two SimHash fingerprints."""
    return 1 - (bin(a ^ b).count("1") / 64)


def drop_near_duplicates(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    threshold: float = 0.9,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Remove near-duplicate rows from ``df_train`` using SimHash.

    Rows in ``df_train`` that are similar to each other or to rows in ``df_val``/
    ``df_test`` are dropped to prevent data leakage. Returns the deduplicated
    DataFrame and a mapping from original index to ``doc_id`` for dropped rows.
    """

    val_hashes = df_val["text_clean"].map(_simhash).to_numpy()
    test_hashes = df_test["text_clean"].map(_simhash).to_numpy()

    drop_idx: list[int] = []
    drop_map: Dict[int, int] = {}
    kept_hashes: list[int] = []

    for idx, row in df_train.iterrows():
        h = _simhash(row["text_clean"])
        similar_to_val = any(_simhash_similarity(h, vh) >= threshold for vh in val_hashes)
        similar_to_test = any(
            _simhash_similarity(h, th) >= threshold for th in test_hashes
        )
        similar_to_train = any(
            _simhash_similarity(h, kh) >= threshold for kh in kept_hashes
        )
        if similar_to_val or similar_to_test or similar_to_train:
            drop_idx.append(idx)
            drop_map[idx] = row["doc_id"]
        else:
            kept_hashes.append(h)

    if drop_idx:
        print(f"Removed {len(drop_idx)} near-duplicate rows from train")
    else:
        print("No near-duplicates detected in train")

    df_dedup = df_train.drop(index=drop_idx).reset_index(drop=True)
    return df_dedup, drop_map

def load_dataframes(
    df_train: Optional[pd.DataFrame] = None,
    df_val: Optional[pd.DataFrame] = None,
    df_test: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, str]] = None,
    dup_threshold: float = 0.9,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    cfg = CONFIG if config is None else config
    if df_train is None:
        df_train = _read_dataframe(cfg["train_path"])
    if df_val is None:
        df_val = _read_dataframe(cfg["val_path"])
    if df_test is None:
        df_test = _read_dataframe(cfg["test_path"])

    _validate_dataframe(df_train, "df_train")
    _validate_dataframe(df_val, "df_val")
    _validate_dataframe(df_test, "df_test")

    df_train = _clean_dataframe(df_train)
    df_val = _clean_dataframe(df_val)
    df_test = _clean_dataframe(df_test)

    df_train, dropped_map = drop_near_duplicates(
        df_train, df_val, df_test, threshold=dup_threshold
    )

    for name, df in [("df_train", df_train), ("df_val", df_val), ("df_test", df_test)]:
        print(
            f"{name} cleaned samples:\n"
            f"{df[['text', 'text_clean', 'summary', 'summary_clean']].head(3)}\n"
        )

    return df_train, df_val, df_test, dropped_map


def analyze_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str = "analysis",
) -> None:
    """Plot length histograms and report vocabulary overlap."""

    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding("cl100k_base")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

    def _percentiles(values: np.ndarray) -> str:
        p = np.percentile(values, [50, 90, 95, 99])
        return "p50={:.1f}, p90={:.1f}, p95={:.1f}, p99={:.1f}".format(*p)

    def _lengths(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        chars = series.str.len().to_numpy()
        tokens = series.map(lambda x: len(enc.encode(x))).to_numpy()
        return chars, tokens

    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for col in ["text_clean", "summary_clean"]:
            chars, tokens = _lengths(df[col])
            for arr, name in [(chars, "char"), (tokens, "token")]:
                plt.figure()
                plt.hist(arr, bins=50)
                plt.title(f"{split} {col} {name} lengths")
                plt.xlabel(f"{name} count")
                plt.ylabel("frequency")
                plt.tight_layout()
                out = os.path.join(output_dir, f"{split}_{col}_{name}_hist.png")
                plt.savefig(out)
                plt.close()
                print(f"{split} {col} {name} percentiles: {_percentiles(arr)}")

    def _lemma_set(texts: pd.Series) -> set:
        return {
            tok.lemma_.lower()
            for doc in nlp.pipe(texts.tolist(), batch_size=100)
            for tok in doc
            if tok.is_alpha
        }

    vocabs = {
        split: {
            col: _lemma_set(df[col])
            for col in ["text_clean", "summary_clean"]
        }
        for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]
    }

    def _jaccard(a: set, b: set) -> float:
        return len(a & b) / len(a | b) if a or b else 0.0

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for col in ["text_clean", "summary_clean"]:
        for s1, s2 in pairs:
            score = _jaccard(vocabs[s1][col], vocabs[s2][col])
            print(f"Jaccard({s1},{s2}) for {col}: {score:.3f}")


if __name__ == "__main__":
    sample = {
        "doc_id": [1, 2, 3],
        "text": [
            "FACTS: Judge Alice Smith heard the case. Page 1\nJohn Doe appeared.",
            "ISSUE: Whether — given the evidence — the defendant Jane Roe was liable.",
            "HELD: Justice Bob Jones concluded the matter on page 2.",
        ],
        "summary": [
            "Judge Alice Smith summarized the facts.",
            "The issue involved Jane Roe's liability.",
            "Justice Bob Jones delivered the holding.",
        ],
    }
    df_t = pd.DataFrame(sample)
    df_v = pd.DataFrame(sample)
    df_te = pd.DataFrame(sample)
    t, v, te, dropped = load_dataframes(df_t, df_v, df_te)
    print(f"Dropped map: {dropped}")
    analyze_datasets(t, v, te)
