import os
import re
import unicodedata
import hashlib
import json
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import tiktoken
from prompts import build_prompt

CONFIG = {
    "train_path": "train.csv",
    "val_path": "val.csv",
    "test_path": "test.csv",
    "train_parquet": "train.parquet",
    "val_parquet": "val.parquet",
    "test_parquet": "test.parquet",
}


CASE_REGEX = re.compile(r"\b(?:Case\s+No\.?\s*)?\d{2,}[-/]\d{2,}\b")


def redact_text(
    text: str, allow_personal: bool = False, nlp=None
) -> Tuple[str, Dict[str, str]]:
    """Redact names, locations and case numbers from ``text``.

    Returns the redacted text and a mapping of placeholders to originals. If
    ``allow_personal`` is True, the text is returned unchanged with an empty
    mapping.
    """

    if allow_personal or not isinstance(text, str) or not text.strip():
        return text, {}
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    offsets = []
    mapping: Dict[str, str] = {}
    person_idx = 1
    loc_idx = 1
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            placeholder = f"PERSON_{person_idx}"
            person_idx += 1
        elif ent.label_ in ("GPE", "LOC"):
            placeholder = f"LOCATION_{loc_idx}"
            loc_idx += 1
        else:
            continue
        offsets.append((ent.start_char, ent.end_char, placeholder))
        mapping[placeholder] = ent.text

    case_idx = 1
    for match in CASE_REGEX.finditer(text):
        placeholder = f"CASE_{case_idx}"
        case_idx += 1
        offsets.append((match.start(), match.end(), placeholder))
        mapping[placeholder] = match.group(0)

    offsets.sort(key=lambda x: x[0], reverse=True)
    redacted = text
    for start, end, placeholder in offsets:
        redacted = redacted[:start] + placeholder + redacted[end:]

    return redacted, mapping


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


def _assert_disjoint_doc_ids(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> None:
    """Raise if any ``doc_id`` appears in more than one split."""

    train_ids, val_ids, test_ids = (
        set(df_train["doc_id"]),
        set(df_val["doc_id"]),
        set(df_test["doc_id"]),
    )
    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise AssertionError(
            f"doc_id overlap across splits: {sorted(list(overlap))[:10]}"
        )
    print("No doc_id overlap detected across splits")


def _save_splits_to_parquet(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    cfg: Dict[str, str],
) -> None:
    """Persist ``df_train``, ``df_val`` and ``df_test`` to Parquet files."""

    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = cfg.get(f"{split}_parquet", f"{split}.parquet")
        df.to_parquet(path, index=False)
        print(f"Saved {split} split to {path}")


def _clean_dataframe(
    df: pd.DataFrame, allow_personal: bool, map_path: str
) -> pd.DataFrame:
    """Add cleaned/redacted text and summary columns to a copy of ``df``."""

    df = df.copy()
    df["text_clean"], _ = zip(*df["text"].map(lambda x: clean_text(x, anonymize=False)))
    df["summary_clean"], _ = zip(
        *df["summary"].map(lambda x: clean_text(x, anonymize=False))
    )

    if allow_personal:
        return df

    nlp = spacy.load("en_core_web_sm")
    with open(map_path, "a") as f:
        for i, row in df.iterrows():
            text, t_map = redact_text(row["text_clean"], nlp=nlp)
            summary, s_map = redact_text(row["summary_clean"], nlp=nlp)
            df.at[i, "text_clean"] = text
            df.at[i, "summary_clean"] = summary
            f.write(
                json.dumps(
                    {"doc_id": row["doc_id"], "text": t_map, "summary": s_map}
                )
                + "\n"
            )

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
    allow_personal: bool = False,
    redaction_path: str = "redactions.jsonl",
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

    _assert_disjoint_doc_ids(df_train, df_val, df_test)

    if not allow_personal:
        open(redaction_path, "w").close()

    df_train = _clean_dataframe(df_train, allow_personal, redaction_path)
    df_val = _clean_dataframe(df_val, allow_personal, redaction_path)
    df_test = _clean_dataframe(df_test, allow_personal, redaction_path)

    df_train, dropped_map = drop_near_duplicates(
        df_train, df_val, df_test, threshold=dup_threshold
    )

    for name, df in [("df_train", df_train), ("df_val", df_val), ("df_test", df_test)]:
        print(
            f"{name} cleaned samples:\n"
            f"{df[['text', 'text_clean', 'summary', 'summary_clean']].head(3)}\n"
        )

    _save_splits_to_parquet(df_train, df_val, df_test, cfg)

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


def _extract_section(text: str, header_regex: str) -> Optional[str]:
    """Return section text following a header pattern or ``None``.

    Parameters
    ----------
    text : str
        Document text to search.
    header_regex : str
        Regular expression matching the section header (without the colon).
    """

    pattern = rf"({header_regex}):?(.*?)(?=\n[A-Z][A-Z ]{{2,}}:|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(2).strip() if match else None


def build_summarization_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare abstractive summarization prompts and targets."""

    out = df[["doc_id", "text_clean", "summary_clean"]].copy()
    out["prompt"] = out["text_clean"].map(
        lambda x: build_prompt(x, style="summarization")
    )
    out["target"] = out["summary_clean"]
    return out[["doc_id", "prompt", "target"]]


def build_legal_qa_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic question/answer pairs from ISSUE and HOLDING sections."""

    rows = []
    for _, row in df.iterrows():
        issue = _extract_section(row["text_clean"], "ISSUE")
        holding = _extract_section(row["text_clean"], "HOLDING|HELD")
        if issue and holding:
            prompt = build_prompt(
                row["text_clean"], style="qa", question=issue
            )
            rows.append({"doc_id": row["doc_id"], "prompt": prompt, "target": holding})
    return pd.DataFrame(rows)


def build_headnote_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Construct headnote generation pairs with structured targets.

    Returns a DataFrame containing ``doc_id``, ``prompt``, ``target`` and
    metadata columns ``source='human'`` and ``weight=1.0`` so the result can be
    mixed with synthetic examples that carry a smaller training weight.
    """

    rows = []
    for _, row in df.iterrows():
        facts = _extract_section(row["text_clean"], "FACTS")
        issue = _extract_section(row["text_clean"], "ISSUE")
        holding = _extract_section(row["text_clean"], "HOLDING|HELD")
        reasoning = _extract_section(row["text_clean"], "REASONING")
        parts = []
        if facts:
            parts.append(f"Facts: {facts}")
        if issue:
            parts.append(f"Issue: {issue}")
        if holding:
            parts.append(f"Holding: {holding}")
        if reasoning:
            parts.append(f"Reasoning: {reasoning}")
        if parts:
            prompt = build_prompt(row["text_clean"], style="headnote")
            rows.append(
                {
                    "doc_id": row["doc_id"],
                    "prompt": prompt,
                    "target": "\n".join(parts),
                    "source": "human",
                    "weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _has_structured_sections(text: str, min_sections: int = 3) -> bool:
    """Return ``True`` if ``text`` contains the expected headnote sections."""

    headers = ["Facts:", "Issue:", "Holding:", "Reasoning:"]
    found = sum(1 for h in headers if h.lower() in text.lower())
    return found >= min_sections


def generate_synthetic_headnotes(
    df: pd.DataFrame,
    model_name: str,
    few_shot: Optional[list[Tuple[str, str]]] = None,
    min_chars: int = 4000,
    weight: float = 0.1,
    max_new_tokens: int = 256,
) -> pd.DataFrame:
    """Use a model to label long judgments with synthetic headnotes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``doc_id`` and ``text_clean`` columns.
    model_name : str
        Hugging Face model identifier or path.
    few_shot : list of tuple(str, str), optional
        Optional list of (text, headnote) pairs appended as few-shot examples.
    min_chars : int, default 4000
        Minimum length of ``text_clean`` required to trigger generation.
    weight : float, default 0.1
        Training weight assigned to synthetic examples.
    max_new_tokens : int, default 256
        Maximum number of new tokens to generate.

    Returns
    -------
    pd.DataFrame
        Rows with ``doc_id``, ``prompt``, ``target``, ``source='synthetic'`` and
        ``weight`` columns.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    long_df = df[df["text_clean"].str.len() >= min_chars]

    rows = []
    for _, row in long_df.iterrows():
        base_prompt = build_prompt(row["text_clean"], style="headnote")
        prompt = base_prompt
        if few_shot:
            shots = []
            for text, headnote in few_shot:
                shot_prompt = build_prompt(text, style="headnote") + headnote
                shots.append(shot_prompt)
            prompt = "\n\n".join(shots) + "\n\n" + base_prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        headnote = generated[len(prompt) :].strip()
        if _has_structured_sections(headnote):
            rows.append(
                {
                    "doc_id": row["doc_id"],
                    "prompt": base_prompt,
                    "target": headnote,
                    "source": "synthetic",
                    "weight": weight,
                }
            )

    return pd.DataFrame(rows)


def augment_headnote_dataset(
    df: pd.DataFrame,
    model_name: str,
    **kwargs,
) -> pd.DataFrame:
    """Combine human headnotes with synthetic ones for training."""

    human_df = build_headnote_dataset(df)
    seen = set(human_df["doc_id"])
    remaining = df[~df["doc_id"].isin(seen)]
    synth_df = generate_synthetic_headnotes(remaining, model_name, **kwargs)
    return pd.concat([human_df, synth_df], ignore_index=True)


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

    print("\nSummarization dataset sample:")
    print(build_summarization_dataset(t).head())

    print("\nLegal QA dataset sample:")
    print(build_legal_qa_dataset(t).head())

    print("\nHeadnote dataset sample:")
    print(build_headnote_dataset(t).head())
