# Legal-LLM

## Colab Environment Setup

Run the following script in Google Colab or a local notebook to install dependencies,
print hardware information, and initialize random seeds.

```python
!python setup_colab.py
```

The script installs required libraries, downloads the `en_core_web_sm` spaCy model if
necessary, displays CUDA/CPU details, defines a `set_seed` helper, and sets the default
seed to `42` for reproducibility.

## Loading training data

Use `data_utils.load_dataframes` to ensure your datasets meet the expected schema. It
accepts preloaded DataFrames or reads CSV/Parquet files from a config and prints shapes,
null counts, and example rows while validating that `text` and `summary` are non-empty
strings. The loader also applies a `clean_text` routine to produce `text_clean` and
`summary_clean` columns that normalize Unicode, standardize punctuation, drop page/line
numbers, and optionally anonymize names. Near-duplicate `text_clean` entries in `df_train`
are removed using a SimHash similarity threshold of `0.9` to prevent leakage against
`df_val` and `df_test`. A mapping of dropped indices to `doc_id` is returned.

The loader verifies that `doc_id` values are unique across splits and writes the
cleaned DataFrames to canonical Parquet files (`train.parquet`, `val.parquet`,
`test.parquet`).

```python
from data_utils import load_dataframes

# Option 1: pass existing DataFrames
df_train, df_val, df_test, dropped = load_dataframes(df_train, df_val, df_test)

# Option 2: read from paths defined in CONFIG
df_train, df_val, df_test, dropped = load_dataframes()
print("Dropped duplicates:", dropped)
print(df_train[["text", "text_clean"]].head())
```

## Dataset statistics

After loading and cleaning the splits, call `analyze_datasets` to inspect length
distributions and vocabulary overlap. The helper saves histogram plots for character
and token lengths using tiktoken's `cl100k_base` encoder and reports percentiles
(`p50`, `p90`, `p95`, `p99`). It also lemmatizes each split with spaCy to compute the
Jaccard overlap of vocabularies between train/val/test.

```python
from data_utils import analyze_datasets

df_train, df_val, df_test, dropped = load_dataframes()
analyze_datasets(df_train, df_val, df_test)
```

## Task-specific dataset transforms

Three helpers in `data_utils` generate `prompt`/`target` pairs for downstream
modeling tasks:

- **Abstractive summarization** – `build_summarization_dataset` uses
  `text_clean` as the input and `summary_clean` as the target.
- **Legal QA** – `build_legal_qa_dataset` extracts `ISSUE` and `HOLDING/HELD`
  sections from `text_clean` to form synthetic question/answer pairs.
- **Headnote generation** – `build_headnote_dataset` creates structured targets
  with `Facts`, `Issue`, `Holding`, and `Reasoning` sections.

Each function returns a DataFrame with `doc_id`, `prompt`, and `target` columns:

```python
from data_utils import (
    build_summarization_dataset,
    build_legal_qa_dataset,
    build_headnote_dataset,
)

df_train, _, _, _ = load_dataframes()
summ_df = build_summarization_dataset(df_train)
qa_df = build_legal_qa_dataset(df_train)
headnote_df = build_headnote_dataset(df_train)
```

## Prompt templates

`prompts.py` exposes reusable templates with explicit `SYSTEM` and `USER` roles for a legal tone.
Use `build_prompt(text, style)` to format case text into a prompt. Available styles are `summarization`, `headnote`, and `qa`:

```python
from prompts import build_prompt

print(build_prompt("Some case text", style="headnote"))
```

The dataset helpers above automatically apply the appropriate templates when generating `prompt`/`target` pairs.
