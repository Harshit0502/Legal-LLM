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
