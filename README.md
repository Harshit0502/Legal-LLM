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

## Data Preprocessing

## Loading training data

Use `data_utils.load_dataframes` to ensure your datasets meet the expected schema. It
accepts preloaded DataFrames or reads CSV/Parquet files from a config and prints shapes,
null counts, and example rows while validating that `text` and `summary` are non-empty
strings. The loader also applies a `clean_text` routine to produce `text_clean` and
`summary_clean` columns that normalize Unicode, standardize punctuation, drop page/line
numbers, and optionally anonymize names.


```python
from data_utils import load_dataframes

# Option 1: pass existing DataFrames
df_train, df_val, df_test = load_dataframes(df_train, df_val, df_test)

# Option 2: read from paths defined in CONFIG
df_train, df_val, df_test = load_dataframes()
print(df_train[["text", "text_clean"]].head())

```
