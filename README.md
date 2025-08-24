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

## Baseline summarizers

`baselines.py` provides quick baselines for extractive and abstractive summarization. The
extractive baseline uses TextRank via `sumy`, while the abstractive baseline relies on a
pretrained transformer such as `google/pegasus-xsum`. Both are evaluated with ROUGE and
BERTScore on a small validation sample:

```python
from baselines import evaluate_baselines
from data_utils import load_dataframes

_, df_val, _, _ = load_dataframes()
metrics = evaluate_baselines(df_val, sample_size=32)
print(metrics)
```

`evaluate_baselines` returns aggregated ROUGE-1/2/L and BERTScore metrics for the two
baselines.

## Long-context summarization

Use `long_context.py` to handle documents that exceed the context window of
standard summarizers. The helper chunk-summarizes with a sliding window and can
optionally leverage locally available long-context models such as
`allenai/led-base-16384` or `mistralai/Mistral-7B-32k`. Chunk summaries are
recombined via either a simple vote/consensus pass or a tree-of-thought
stitching strategy:

```python
from long_context import long_context_summary

text = "... very long legal document ..."
summary = long_context_summary(text, model_name="allenai/led-base-16384", strategy="tree")
print(summary)
```

If the requested model is unavailable, the function falls back to
`google/pegasus-xsum` and performs hierarchical summarization over sliding
window chunks.

## Fine-tuning models

`finetune.py` offers a utility to fine-tune instruction models with either LoRA adapters or full parameter updates. Supported backbones include `mistralai/Mistral-7B-Instruct-v0.3`, `meta-llama/Meta-Llama-3-8B-Instruct`, and `Qwen2.5-7B-Instruct`.

The helper loads a model and tokenizer, masks out prompt tokens with `-100` for supervised fine-tuning, and can optionally pack multiple examples into fixed-length sequences for efficiency. During training the `Trainer` computes ROUGE and BERTScore on a validation set, logs metrics to Weights & Biases (`project="legal-llm"`), and saves the best checkpoint by ROUGE-L to `out/legal-llm-sft`. LoRA uses `r=16`, `alpha=32`, and `dropout=0.05`. When `load_in_4bit=True`, the model is prepared for QLoRA training via `prepare_model_for_kbit_training`.

Example usage:

```python
from datasets import Dataset
from finetune import train

# df_train/df_val contain columns: doc_id, prompt, target
train_ds = Dataset.from_pandas(df_train)
val_ds = Dataset.from_pandas(df_val)
train(
    train_ds,
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    eval_dataset=val_ds,
    use_lora=True,
    load_in_4bit=True,
    gradient_accumulation_steps=4,
)
```

`TrainingArguments` expose common knobs such as `gradient_accumulation_steps`, `lr_scheduler_type`, and `save_strategy='epoch'`.

### Domain-adaptive pretraining (DAPT)

Before supervised fine-tuning, you can run a lightweight domain-adaptive pretraining
step over `text_clean` only. The `run_dapt` helper trains a causal LM with short
sequences (512–2048 tokens) using the same backbone as SFT. The resulting
checkpoint can then be fed into `train` for summarization or other tasks.

```python
from datasets import Dataset
from finetune import run_dapt, train

# text_df contains a doc_id and text_clean column
text_ds = Dataset.from_pandas(text_df[["doc_id", "text_clean"]])
dapt_dir = run_dapt(text_ds, model_name="mistralai/Mistral-7B-Instruct-v0.3", output_dir="mistral_dapt")

# summarization_df has prompt/target columns produced by data_utils
summ_ds = Dataset.from_pandas(summarization_df)
train(summ_ds, model_name=dapt_dir, output_dir="mistral_sft", use_lora=True)
```

For convenience, `dapt_then_sft` chains the two stages sequentially:

```python
from finetune import dapt_then_sft
dapt_then_sft(text_ds, summ_ds, model_name="mistralai/Mistral-7B-Instruct-v0.3", dapt_dir="mistral_dapt", sft_dir="mistral_sft")
```

## Retrieval-augmented generation

`rag.py` provides a small retrieval stack that chunks `text_clean` into 1k-token
segments with 200-token overlap, embeds them with
`sentence-transformers/all-MiniLM-L6-v2`, and indexes the vectors in FAISS.
Given a legal question, the retriever returns the top-k relevant chunks and the
`RAGPipeline` composes a prompt that cites each chunk by `doc_id:chunk_id`
before generating an answer with a causal language model.

```python
import pandas as pd
from rag import chunk_dataframe, FaissRetriever, RAGPipeline

# df_train has doc_id and text_clean
chunks = chunk_dataframe(df_train)
retriever = FaissRetriever()
retriever.build(chunks)

pipeline = RAGPipeline(retriever)
result = pipeline.generate("What is the holding regarding liability?", top_k=3)
print(result["answer"])
print("Citations:", result["citations"])
```

## Faithfulness and factuality evaluation

`faithfulness.py` provides heuristics to check whether summaries stay grounded in
their source documents. It performs question‑answer generation (QAG) faithfulness,
natural language inference with `roberta-large-mnli`, and a hallucination-rate
proxy based on retrieved contexts. Length‑controlled ROUGE and BERTScore metrics
are also reported. The helper returns a pandas DataFrame of per‑document scores:

```python
from faithfulness import evaluate_faithfulness

doc_ids = ["case-1"]
sources = ["The court held the contract void due to fraud."]
summaries = ["The contract was voided for fraud, ruled the court."]
df = evaluate_faithfulness(doc_ids, sources, summaries)
print(df)
```

Use the `--demo` flag for a minimal run:

```bash
python faithfulness.py --demo
```
