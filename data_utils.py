import pandas as pd
from typing import Optional, Tuple, Dict

CONFIG = {
    "train_path": "train.csv",
    "val_path": "val.csv",
    "test_path": "test.csv",
}

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


def load_dataframes(
    df_train: Optional[pd.DataFrame] = None,
    df_val: Optional[pd.DataFrame] = None,
    df_test: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    return df_train, df_val, df_test


if __name__ == "__main__":
    sample = {
        "doc_id": [1, 2],
        "text": ["Example text", "Another text"],
        "summary": ["Summary one", "Summary two"],
    }
    df_t = pd.DataFrame(sample)
    df_v = pd.DataFrame(sample)
    df_te = pd.DataFrame(sample)
    load_dataframes(df_t, df_v, df_te)
