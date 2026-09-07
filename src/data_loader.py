"""Load the official ASI monument visitor file."""

from pathlib import Path

import pandas as pd

PERIODS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

RAW_VISITORS = Path("data/raw/monument_visitors_2019_20_to_2024_25_FULL.csv")


def default_raw_path() -> Path:
    return RAW_VISITORS


def load_asi_wide(path: Path | None = None) -> pd.DataFrame:
    path = path or default_raw_path()
    df = pd.read_csv(path)
    expected = {
        "s_no",
        "asi_circle",
        "monument",
        "flag_2022_23_mismatch",
        "flag_source_anomaly",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"ASI file missing columns: {missing}")
    return df
