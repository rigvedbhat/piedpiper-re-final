"""Wide-to-long conversion and documented quality treatment.

Official printed values are never overwritten. Invalid/non-numeric cells
become missing. Missing is not zero. Source-anomaly and likely extraction
errors are flagged and excluded from training targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import PERIODS, load_asi_wide

# Only this cell is documented as a source-PDF typo (Cooch Bihar 2021-22).
SOURCE_ANOMALY_PERIOD = "2021-22"

LIKELY_FOREIGN_COPY_THRESHOLD = 0.05
LIKELY_FOREIGN_MIN = 1_000


def parse_visitor_cell(value) -> float:
    """Parse a visitor count. Non-numeric tokens become NaN, not 0."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else np.nan
    text = str(value).strip()
    if text == "" or text == "-" or text.lower() in {"nan", "none", "s", "na"}:
        return np.nan
    text = text.replace(",", "").replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in wide.iterrows():
        for period in PERIODS:
            # File columns use 2019_20, not 2019-20.
            d_col = f"domestic_{period.replace('-', '_')}"
            f_col = f"foreign_{period.replace('-', '_')}"
            domestic = parse_visitor_cell(rec[d_col])
            foreign = parse_visitor_cell(rec[f_col])

            flag_source = bool(rec.get("flag_source_anomaly", False))
            is_source_anomaly = flag_source and period == SOURCE_ANOMALY_PERIOD

            likely_foreign_copy = False
            if (
                np.isfinite(domestic)
                and np.isfinite(foreign)
                and domestic >= LIKELY_FOREIGN_MIN
                and abs(foreign - domestic) / max(domestic, 1.0) <= LIKELY_FOREIGN_COPY_THRESHOLD
                and period == "2020-21"
            ):
                likely_foreign_copy = True

            if np.isfinite(domestic) and np.isfinite(foreign) and not likely_foreign_copy:
                total = domestic + foreign
            elif np.isfinite(domestic) and likely_foreign_copy:
                # Foreign looks duplicated from domestic; do not double-count.
                total = domestic
                foreign = np.nan
            elif np.isfinite(domestic) and not np.isfinite(foreign):
                total = np.nan  # incomplete pair: missing is not zero
            elif np.isfinite(foreign) and not np.isfinite(domestic):
                total = np.nan
            else:
                total = np.nan

            exclude_target = is_source_anomaly or likely_foreign_copy or not np.isfinite(total)

            rows.append(
                {
                    "s_no": rec["s_no"],
                    "asi_circle": rec["asi_circle"],
                    "monument": rec["monument"],
                    "period": period,
                    "year_index": PERIODS.index(period),
                    "domestic_visitors": domestic,
                    "foreign_visitors": foreign,
                    "total_visitors": total,
                    "flag_source_anomaly": is_source_anomaly,
                    "flag_likely_extraction_error": likely_foreign_copy,
                    "flag_incomplete_pair": (
                        (np.isfinite(domestic) != np.isfinite(foreign))
                        if not likely_foreign_copy
                        else False
                    ),
                    "exclude_from_training": exclude_target,
                    "covid_year": int(period == "2020-21"),
                    "recovery_year": int(period == "2021-22"),
                    "post_covid": int(period >= "2022-23"),
                }
            )
    return pd.DataFrame(rows)


def build_processed_panel(path=None) -> pd.DataFrame:
    wide = load_asi_wide(path)
    long = wide_to_long(wide)
    return long.sort_values(["monument", "year_index"]).reset_index(drop=True)
