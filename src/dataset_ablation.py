"""Controlled dataset ablation. Does not replace src.train.

Run from project root:

    python -m src.dataset_ablation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import PERIODS, load_asi_wide
from src.destination_matching import match_destinations, unique_kaggle_places
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    chronological_masks,
)
from src.preprocessing import build_processed_panel
from src.feature_engineering import build_forecast_table
from src.train import metrics_dict

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
KAGGLE_FILE = RAW / "indian_tourism_kaggle_travel_data.csv"
NATIONAL_FILE = RAW / "national_aggregate_2004_2024_25.csv"
STATE_FILE = RAW / "statewise_visits_2019_2020.csv"
QUARTER_FILE = RAW / "national_quarterly_seasonality_2001_2019.csv"
MAPPING_PATH = Path("destination_mapping.csv")
RANDOM_STATE = 42
CROWD_LABELS = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]

# Circle -> official state names used in statewise file (conservative).
CIRCLE_TO_STATE = {
    "Agra": "Uttar Pradesh",
    "Lucknow": "Uttar Pradesh",
    "Jhansi": "Uttar Pradesh",
    "Sarnath": "Uttar Pradesh",
    "Delhi": "Delhi",
    "Chennai": "Tamil Nadu",
    "Thiruchirapalli": "Tamil Nadu",
    "Thrissur": "Kerala",
    "Bhopal": "Madhya Pradesh",
    "Jabalpur": "Madhya Pradesh",
    "Dharwad": "Karnataka",
    "Hampi": "Karnataka",
    "Bangalore": "Karnataka",
    "Raiganj": "West Bengal",
    "Kolkata": "West Bengal",
    "Rajkot": "Gujarat",
    "Vadodara": "Gujarat",
    "Bhubaneswar": "Odisha",
    "Aurangabad": "Maharashtra",
    "Mumbai": "Maharashtra",
    "Nagpur": "Maharashtra",
    "Chandigarh": "Haryana",
    "Guwahati": "Assam",
    "Goa": "Goa",
    "Hyderabad": "Telengana",
    "Jaipur": "Rajasthan",
    "Jodhpur": "Rajasthan",
    "Leh": "Ladakh",
    "Patna": "Bihar",
    "Raipur": "Chhattisgarh",
    "Shimla": "Himachal Pradesh",
    "Srinagar": "J&K",
    "Amaravati": "Andhra Pradesh",
}

KAGGLE_NUM = ["kaggle_google_rating", "kaggle_ticket_price", "kaggle_matched"]
KAGGLE_CAT = ["kaggle_place_type", "kaggle_zone", "kaggle_significance", "kaggle_airport"]
NATIONAL_NUM = [
    "national_lag1_total",
    "national_lag1_growth",
    "national_lag1_foreign_share",
    "state_covid_domestic_drop",
]


def audit_kaggle(path: Path) -> dict:
    df = pd.read_csv(path)
    visitors_ok = False
    visitors_note = "no Visitors_Count column"
    if "Visitors_Count" in df.columns:
        visitors_ok = False
        visitors_note = (
            "Visitors_Count exists but is not compatible with ASI annual monument "
            f"counts (Kaggle min={df['Visitors_Count'].min()}, max={df['Visitors_Count'].max()}, "
            "median looks like daily-scale synthetic/operational counts, not ASI FY totals)."
        )
    return {
        "filename": str(path.name),
        "source_note": (
            "Not originally in workspace; downloaded public copy of "
            "sushanthnaidu24/indian-tourism-dataset as used by "
            "aryankasundra509/tourism-demand-forecasting (travel_data.csv)."
        ),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missingness": {c: int(df[c].isna().sum()) for c in df.columns},
        "unique_Place_Name": int(df["Place_Name"].nunique()) if "Place_Name" in df.columns else None,
        "duplicate_rows": int(df.duplicated().sum()),
        "years": sorted(df["Year"].dropna().unique().tolist()) if "Year" in df.columns else [],
        "has_Visitors_Count": "Visitors_Count" in df.columns,
        "visitors_compatible_with_ASI": visitors_ok,
        "visitors_note": visitors_note,
        "data_character": (
            "mixed: destination metadata repeated across dated rows 2023–2025; "
            "Visitors_Count appears generated/operational-scale, not official ASI annual footfall"
        ),
        "sample_head": df.head(2).to_dict(orient="records"),
    }


def load_national_lookup() -> dict[str, dict]:
    nat = pd.read_csv(NATIONAL_FILE)
    lookup = {}
    for _, row in nat.iterrows():
        period = str(row["period"]).strip()
        lookup[period] = {
            "total": float(row["total_visitors"]),
            "domestic": float(row["domestic_visitors"]),
            "foreign": float(row["foreign_visitors"]),
        }
    return lookup


def load_state_covid_drop() -> dict[str, float]:
    st = pd.read_csv(STATE_FILE)
    out = {}
    for _, row in st.iterrows():
        name = str(row["States/UTs"]).strip()
        if name == "Grand Total":
            continue
        d19 = pd.to_numeric(row["Domestic -2019"], errors="coerce")
        d20 = pd.to_numeric(row["Domestic -2020"], errors="coerce")
        if pd.notna(d19) and d19 > 0 and pd.notna(d20):
            out[name] = float((d20 - d19) / d19)
    return out


def attach_kaggle(table: pd.DataFrame, mapping: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
    meta = mapping.merge(
        places,
        how="left",
        left_on="kaggle_name",
        right_on="kaggle_name",
    )
    out = table.merge(
        meta[
            [
                "official_name",
                "kaggle_name",
                "match_method",
                "place_type",
                "zone",
                "significance",
                "google_rating",
                "ticket_price",
                "airport_within_50km",
            ]
        ],
        how="left",
        left_on="monument",
        right_on="official_name",
    )
    matched = out["kaggle_name"].fillna("").ne("")
    out["kaggle_matched"] = matched.astype(int)
    out["kaggle_place_type"] = out["place_type"].fillna("unmatched")
    out["kaggle_zone"] = out["zone"].fillna("unmatched")
    out["kaggle_significance"] = out["significance"].fillna("unmatched")
    out["kaggle_airport"] = out["airport_within_50km"].fillna("unmatched")
    out["kaggle_google_rating"] = out["google_rating"].fillna(out["google_rating"].median())
    out["kaggle_ticket_price"] = out["ticket_price"].fillna(out["ticket_price"].median())
    return out


def attach_national_state(table: pd.DataFrame) -> pd.DataFrame:
    nat = load_national_lookup()
    covid_drop = load_state_covid_drop()
    out = table.copy()
    lag1_totals = []
    lag1_growth = []
    lag1_fshare = []
    state_drop = []
    for _, rec in out.iterrows():
        t = int(rec["year_index"])
        p_lag1 = PERIODS[t - 1]
        p_lag2 = PERIODS[t - 2] if t >= 2 else None
        n1 = nat.get(p_lag1)
        n2 = nat.get(p_lag2) if p_lag2 else None
        if n1:
            lag1_totals.append(n1["total"])
            lag1_fshare.append(n1["foreign"] / n1["total"] if n1["total"] else 0.0)
        else:
            lag1_totals.append(np.nan)
            lag1_fshare.append(np.nan)
        if n1 and n2 and n2["total"] > 0:
            lag1_growth.append(n1["total"] / n2["total"] - 1.0)
        else:
            lag1_growth.append(0.0)
        state = CIRCLE_TO_STATE.get(rec["asi_circle"], "")
        state_drop.append(covid_drop.get(state, np.nan))
    out["national_lag1_total"] = lag1_totals
    out["national_lag1_growth"] = lag1_growth
    out["national_lag1_foreign_share"] = lag1_fshare
    med = np.nanmedian(state_drop)
    out["state_covid_domestic_drop"] = [med if not np.isfinite(x) else x for x in state_drop]
    med_nat = np.nanmedian(lag1_totals)
    out["national_lag1_total"] = out["national_lag1_total"].fillna(med_nat)
    out["national_lag1_foreign_share"] = out["national_lag1_foreign_share"].fillna(
        np.nanmedian(lag1_fshare)
    )
    return out


def gb_pipeline(num_cols, cat_cols) -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), num_cols),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            cat_cols,
                        ),
                    ]
                ),
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_gb(train, val, test, num_cols, cat_cols):
    cols = num_cols + cat_cols
    model = gb_pipeline(num_cols, cat_cols)
    model.fit(train[cols], np.log1p(train[TARGET_COL]))
    pred_val = np.expm1(np.clip(model.predict(val[cols]), 0, None))
    pred_test = np.expm1(np.clip(model.predict(test[cols]), 0, None))
    return pred_val, pred_test


def historical_hist(panel: pd.DataFrame, monument: str, before_year_index: int) -> np.ndarray:
    sub = panel[
        (panel["monument"] == monument)
        & (panel["year_index"] < before_year_index)
        & panel["total_visitors"].notna()
    ]
    return sub["total_visitors"].to_numpy(dtype=float)


def label_from_hist(value: float, hist: np.ndarray, global_cuts: tuple) -> str:
    if len(hist) >= 4:
        p25, p50, p75 = np.percentile(hist, [25, 50, 75])
    else:
        p25, p50, p75 = global_cuts
    if value <= p25:
        return "LOW"
    if value <= p50:
        return "MODERATE"
    if value <= p75:
        return "HIGH"
    return "VERY HIGH"


def crowd_eval(panel, frame, pred) -> dict:
    global_hist = panel.loc[panel["total_visitors"].notna(), "total_visitors"].to_numpy()
    cuts = tuple(np.percentile(global_hist, [25, 50, 75]))
    y_true = []
    y_pred = []
    year_index = int(frame["year_index"].iloc[0])
    for i, (_, rec) in enumerate(frame.iterrows()):
        hist = historical_hist(panel, rec["monument"], year_index)
        y_true.append(label_from_hist(rec[TARGET_COL], hist, cuts))
        y_pred.append(label_from_hist(pred[i], hist, cuts))
    labels = CROWD_LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    high_true = [t in {"HIGH", "VERY HIGH"} for t in y_true]
    high_pred = [p in {"HIGH", "VERY HIGH"} for p in y_pred]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "high_veryhigh_recall": float(recall_score(high_true, high_pred, zero_division=0)),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "n": len(y_true),
    }


def run_config(name, train, val, test, num_cols, cat_cols, panel):
    pred_val, pred_test = fit_gb(train, val, test, num_cols, cat_cols)
    return {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "val": metrics_dict(val[TARGET_COL], pred_val),
        "test": metrics_dict(test[TARGET_COL], pred_test),
        "crowd_val": crowd_eval(panel, val, pred_val),
        "crowd_test": crowd_eval(panel, test, pred_test),
        "pred_test": pred_test,
    }


def error_rows(test, pred, names):
    frame = test.copy()
    frame["pred"] = pred
    frame["abs_err"] = (frame[TARGET_COL] - frame["pred"]).abs()
    focus = []
    for name in names:
        sub = frame[frame["monument"].str.contains(name, case=False, regex=False)]
        if sub.empty:
            sub = frame[frame["monument"] == name]
        if sub.empty:
            continue
        r = sub.iloc[0]
        focus.append(
            {
                "monument": r["monument"],
                "actual": float(r[TARGET_COL]),
                "predicted": float(r["pred"]),
                "baseline": float(r["baseline_pred"]),
                "abs_err": float(r["abs_err"]),
            }
        )
    worst = (
        frame.nlargest(5, "abs_err")[
            ["monument", TARGET_COL, "pred", "baseline_pred", "abs_err"]
        ]
        .rename(columns={TARGET_COL: "actual", "pred": "predicted", "baseline_pred": "baseline"})
        .to_dict(orient="records")
    )
    return focus, worst


def write_markdown(path: Path, payload: dict) -> None:
    a = payload["results"]
    lines = [
        "# Dataset ablation report",
        "",
        "Controlled experiment: same ASI target, same chronological splits, Gradient Boosting vs persistence.",
        "",
        "## 1. Datasets actually used",
        "",
        f"- ASI visitors: `{payload['asi_file']}`",
        f"- Kaggle: `{payload['kaggle_audit']['filename']}` — {payload['kaggle_audit']['source_note']}",
        "- National ASI aggregates 2004–2024/25 (lagged totals/growth only)",
        "- Statewise visits 2019 vs 2020 (static COVID drop by mapped circle→state)",
        "- National quarterly FTA 2001–2019: **not used as a forecast feature** (within-year shares; no between-row variation for annual targets)",
        "",
        "## 2. Kaggle audit (short)",
        "",
        f"- Rows × cols: {payload['kaggle_audit']['rows']} × {payload['kaggle_audit']['columns']}",
        f"- Unique Place_Name: {payload['kaggle_audit']['unique_Place_Name']}",
        f"- Duplicate rows: {payload['kaggle_audit']['duplicate_rows']}",
        f"- Years: {payload['kaggle_audit']['years']}",
        f"- Visitors_Count used in model? **No.** {payload['kaggle_audit']['visitors_note']}",
        f"- Character: {payload['kaggle_audit']['data_character']}",
        "",
        "## 3. Destination matching",
        "",
        f"- ASI monuments: {payload['matching']['n_asi']}",
        f"- Matched: {payload['matching']['n_matched']}",
        f"- Unmatched: {payload['matching']['n_unmatched']}",
        f"- Ambiguous (left unmatched): {payload['matching']['n_ambiguous']}",
        "",
        "Kaggle `Visitors_Count` was excluded from prediction.",
        "",
        "## 4. Leakage notes",
        "",
        "- Kaggle type/zone/rating/ticket/airport: treated as **static destination attributes** (modes/medians). Not aligned to ASI FY. Safe as metadata, weak as demand drivers.",
        "- National totals: only **lag-1 and lag-2 national ASI ticketed totals** (known before the target FY).",
        "- State COVID drop: 2019→2020 only; known before 2023-24/2024-25. Time-invariant.",
        "- Same-year national 2024-25 was **not** used when predicting 2024-25.",
        "",
        "## 5. Regression results",
        "",
        "| Configuration | Val MAE | Val R² | Test MAE | Test R² |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    order = [
        ("persistence", "Persistence baseline"),
        ("asi_only", "ASI only"),
        ("asi_kaggle", "ASI + Kaggle"),
        ("asi_national", "ASI + National/State"),
        ("asi_kaggle_national", "ASI + Kaggle + National/State"),
    ]
    for key, label in order:
        r = a[key]
        lines.append(
            f"| {label} | {r['val']['MAE']:,.0f} | {r['val']['R2']:.3f} | "
            f"{r['test']['MAE']:,.0f} | {r['test']['R2']:.3f} |"
        )
    lines += [
        "",
        "## 6. Crowd-level (test 2024-25)",
        "",
        "Labels use monument-specific quartiles of **prior** years only (global quartiles if <4 observations).",
        "This is relative demand, not occupancy.",
        "",
        "| Configuration | Accuracy | Macro F1 | HIGH/VERY HIGH recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in order:
        c = a[key]["crowd_test"]
        lines.append(
            f"| {label} | {c['accuracy']:.3f} | {c['macro_f1']:.3f} | {c['high_veryhigh_recall']:.3f} |"
        )
    lines += [
        "",
        "## 7. Decision",
        "",
        payload["decision"],
        "",
        "## 8. Error analysis (test, best ML config = ASI only GB)",
        "",
        "Focus destinations:",
        "",
    ]
    for row in payload["error_focus"]:
        lines.append(
            f"- {row['monument']}: actual={row['actual']:,.0f}, "
            f"GB={row['predicted']:,.0f}, persistence={row['baseline']:,.0f}, "
            f"|err|={row['abs_err']:,.0f}"
        )
    lines += ["", "Largest GB errors:", ""]
    for row in payload["error_worst"]:
        lines.append(
            f"- {row['monument']}: actual={row['actual']:,.0f}, "
            f"GB={row['predicted']:,.0f}, persistence={row['baseline']:,.0f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict:
    REPORTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(exist_ok=True)

    kaggle = pd.read_csv(KAGGLE_FILE)
    kaggle_audit = audit_kaggle(KAGGLE_FILE)

    wide = load_asi_wide()
    asi_names = wide[["monument", "asi_circle"]].drop_duplicates()
    places = unique_kaggle_places(kaggle)
    mapping = match_destinations(asi_names, places)
    mapping.to_csv(MAPPING_PATH, index=False)
    n_matched = int((mapping["kaggle_name"] != "").sum())
    matching_stats = {
        "n_asi": int(len(mapping)),
        "n_matched": n_matched,
        "n_unmatched": int((mapping["kaggle_name"] == "").sum()),
        "n_ambiguous": len(mapping.attrs.get("ambiguous", [])),
        "ambiguous": mapping.attrs.get("ambiguous", []),
        "match_methods": mapping["match_method"].value_counts().to_dict(),
    }

    panel = build_processed_panel()
    table = build_forecast_table(panel)
    table = attach_kaggle(table, mapping, places)
    table = attach_national_state(table)
    table.to_csv(PROCESSED / "forecast_table_enriched.csv", index=False)

    masks = chronological_masks(table)
    train = table[masks["train"]].copy()
    val = table[masks["val"]].copy()
    test = table[masks["test"]].copy()

    persistence = {
        "val": metrics_dict(val[TARGET_COL], val["baseline_pred"]),
        "test": metrics_dict(test[TARGET_COL], test["baseline_pred"]),
        "crowd_val": crowd_eval(panel, val, val["baseline_pred"].to_numpy()),
        "crowd_test": crowd_eval(panel, test, test["baseline_pred"].to_numpy()),
        "numeric_features": ["lag_1_total"],
        "categorical_features": [],
    }

    results = {"persistence": persistence}
    results["asi_only"] = run_config(
        "asi_only", train, val, test, NUMERIC_FEATURES, CATEGORICAL_FEATURES, panel
    )
    results["asi_kaggle"] = run_config(
        "asi_kaggle",
        train,
        val,
        test,
        NUMERIC_FEATURES + KAGGLE_NUM,
        CATEGORICAL_FEATURES + KAGGLE_CAT,
        panel,
    )
    results["asi_national"] = run_config(
        "asi_national",
        train,
        val,
        test,
        NUMERIC_FEATURES + NATIONAL_NUM,
        CATEGORICAL_FEATURES,
        panel,
    )
    results["asi_kaggle_national"] = run_config(
        "asi_kaggle_national",
        train,
        val,
        test,
        NUMERIC_FEATURES + KAGGLE_NUM + NATIONAL_NUM,
        CATEGORICAL_FEATURES + KAGGLE_CAT,
        panel,
    )

    # Serialize without huge pred arrays in json
    json_results = {}
    pred_store = {}
    for key, valr in results.items():
        pred_store[key] = valr.get("pred_test")
        json_results[key] = {k: v for k, v in valr.items() if k != "pred_test"}

    ml_keys = ["asi_only", "asi_kaggle", "asi_national", "asi_kaggle_national"]
    best_ml = min(ml_keys, key=lambda k: json_results[k]["val"]["MAE"])
    persist_val = json_results["persistence"]["val"]["MAE"]
    persist_test = json_results["persistence"]["test"]["MAE"]
    any_beat_val = any(json_results[k]["val"]["MAE"] < persist_val for k in ml_keys)
    any_beat_test = any(json_results[k]["test"]["MAE"] < persist_test for k in ml_keys)

    crowd_persist = json_results["persistence"]["crowd_test"]["macro_f1"]
    crowd_gains = {
        k: json_results[k]["crowd_test"]["macro_f1"] - crowd_persist for k in ml_keys
    }

    if not any_beat_val and not any_beat_test:
        decision = (
            "CASE C (+ B): Persistence remains the best numerical forecaster on both "
            "validation and test. Adding Kaggle metadata and lagged national/state context "
            "did not produce a leakage-safe ML model that beats last-year demand. "
            "Keep ASI historical demand (persistence) as the prediction layer. "
            "Reserve Kaggle metadata for later recommendation. "
            "Reserve national quarterly seasonality for a labeled time-period heuristic, "
            "not annual regression."
        )
    elif any_beat_test:
        decision = (
            f"CASE A: {best_ml} improved holdout vs persistence. Keep those extra features."
        )
    else:
        decision = (
            "CASE B: Extra datasets did not improve generalization vs persistence. "
            "Do not force them into the prediction model."
        )

    focus, worst = error_rows(
        test,
        pred_store["asi_only"],
        [
            "Taj Mahal",
            "Hampi",
            "Sanchi",
            "Mamallapuram",
            "Qutub Minar",
        ],
    )

    payload = {
        "asi_file": "monument_visitors_2019_20_to_2024_25_FULL.csv",
        "kaggle_audit": kaggle_audit,
        "matching": matching_stats,
        "splits": {
            "train": "target periods 2020-21 through 2022-23",
            "val": "2023-24",
            "test": "2024-25",
        },
        "kaggle_columns_selected": KAGGLE_NUM + KAGGLE_CAT,
        "national_columns_selected": NATIONAL_NUM,
        "excluded": [
            "Visitors_Count",
            "Revenue",
            "Tourist_Type row-level",
            "same-year national totals",
            "quarterly FTA shares as annual features",
        ],
        "best_ml_by_val_mae": best_ml,
        "ml_beats_persistence_val": any_beat_val,
        "ml_beats_persistence_test": any_beat_test,
        "crowd_f1_delta_vs_persistence_test": crowd_gains,
        "results": json_results,
        "decision": decision,
        "error_focus": focus,
        "error_worst": worst,
    }

    # JSON-safe: drop sample_head if needed size; keep it
    (REPORTS / "dataset_ablation_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    write_markdown(REPORTS / "dataset_ablation_report.md", payload)
    print(decision)
    print(
        f"Matched {n_matched}/{len(mapping)} monuments. "
        f"Persistence val MAE={persist_val:,.0f} test MAE={persist_test:,.0f}"
    )
    for key, label in [
        ("asi_only", "ASI only"),
        ("asi_kaggle", "ASI+Kaggle"),
        ("asi_national", "ASI+Nat"),
        ("asi_kaggle_national", "ASI+both"),
    ]:
        r = json_results[key]
        print(
            f"{label:22} val MAE={r['val']['MAE']:,.0f}  test MAE={r['test']['MAE']:,.0f}  "
            f"crowd F1={r['crowd_test']['macro_f1']:.3f}"
        )
    return payload


if __name__ == "__main__":
    run()
