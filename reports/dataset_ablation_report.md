# Dataset ablation report

Controlled experiment: same ASI target, same chronological splits, Gradient Boosting vs persistence.

## 1. Datasets actually used

- ASI visitors: `monument_visitors_2019_20_to_2024_25_FULL.csv`
- Kaggle: `indian_tourism_kaggle_travel_data.csv` — Not originally in workspace; downloaded public copy of sushanthnaidu24/indian-tourism-dataset as used by aryankasundra509/tourism-demand-forecasting (travel_data.csv).
- National ASI aggregates 2004–2024/25 (lagged totals/growth only)
- Statewise visits 2019 vs 2020 (static COVID drop by mapped circle→state)
- National quarterly FTA 2001–2019: **not used as a forecast feature** (within-year shares; no between-row variation for annual targets)

## 2. Kaggle audit (short)

- Rows × cols: 20000 × 25
- Unique Place_Name: 321
- Duplicate rows: 0
- Years: [2023, 2024, 2025]
- Visitors_Count used in model? **No.** Visitors_Count exists but is not compatible with ASI annual monument counts (Kaggle min=8, max=4237, median looks like daily-scale synthetic/operational counts, not ASI FY totals).
- Character: mixed: destination metadata repeated across dated rows 2023–2025; Visitors_Count appears generated/operational-scale, not official ASI annual footfall

## 3. Destination matching

- ASI monuments: 145
- Matched: 15
- Unmatched: 130
- Ambiguous (left unmatched): 0

Kaggle `Visitors_Count` was excluded from prediction.

## 4. Leakage notes

- Kaggle type/zone/rating/ticket/airport: treated as **static destination attributes** (modes/medians). Not aligned to ASI FY. Safe as metadata, weak as demand drivers.
- National totals: only **lag-1 and lag-2 national ASI ticketed totals** (known before the target FY).
- State COVID drop: 2019→2020 only; known before 2023-24/2024-25. Time-invariant.
- Same-year national 2024-25 was **not** used when predicting 2024-25.

## 5. Regression results

| Configuration | Val MAE | Val R² | Test MAE | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Persistence baseline | 77,256 | 0.927 | 39,719 | 0.987 |
| ASI only | 273,508 | 0.205 | 243,210 | 0.359 |
| ASI + Kaggle | 269,481 | 0.278 | 234,685 | 0.469 |
| ASI + National/State | 301,035 | 0.129 | 299,076 | 0.178 |
| ASI + Kaggle + National/State | 293,100 | 0.162 | 299,577 | 0.199 |

## 6. Crowd-level (test 2024-25)

Labels use monument-specific quartiles of **prior** years only (global quartiles if <4 observations).
This is relative demand, not occupancy.

| Configuration | Accuracy | Macro F1 | HIGH/VERY HIGH recall |
| --- | ---: | ---: | ---: |
| Persistence baseline | 0.559 | 0.484 | 0.887 |
| ASI only | 0.168 | 0.150 | 0.304 |
| ASI + Kaggle | 0.161 | 0.141 | 0.287 |
| ASI + National/State | 0.105 | 0.091 | 0.087 |
| ASI + Kaggle + National/State | 0.112 | 0.093 | 0.113 |

## 7. Decision

CASE C (+ B): Persistence remains the best numerical forecaster on both validation and test. Adding Kaggle metadata and lagged national/state context did not produce a leakage-safe ML model that beats last-year demand. Keep ASI historical demand (persistence) as the prediction layer. Reserve Kaggle metadata for later recommendation. Reserve national quarterly seasonality for a labeled time-period heuristic, not annual regression.

## 8. Error analysis (test, best ML config = ASI only GB)

Focus destinations:

- Taj Mahal: actual=6,909,849, GB=1,760,206, persistence=6,780,215, |err|=5,149,643
- Group of Monuments, Hampi: actual=1,008,721, GB=333,783, persistence=1,012,130, |err|=674,938
- Buddhist Monuments, Sanchi: actual=315,532, GB=140,263, persistence=311,760, |err|=175,269
- Group of Monuments, Mamallapuram: actual=1,008,464, GB=842,498, persistence=1,256,965, |err|=165,966
- Qutub Minar: actual=3,424,804, GB=883,541, persistence=3,343,660, |err|=2,541,263

Largest GB errors:

- Taj Mahal: actual=6,909,849, GB=1,760,206, persistence=6,780,215
- Sun Temple, Konarak: actual=3,576,348, GB=819,933, persistence=3,201,973
- Qutub Minar: actual=3,424,804, GB=883,541, persistence=3,343,660
- Red Fort: actual=2,963,710, GB=1,255,426, persistence=2,878,260
- Tomb of Rabia Dura ni (Bibi Ka Maqbara): actual=2,015,147, GB=366,016, persistence=1,304,682
