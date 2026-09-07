# Data quality treatment (prediction experiment)

Official printed visitor counts are never overwritten in the raw file.

## Missing values

- Tokens `-`, empty, `s`, `NA` parse to missing, **not zero**.
- If domestic or foreign is missing, `total_visitors` is missing (incomplete pair). That year cannot be a training target.

## Documented source anomaly

- Cooch Bihar Palace, 2021-22 domestic is flagged `flag_source_anomaly`.
- That monument-year is excluded from training targets. The printed value remains in the long table.

## Likely extraction errors (Patna circle, 2020-21)

- If foreign ≈ domestic (within 5%) and domestic ≥ 1,000 in 2020-21, foreign is treated as missing and total uses domestic only so we do not double-count.
- The original CSV is unchanged.

## COVID

- 2020-21 is retained as a real shock (Model A).
- Sensitivity: Gradient Boosting retrained without 2020-21 **target** rows.

## Duplicates

- One row per monument-year after the wide-to-long transform.
