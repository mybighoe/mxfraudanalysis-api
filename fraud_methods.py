import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. BASIC SUMMARY
# ---------------------------------------------------------
def basic_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": list(df.columns),
    }


# ---------------------------------------------------------
# 2. DUPLICATE ROW DETECTION
# ---------------------------------------------------------
def detect_duplicates(df: pd.DataFrame) -> dict:
    duplicate_rows = df[df.duplicated()]
    return {
        "duplicate_count": int(len(duplicate_rows)),
    }


# ---------------------------------------------------------
# 3. MISSING VALUE ANALYSIS
# ---------------------------------------------------------
def missing_values(df: pd.DataFrame) -> dict:
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())

    return {
        "total_missing_values": total_missing,
        "missing_by_column": missing_counts.to_dict()
    }


# ---------------------------------------------------------
# 4. Z-SCORE OUTLIER DETECTION
# ---------------------------------------------------------
def zscore_outliers(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {"outliers_found": 0, "columns": {}}

    # Calculate Z-scores
    zscores = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
    outlier_mask = (zscores.abs() > 3)

    outliers_per_column = outlier_mask.sum().to_dict()
    total_outliers = int(outlier_mask.sum().sum())

    return {
        "outliers_found": total_outliers,
        "columns": outliers_per_column
    }


# ---------------------------------------------------------
# 5. BENFORD'S LAW FIRST-DIGIT ANALYSIS
# ---------------------------------------------------------
def benfords_law(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {"benford_deviation": "no numeric columns"}

    first_digits = []

    # Extract first digits column-by-column
    for col in numeric_df.columns:
        series = numeric_df[col].dropna().astype(str)
        digits = series.map(lambda x: int(x[0]) if x[0].isdigit() else None)
        first_digits.extend([d for d in digits if d is not None])

    if not first_digits:
        return {"benford_deviation": "no valid numeric values"}

    # Observed distribution
    observed = (
        pd.Series(first_digits)
        .value_counts(normalize=True)
        .sort_index()
        .to_dict()
    )

    # Expected Benford distribution
    benford_expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}

    return {
        "observed_distribution": observed,
        "expected_distribution": benford_expected
    }