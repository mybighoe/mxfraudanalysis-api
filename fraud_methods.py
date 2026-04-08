import pandas as pd
import numpy as np

def basic_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": list(df.columns),
    }

def detect_duplicates(df: pd.DataFrame) -> dict:
    duplicate_rows = df[df.duplicated()]
    return {
        "duplicate_count": len(duplicate_rows),
    }

def missing_values(df: pd.DataFrame) -> dict:
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())

    return {
        "total_missing_values": total_missing,
        "missing_by_column": missing_counts.to_dict()
    }

def zscore_outliers(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {"outliers_found": 0, "columns": {}}

    zscores = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
    outlier_mask = (zscores.abs() > 3)

    outliers_per_column = outlier_mask.sum().to_dict()
    total_outliers = int(outlier_mask.sum().sum())

    return {
        "outliers_found": total_outliers,
        "columns": outliers_per_column
    }

def benfords_law(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {"benford_deviation": "no numeric columns"}

    first_digits = numeric_df.applymap(lambda x: int(str(x)[0]) if pd.notnull(x) and str(x)[0].isdigit() else None)
    first_digits = first_digits.stack().dropna()

    observed = first_digits.value_counts(normalize=True).sort_index().to_dict()

    benford_expected = {
        d: np.log10(1 + 1/d) for d in range(1, 10)
    }

    return {
        "observed_distribution": observed,
        "expected_distribution": benford_expected
    }
