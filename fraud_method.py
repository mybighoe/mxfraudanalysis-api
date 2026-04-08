import pandas as pd

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

# Later you’ll add your 23 fraud analysis methods here.