from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

from fraud_methods import (
    basic_summary,
    detect_duplicates,
    missing_values,
    zscore_outliers,
    benfords_law
)

app = FastAPI()

# Allow your Hostinger domain to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict later to: ["https://bgnveranda.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyse")
async def analyse_file(file: UploadFile = File(...)):
    try:
        # Read uploaded file into a DataFrame
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Run analysis methods
        summary = basic_summary(df)
        duplicates = detect_duplicates(df)
        missing = missing_values(df)
        zscore = zscore_outliers(df)
        benford = benfords_law(df)

        # Return JSON response
        return {
            "summary": summary,
            "duplicates": duplicates,
            "missing_values": missing,
            "zscore_outliers": zscore,
            "benfords_law": benford
        }

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}