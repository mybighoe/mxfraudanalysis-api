from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fraud_methods import basic_summary, detect_duplicates

app = FastAPI()

# Allow your Hostinger domain to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyse")
async def analyse_file(file: UploadFile = File(...)):
    # Read uploaded file into a DataFrame
    contents = await file.read()
    df = pd.read_csv(
        filepath_or_buffer=pd.io.common.BytesIO(contents)
    )

    # Run analysis methods
    summary = basic_summary(df)
    duplicates = detect_duplicates(df)

    # Return JSON response
    return {
        "summary": summary,
        "duplicates": duplicates,
        # later: add all 23 methods here
    }