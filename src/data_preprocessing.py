import pandas as pd
import os
import re

PRODUCTS = [
    "Credit card", "Personal loan", "Buy Now, Pay Later (BNPL)",
    "Savings account", "Money transfers"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_complaints(input_path="data/raw/complaints.csv", output_path="data/interim/filtered_complaints.csv"):
    # ✅ Fix dtype warning
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Original dataset: {df.shape[0]} rows")

    # Drop rows with missing narratives
    df = df[df["Consumer complaint narrative"].notna()]

    # Filter for specified products
    df = df[df["Product"].isin(PRODUCTS)]

    # Clean the narrative text
    df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_text)

    print(f"Filtered dataset: {df.shape[0]} rows")

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
