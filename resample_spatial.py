import pandas as pd
import numpy as np
import os

INPUT = "scripts/artifacts/spatial_dataset/causeway_spatial.csv"
OUTPUT = "scripts/artifacts/spatial_dataset/causeway_5min.csv"

df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.set_index("timestamp").sort_index()

# Resample to fixed 5-minute intervals
df = df.resample("5min").asfreq()

# Interpolate ONLY short gaps
# Maximum 2 consecutive missing 5-min points = 10 minutes
numeric_cols = df.columns

df[numeric_cols] = (
    df[numeric_cols]
    .interpolate(method="linear", limit=2, limit_direction="both")
)

# Remove rows where anything is still missing
df = df.dropna()

# Save
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
df.to_csv(OUTPUT)

print("Saved:", OUTPUT)
print("Shape:", df.shape)
print()
print("First rows:")
print(df.head())
print()
print("Last rows:")
print(df.tail())