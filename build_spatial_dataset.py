import pandas as pd
import numpy as np
import os

INPUT = "scripts/result_ml/clustered_traffic.csv"
OUTPUT = "scripts/artifacts/spatial_dataset/causeway_spatial.csv"

CAMERAS = [2701, 2702, 2704, 2706]

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# Load data
df = pd.read_csv(INPUT, parse_dates=["timestamp"])

# Keep Causeway cameras
df = df[df["camera_id"].isin(CAMERAS)].copy()

# Keep useful traffic information
df = df[
    ["timestamp", "camera_id", "total_vehicles",
     "vehicles_per_mpx", "cluster"]
]

# Pivot cameras into columns
wide = df.pivot_table(
    index="timestamp",
    columns="camera_id",
    values=[
        "total_vehicles",
        "vehicles_per_mpx",
        "cluster"
    ],
    aggfunc="first"
)

# Flatten column names
wide.columns = [
    f"{feature}_{camera}"
    for feature, camera in wide.columns
]

# Remove timestamps where any camera is missing
wide = wide.dropna()

# Sort chronologically
wide = wide.sort_index()

# Save
wide.to_csv(OUTPUT)

print("Saved:", OUTPUT)
print("Shape:", wide.shape)
print()
print(wide.head())
print()
print("First timestamp:", wide.index.min())
print("Last timestamp:", wide.index.max())