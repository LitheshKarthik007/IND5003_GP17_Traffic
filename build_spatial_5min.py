import pandas as pd
import numpy as np
import os

INPUT = "scripts/result_ml/clustered_traffic.csv"
OUTPUT = "scripts/artifacts/spatial_dataset/causeway_5min_raw.csv"

CAMERAS = [2701, 2702, 2704, 2706]

df = pd.read_csv(INPUT, parse_dates=["timestamp"])

df = df[df["camera_id"].isin(CAMERAS)].copy()

# Round each camera observation to nearest 5-minute bucket
df["timestamp"] = df["timestamp"].dt.round("5min")

# Keep only traffic measurements
df = df[
    [
        "timestamp",
        "camera_id",
        "total_vehicles",
        "vehicles_per_mpx",
        "cluster"
    ]
]

# If multiple observations fall into the same bucket,
# keep the first one.
df = (
    df.sort_values("timestamp")
      .drop_duplicates(["timestamp", "camera_id"])
)

# Pivot each camera into columns
wide = df.pivot(
    index="timestamp",
    columns="camera_id",
    values=[
        "total_vehicles",
        "vehicles_per_mpx",
        "cluster"
    ]
)

wide.columns = [
    f"{feature}_{camera}"
    for feature, camera in wide.columns
]

wide = wide.sort_index()

# Make sure all expected columns exist
for feature in ["total_vehicles", "vehicles_per_mpx", "cluster"]:
    for cam in CAMERAS:
        col = f"{feature}_{cam}"
        if col not in wide.columns:
            wide[col] = np.nan

wide = wide[
    [
        f"{feature}_{cam}"
        for feature in ["total_vehicles", "vehicles_per_mpx", "cluster"]
        for cam in CAMERAS
    ]
]

# Add missing-data indicators
for cam in CAMERAS:
    wide[f"missing_{cam}"] = (
        wide[f"total_vehicles_{cam}"].isna().astype(int)
    )

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

wide.to_csv(OUTPUT)

print("Saved:", OUTPUT)
print("Shape:", wide.shape)

print("\nMissing percentage:")
for cam in CAMERAS:
    print(
        cam,
        f"{wide[f'total_vehicles_{cam}'].isna().mean()*100:.2f}%"
    )

print("\nFirst rows:")
print(wide.head().to_string())