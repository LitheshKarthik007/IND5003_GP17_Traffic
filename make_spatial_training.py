import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

INPUT = "scripts/artifacts/spatial_dataset/causeway_5min_raw.csv"

OUTPUT_DIR = "scripts/artifacts/spatial_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT = os.path.join(OUTPUT_DIR, "causeway_spatial_dataset.npz")

# We use the previous 2 hours
# 24 steps × 5 minutes = 120 minutes
T_IN = 24

# Predict next 30 minutes
# 6 steps × 5 minutes = 30 minutes
T_OUT = 6

TARGET = 2704

CAMERAS = [2701, 2702, 2704, 2706]


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading:", INPUT)

df = pd.read_csv(INPUT, parse_dates=["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("Original shape:", df.shape)


# ============================================================
# 2. FEATURES
# ============================================================

feature_columns = []

for cam in CAMERAS:

    feature_columns.append(f"total_vehicles_{cam}")

    feature_columns.append(f"vehicles_per_mpx_{cam}")

print("\nFeatures:")
print(feature_columns)


# ============================================================
# 3. REMOVE ROWS WITH MISSING TARGET
# ============================================================

target_column = f"cluster_{TARGET}"

print("\nTarget:", target_column)

before = len(df)

df = df.dropna(subset=[target_column]).copy()

print(
    "Removed rows with missing target:",
    before - len(df)
)


# ============================================================
# 4. HANDLE MISSING FEATURES
# ============================================================

print("\nMissing values before filling:")

print(
    df[feature_columns]
    .isna()
    .sum()
)


# Forward fill followed by backward fill
df[feature_columns] = (
    df[feature_columns]
    .ffill()
    .bfill()
)


print("\nMissing values after filling:")

print(
    df[feature_columns]
    .isna()
    .sum()
)


# ============================================================
# 5. BUILD FEATURE MATRIX
# ============================================================

X_raw = df[feature_columns].values.astype(np.float32)

# Target congestion state
y_raw = df[target_column].astype(int).values


print("\nX_raw shape:", X_raw.shape)
print("y_raw shape:", y_raw.shape)

print(
    "Target classes:",
    np.unique(y_raw)
)


# ============================================================
# 6. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_raw)


# ============================================================
# 7. CREATE TEMPORAL WINDOWS
# ============================================================

X_data = []
y_data = []

timestamps = df["timestamp"].values


for i in range(T_IN, len(X_scaled) - T_OUT + 1):

    # Previous 24 timesteps
    X_window = X_scaled[i - T_IN:i]

    # Future 6 congestion states
    y_window = y_raw[i:i + T_OUT]

    X_data.append(X_window)
    y_data.append(y_window)


X_data = np.array(X_data, dtype=np.float32)

y_data = np.array(y_data, dtype=np.int64)


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
# ============================================================

split = int(len(X_data) * 0.8)

X_train = X_data[:split]
y_train = y_data[:split]

X_val = X_data[split:]
y_val = y_data[split:]


# ============================================================
# 9. SAVE
# ============================================================

np.savez(
    OUTPUT,

    X_train=X_train,
    y_train=y_train,

    X_val=X_val,
    y_val=y_val
)


# ============================================================
# 10. REPORT
# ============================================================

print("\n======================================")
print("SPATIAL TRAINING DATASET CREATED")
print("======================================")

print("Total samples :", len(X_data))

print("X_train       :", X_train.shape)
print("y_train       :", y_train.shape)

print("X_val         :", X_val.shape)
print("y_val         :", y_val.shape)

print("Output:")
print(OUTPUT)

print("\nFeature count:", X_train.shape[2])

print("\nTarget classes:")
print(np.unique(y_data))