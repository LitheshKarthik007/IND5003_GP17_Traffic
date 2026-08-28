import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = "scripts/result_ml/clustered_traffic.csv"

OUTPUT_DIR = "scripts/artifacts/downstream_prediction"

OUTPUT = os.path.join(
    OUTPUT_DIR,
    "causeway_downstream_dataset.npz"
)

# Causeway cameras
CAMERAS = [2701, 2702, 2704, 2706]

# ------------------------------------------------------------
# Input history
# 6 × 5 minutes = previous 30 minutes
# ------------------------------------------------------------
T_IN = 6

# ------------------------------------------------------------
# Prediction horizon
# 6 × 5 minutes = next 30 minutes
#
# T+5
# T+10
# T+15
# T+20
# T+25
# T+30
# ------------------------------------------------------------
T_OUT = 6


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("BUILDING TRAFLOW DOWNSTREAM DATASET")
print("=" * 60)

print("\nLoading:", INPUT)

df = pd.read_csv(
    INPUT,
    parse_dates=["timestamp"]
)

# Keep only Causeway cameras
df = df[
    df["camera_id"].isin(CAMERAS)
].copy()

df = df.sort_values(
    ["timestamp", "camera_id"]
)

print("Rows:", len(df))

print(
    "Cameras:",
    sorted(df["camera_id"].unique())
)


# ============================================================
# CREATE SYNCHRONIZED 5-MINUTE CAMERA TABLE
# ============================================================

print("\nCreating synchronized 5-minute data...")

tables = []

for cam in CAMERAS:

    print(
        f"Processing camera {cam}..."
    )

    c = df[
        df["camera_id"] == cam
    ].copy()

    c = c.set_index("timestamp")

    # Important traffic measurements
    c = c[
        [
            "total_vehicles",
            "vehicles_per_mpx",
            "cluster"
        ]
    ]

    # Rename columns
    c.columns = [
        f"vehicles_{cam}",
        f"density_{cam}",
        f"state_{cam}"
    ]

    # Convert to 5-minute grid
    c = c.resample("5min").mean()

    tables.append(c)


# ============================================================
# MERGE ALL CAMERAS
# ============================================================

data = pd.concat(
    tables,
    axis=1
)

print(
    "Synchronized shape:",
    data.shape
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print(
    "\nMissing values before filling:"
)

print(
    data.isna().sum().to_string()
)

# Interpolate only short gaps.
#
# limit=3 means at most 3 consecutive
# missing 5-minute points are filled.
data = data.interpolate(
    method="linear",
    limit=3
)

# Remove rows where missing values still remain.
data = data.dropna()

print(
    "\nShape after cleaning:",
    data.shape
)


# ============================================================
# CREATE TEMPORAL FEATURES
# ============================================================

data["hour"] = data.index.hour

data["weekday"] = data.index.dayofweek


# Cyclic hour encoding
data["hour_sin"] = np.sin(
    2 * np.pi * data["hour"] / 24
)

data["hour_cos"] = np.cos(
    2 * np.pi * data["hour"] / 24
)


# Cyclic weekday encoding
data["weekday_sin"] = np.sin(
    2 * np.pi * data["weekday"] / 7
)

data["weekday_cos"] = np.cos(
    2 * np.pi * data["weekday"] / 7
)


# ============================================================
# CREATE TRAFFIC CHANGE FEATURES
# ============================================================

for cam in CAMERAS:

    data[
        f"vehicle_change_{cam}"
    ] = (
        data[f"vehicles_{cam}"].diff()
    )

    data[
        f"density_change_{cam}"
    ] = (
        data[f"density_{cam}"].diff()
    )


# Remove first row created by diff()
data = data.dropna()


# ============================================================
# FEATURE SELECTION
# ============================================================

FEATURES = []

for cam in CAMERAS:

    FEATURES.extend(
        [
            f"vehicles_{cam}",
            f"density_{cam}",
            f"state_{cam}",
            f"vehicle_change_{cam}",
            f"density_change_{cam}"
        ]
    )


# Temporal features
FEATURES.extend(
    [
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos"
    ]
)


print(
    "\nNumber of features:",
    len(FEATURES)
)

print("\nFeatures:")

for feature in FEATURES:

    print(
        " ",
        feature
    )


# ============================================================
# TARGET
# ============================================================

# We predict the future congestion state
# of downstream camera 2704.

TARGET = "state_2704"

print(
    "\nTarget:",
    TARGET
)


# ============================================================
# CHECK ORIGINAL TARGET VALUES
# ============================================================

print(
    "\nTarget values before conversion:"
)

print(
    sorted(
        data[TARGET]
        .dropna()
        .unique()
    )
)


# ============================================================
# BUILD CONTINUOUS SLIDING WINDOWS
#
# IMPORTANT:
#
# We only create a window when every timestamp is exactly
# 5 minutes apart.
#
# This prevents the model from learning across large
# missing-data gaps.
# ============================================================

print(
    "\nBuilding continuous sequences..."
)

X = []

Y = []

TIMES = []


feature_values = data[
    FEATURES
].values

target_values = data[
    TARGET
].values

timestamps = data.index


# ------------------------------------------------------------
# Find continuous sequences
# ------------------------------------------------------------

timestamp_diff = (
    timestamps.to_series().diff()
)

sequence_id = (
    timestamp_diff != pd.Timedelta(
        minutes=5
    )
).cumsum()


sequence_lengths = (
    pd.Series(sequence_id)
    .value_counts()
    .sort_index()
)


print(
    "Number of sequences:",
    len(sequence_lengths)
)

print(
    "Longest sequence:",
    sequence_lengths.max()
)

print(
    "Sequences with >= 12 points:",
    (
        sequence_lengths
        >= (T_IN + T_OUT)
    ).sum()
)


# ============================================================
# PROCESS EACH CONTINUOUS SEQUENCE
# ============================================================

for seq_id in sorted(
    sequence_lengths.index
):

    indices = np.where(
        sequence_id.values
        == seq_id
    )[0]

    # Need:
    #
    # 6 historical points
    # +
    # 6 future points
    #
    # = 12 continuous points

    if len(indices) < (
        T_IN + T_OUT
    ):
        continue


    seq_features = (
        feature_values[indices]
    )

    seq_targets = (
        target_values[indices]
    )

    seq_times = (
        timestamps[indices]
    )


    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    for i in range(
        T_IN,
        len(indices) - T_OUT + 1
    ):

        # Previous 30 minutes
        x_window = seq_features[
            i - T_IN:i
        ]

        # Next 30 minutes
        y_window = seq_targets[
            i:i + T_OUT
        ]

        X.append(
            x_window
        )

        Y.append(
            y_window
        )

        # Timestamp corresponding to T+5
        TIMES.append(
            seq_times[i]
        )


# Convert lists to NumPy arrays

X = np.array(X)

Y = np.array(Y)

TIMES = np.array(TIMES)


# ============================================================
# GENERATED DATASET CHECK
# ============================================================

print(
    "\nGenerated raw dataset:"
)

print(
    "X:",
    X.shape
)

print(
    "Y:",
    Y.shape
)


# ============================================================
# TARGET CLASS HANDLING
# ============================================================

# IMPORTANT:
#
# The `cluster` column in clustered_traffic.csv is ALREADY
# 0-based.
#
# Therefore:
#
# 0 = Very Light
# 1 = Light
# 2 = Moderate
# 3 = Severe
#
# DO NOT subtract 1.
#
# The previous version did:
#
#     Y - 1
#
# which incorrectly created class -1.
#
# We simply convert the values to integer.

Y = Y.astype(
    np.int64
)


# ============================================================
# VALIDATE TARGET CLASSES
# ============================================================

unique_classes = np.unique(Y)

print(
    "\nTarget classes found:"
)

print(
    unique_classes
)


# Make sure no invalid classes exist
valid_classes = {
    0,
    1,
    2,
    3
}

invalid_classes = [
    int(x)
    for x in unique_classes
    if int(x) not in valid_classes
]

if invalid_classes:

    raise ValueError(
        "Invalid target classes found: "
        + str(invalid_classes)
    )


# ============================================================
# TEMPORAL TRAIN / VALIDATION SPLIT
# ============================================================

# IMPORTANT:
#
# We do NOT randomly shuffle.
#
# Older observations -> training
# Newer observations -> validation
#
# This better represents real-world forecasting.

split = int(
    len(X) * 0.8
)


X_train = X[
    :split
]

Y_train = Y[
    :split
]

X_val = X[
    split:
]

Y_val = Y[
    split:
]


times_train = TIMES[
    :split
]

times_val = TIMES[
    split:
]


# ============================================================
# CHECK GENERATED WINDOW TIMESTAMPS
# ============================================================

print(
    "\nChecking generated windows..."
)


def check_windows(
    times,
    name
):

    if len(times) == 0:

        print(
            name,
            ": no samples"
        )

        return


    gaps = (
        pd.Series(
            pd.to_datetime(times)
        )
        .diff()
        .dt.total_seconds()
        .div(60)
    )


    print(
        "\n",
        name,
        "sample-start gap statistics:"
    )

    print(
        gaps.describe()
    )


check_windows(
    times_train,
    "Training"
)

check_windows(
    times_val,
    "Validation"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print(
    "\nTraining target distribution:"
)

unique_train, counts_train = np.unique(
    Y_train,
    return_counts=True
)

for cls, count in zip(
    unique_train,
    counts_train
):

    print(
        f"Class {cls}: {count}"
    )


print(
    "\nValidation target distribution:"
)

unique_val, counts_val = np.unique(
    Y_val,
    return_counts=True
)

for cls, count in zip(
    unique_val,
    counts_val
):

    print(
        f"Class {cls}: {count}"
    )


# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

print("\n")

print(
    "=" * 60
)

print(
    "CONTINUOUS DOWNSTREAM DATASET CREATED"
)

print(
    "=" * 60
)

print(
    "X_train:",
    X_train.shape
)

print(
    "Y_train:",
    Y_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "Y_val:",
    Y_val.shape
)


print(
    "\nInput history:"
)

print(
    "30 minutes"
)

print(
    "6 × 5-minute steps"
)


print(
    "\nPrediction horizons:"
)

for i in range(
    T_OUT
):

    print(
        f"T+{(i + 1) * 5} minutes"
    )


print(
    "\nTarget:"
)

print(
    "Camera 2704 congestion state"
)


print(
    "\nTarget classes:"
)

print(
    "0 = Very Light"
)

print(
    "1 = Light"
)

print(
    "2 = Moderate"
)

print(
    "3 = Severe"
)


# ============================================================
# SAVE DATASET
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


np.savez(
    OUTPUT,

    X_train=X_train,
    Y_train=Y_train,

    X_val=X_val,
    Y_val=Y_val,

    times_train=times_train,
    times_val=times_val
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\nSaved to:"
)

print(
    OUTPUT
)

print(
    "\nDONE."
)