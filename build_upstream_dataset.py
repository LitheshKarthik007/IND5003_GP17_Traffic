import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = "scripts/result_ml/clustered_traffic.csv"

OUTPUT_DIR = "scripts/artifacts/upstream_prediction"

OUTPUT = os.path.join(
    OUTPUT_DIR,
    "causeway_upstream_dataset.npz"
)

# ------------------------------------------------------------
# UPSTREAM CAMERAS
# ------------------------------------------------------------

INPUT_CAMERAS = [
    2701,
    2702,
    2706
]

# ------------------------------------------------------------
# DOWNSTREAM TARGET CAMERA
# ------------------------------------------------------------

TARGET_CAMERA = 2704

# ------------------------------------------------------------
# INPUT HISTORY
#
# 6 x 5 minutes = 30 minutes
# ------------------------------------------------------------

T_IN = 6

# ------------------------------------------------------------
# OUTPUT HORIZON
#
# 6 x 5 minutes = next 30 minutes
# ------------------------------------------------------------

T_OUT = 6


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("BUILDING TRAFLOW UPSTREAM-ONLY DATASET")
print("=" * 60)

print("\nLoading:", INPUT)

df = pd.read_csv(
    INPUT,
    parse_dates=["timestamp"]
)

# All cameras needed for this dataset
ALL_CAMERAS = INPUT_CAMERAS + [TARGET_CAMERA]

df = df[
    df["camera_id"].isin(ALL_CAMERAS)
].copy()

df = df.sort_values(
    ["timestamp", "camera_id"]
)

print(
    "Rows:",
    len(df)
)

print(
    "Input cameras:",
    INPUT_CAMERAS
)

print(
    "Target camera:",
    TARGET_CAMERA
)


# ============================================================
# CREATE SYNCHRONIZED 5-MINUTE DATA
# ============================================================

print(
    "\nCreating synchronized 5-minute data..."
)

tables = []


for cam in ALL_CAMERAS:

    print(
        f"Processing camera {cam}..."
    )

    c = df[
        df["camera_id"] == cam
    ].copy()

    c = c.set_index(
        "timestamp"
    )

    # ========================================================
    # CONTINUOUS FEATURES
    # ========================================================
    #
    # These are numerical measurements:
    #
    # total_vehicles
    # vehicles_per_mpx
    #
    # Mean is appropriate here.
    # ========================================================

    traffic = (
        c[
            [
                "total_vehicles",
                "vehicles_per_mpx"
            ]
        ]
        .resample("5min")
        .mean()
    )

    traffic.columns = [
        f"vehicles_{cam}",
        f"density_{cam}"
    ]


    # ========================================================
    # CATEGORICAL CONGESTION STATE
    # ========================================================
    #
    # IMPORTANT:
    #
    # DO NOT USE:
    #
    #     resample("5min").mean()
    #
    # on cluster.
    #
    # Your CSV already uses:
    #
    # 0 = Very Light
    # 1 = Light
    # 2 = Moderate
    # 3 = Severe
    #
    # We use the LAST state observed in the 5-minute bucket.
    # ========================================================

    state = (
        c["cluster"]
        .resample("5min")
        .last()
        .rename(
            f"state_{cam}"
        )
    )


    # Combine traffic + state
    table = pd.concat(
        [
            traffic,
            state
        ],
        axis=1
    )

    tables.append(
        table
    )


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
# MISSING VALUES
# ============================================================

print(
    "\nMissing values before filling:"
)

print(
    data.isna().sum().to_string()
)


# ============================================================
# INTERPOLATE CONTINUOUS FEATURES ONLY
# ============================================================

for cam in ALL_CAMERAS:

    data[
        f"vehicles_{cam}"
    ] = (
        data[
            f"vehicles_{cam}"
        ]
        .interpolate(
            method="linear",
            limit=3
        )
    )

    data[
        f"density_{cam}"
    ] = (
        data[
            f"density_{cam}"
        ]
        .interpolate(
            method="linear",
            limit=3
        )
    )


# ============================================================
# REMOVE MISSING STATE VALUES
# ============================================================

state_columns = [
    f"state_{cam}"
    for cam in ALL_CAMERAS
]

data = data.dropna(
    subset=state_columns
)


# ============================================================
# REMOVE REMAINING MISSING TRAFFIC VALUES
# ============================================================

traffic_columns = []

for cam in ALL_CAMERAS:

    traffic_columns.extend(
        [
            f"vehicles_{cam}",
            f"density_{cam}"
        ]
    )


data = data.dropna(
    subset=traffic_columns
)


print(
    "\nShape after cleaning:",
    data.shape
)


# ============================================================
# CONVERT STATE TO INTEGER
# ============================================================

for cam in ALL_CAMERAS:

    data[
        f"state_{cam}"
    ] = (
        data[
            f"state_{cam}"
        ]
        .round()
        .astype(int)
    )


# ============================================================
# CHECK STATE VALUES
# ============================================================

print(
    "\nState values after resampling:"
)

for cam in ALL_CAMERAS:

    values = sorted(
        data[
            f"state_{cam}"
        ]
        .unique()
        .tolist()
    )

    print(
        f"Camera {cam}: {values}"
    )


# ============================================================
# TEMPORAL FEATURES
# ============================================================

data["hour"] = (
    data.index.hour
)

data["weekday"] = (
    data.index.dayofweek
)


data["hour_sin"] = np.sin(
    2 * np.pi *
    data["hour"] /
    24
)

data["hour_cos"] = np.cos(
    2 * np.pi *
    data["hour"] /
    24
)

data["weekday_sin"] = np.sin(
    2 * np.pi *
    data["weekday"] /
    7
)

data["weekday_cos"] = np.cos(
    2 * np.pi *
    data["weekday"] /
    7
)


# ============================================================
# TRAFFIC CHANGE FEATURES
# ============================================================
#
# Only upstream cameras are used as input.
#
# Target camera 2704 is NOT used as an input feature.
# ============================================================

for cam in INPUT_CAMERAS:

    data[
        f"vehicle_change_{cam}"
    ] = (
        data[
            f"vehicles_{cam}"
        ].diff()
    )

    data[
        f"density_change_{cam}"
    ] = (
        data[
            f"density_{cam}"
        ].diff()
    )


# Remove first row caused by diff()
data = data.dropna()


# ============================================================
# SELECT INPUT FEATURES
# ============================================================

FEATURES = []


for cam in INPUT_CAMERAS:

    FEATURES.extend(
        [
            f"vehicles_{cam}",
            f"density_{cam}",
            f"state_{cam}",
            f"vehicle_change_{cam}",
            f"density_change_{cam}"
        ]
    )


# Add temporal features
FEATURES.extend(
    [
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos"
    ]
)


print(
    "\nNumber of input features:",
    len(FEATURES)
)


print(
    "\nInput features:"
)

for feature in FEATURES:

    print(
        " ",
        feature
    )


# ============================================================
# TARGET
# ============================================================

TARGET = (
    f"state_{TARGET_CAMERA}"
)


print(
    "\nTarget:",
    TARGET
)

print(
    "\nIMPORTANT:"
)

print(
    "Camera 2704 is NOT used as an input feature."
)

print(
    "Only its FUTURE congestion state is predicted."
)


# ============================================================
# VALIDATE TARGET
# ============================================================
#
# YOUR CSV ALREADY USES:
#
# 0 = Very Light
# 1 = Light
# 2 = Moderate
# 3 = Severe
#
# Therefore DO NOT subtract 1 later.
# ============================================================

print(
    "\nTarget values after correction:"
)

target_unique = sorted(
    data[
        TARGET
    ]
    .unique()
    .tolist()
)

print(
    target_unique
)


VALID_CLASSES = [
    0,
    1,
    2,
    3
]


if not np.all(
    np.isin(
        data[TARGET].values,
        VALID_CLASSES
    )
):

    print(
        "\nERROR: Invalid target class found."
    )

    print(
        "Expected:",
        VALID_CLASSES
    )

    print(
        "Found:",
        target_unique
    )

    raise ValueError(
        "Target must contain only classes 0,1,2,3."
    )


# ============================================================
# BUILD CONTINUOUS 5-MINUTE SEQUENCES
# ============================================================

print(
    "\nBuilding continuous 5-minute sequences..."
)


feature_values = data[
    FEATURES
].values

target_values = data[
    TARGET
].values

timestamps = data.index


# ------------------------------------------------------------
# Find timestamp gaps
# ------------------------------------------------------------

timestamp_diff = (
    timestamps
    .to_series()
    .diff()
)


# ------------------------------------------------------------
# New sequence whenever gap != 5 minutes
# ------------------------------------------------------------

sequence_id = (
    timestamp_diff !=
    pd.Timedelta(
        minutes=5
    )
).cumsum()


sequence_lengths = (
    pd.Series(
        sequence_id
    )
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
        sequence_lengths >=
        (
            T_IN +
            T_OUT
        )
    ).sum()
)


# ============================================================
# CREATE SLIDING WINDOWS
# ============================================================

X = []

Y = []

TIMES = []


for seq_id in sorted(
    sequence_lengths.index
):

    indices = np.where(
        sequence_id.values ==
        seq_id
    )[0]


    # Need:
    #
    # 6 historical points
    # +
    # 6 future points
    #
    # = 12 points

    if len(indices) < (
        T_IN +
        T_OUT
    ):

        continue


    seq_features = (
        feature_values[
            indices
        ]
    )

    seq_targets = (
        target_values[
            indices
        ]
    )

    seq_times = (
        timestamps[
            indices
        ]
    )


    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    for i in range(
        T_IN,
        len(indices) -
        T_OUT +
        1
    ):

        # Past 30 minutes
        x_window = (
            seq_features[
                i - T_IN:i
            ]
        )


        # Future 30 minutes
        y_window = (
            seq_targets[
                i:i + T_OUT
            ]
        )


        X.append(
            x_window
        )

        Y.append(
            y_window
        )

        TIMES.append(
            seq_times[i]
        )


# Convert to NumPy
X = np.array(
    X,
    dtype=np.float32
)

Y = np.array(
    Y,
    dtype=np.int64
)

TIMES = np.array(
    TIMES
)


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
# TARGET IS ALREADY 0-BASED
# ============================================================
#
# DO NOT DO:
#
#     Y = Y - 1
#
# because your CSV already contains 0,1,2,3.
# ============================================================

Y = Y.astype(
    np.int64
)


print(
    "\nTarget classes found:"
)

print(
    np.unique(Y)
)


# ============================================================
# SAFETY CHECK
# ============================================================

if not np.all(
    np.isin(
        Y,
        [0, 1, 2, 3]
    )
):

    raise ValueError(
        "Generated Y contains invalid classes."
    )


# ============================================================
# TEMPORAL TRAIN / VALIDATION SPLIT
# ============================================================
#
# IMPORTANT:
# No random shuffle.
#
# Earlier time = training
# Later time = validation
#
# This avoids temporal leakage.
# ============================================================

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
# TARGET DISTRIBUTION
# ============================================================

print(
    "\nTraining target distribution:"
)

train_counts = np.bincount(
    Y_train.flatten(),
    minlength=4
)


for cls in range(4):

    print(
        f"Class {cls}: "
        f"{train_counts[cls]}"
    )


print(
    "\nValidation target distribution:"
)

val_counts = np.bincount(
    Y_val.flatten(),
    minlength=4
)


for cls in range(4):

    print(
        f"Class {cls}: "
        f"{val_counts[cls]}"
    )


# ============================================================
# CHECK TRAIN / VALIDATION TIME
# ============================================================

print(
    "\nTraining period:"
)

if len(times_train) > 0:

    print(
        "Start:",
        times_train[0]
    )

    print(
        "End:",
        times_train[-1]
    )


print(
    "\nValidation period:"
)

if len(times_val) > 0:

    print(
        "Start:",
        times_val[0]
    )

    print(
        "End:",
        times_val[-1]
    )


# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

print(
    "\n"
)

print(
    "=" * 60
)

print(
    "TRAFLOW UPSTREAM-ONLY DATASET CREATED"
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
    "\nInput cameras:"
)

print(
    INPUT_CAMERAS
)


print(
    "\nTarget camera:"
)

print(
    TARGET_CAMERA
)


print(
    "\nTarget:"
)

print(
    "Future congestion state of camera 2704"
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
# VERIFY SAVED FILE
# ============================================================

print(
    "\nSaved to:"
)

print(
    OUTPUT
)


# Re-open saved file to verify
saved = np.load(
    OUTPUT,
    allow_pickle=True
)


print(
    "\nSaved dataset verification:"
)

print(
    "X_train:",
    saved["X_train"].shape
)

print(
    "Y_train:",
    saved["Y_train"].shape
)

print(
    "X_val:",
    saved["X_val"].shape
)

print(
    "Y_val:",
    saved["Y_val"].shape
)


print(
    "\nDONE."
)