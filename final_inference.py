import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ============================================================
# TRAFLOW FINAL INFERENCE PIPELINE
# ============================================================

TRAFFIC_FILE = (
    "scripts/result_ml/clustered_traffic.csv"
)

MODEL_FILE = (
    "scripts/artifacts/downstream_prediction/"
    "causeway_downstream_lstm.keras"
)

RISK_FILE = (
    "scripts/artifacts/propagation_analysis/"
    "propagation_link_risk.csv"
)

OUTPUT_DIR = (
    "scripts/artifacts/final_inference"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

UPSTREAM_CAMERAS = [
    2701,
    2702,
    2706
]

TARGET_CAMERA = 2704

ALL_CAMERAS = [
    2701,
    2702,
    2706,
    2704
]

T_IN = 6
T_OUT = 6

STATE_NAMES = {
    0: "Very Light",
    1: "Light",
    2: "Moderate",
    3: "Severe"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def state_name(state):

    state = int(
        np.clip(
            round(state),
            0,
            3
        )
    )

    return STATE_NAMES[state]


def risk_level(score):

    if score >= 75:
        return "HIGH"

    elif score >= 50:
        return "MEDIUM"

    elif score >= 25:
        return "LOW"

    else:
        return "MINIMAL"


def calculate_base_risk(
    current_states,
    predicted_states
):

    upstream_values = [
        current_states[cam]
        for cam in UPSTREAM_CAMERAS
    ]

    upstream_score = (
        np.mean(upstream_values) / 3.0
    )

    future_score = (
        np.mean(predicted_states) / 3.0
    )

    risk = (
        0.40 * upstream_score
        +
        0.60 * future_score
    ) * 100

    return float(
        np.clip(
            risk,
            0,
            100
        )
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("TRAFLOW FINAL INFERENCE PIPELINE")
print("=" * 65)


# ============================================================
# LOAD TRAFFIC DATA
# ============================================================

print("\nLoading traffic data...")

if not os.path.exists(
    TRAFFIC_FILE
):

    raise FileNotFoundError(
        f"Traffic file not found:\n"
        f"{TRAFFIC_FILE}"
    )


df = pd.read_csv(
    TRAFFIC_FILE,
    parse_dates=[
        "timestamp"
    ]
)

df = df[
    df["camera_id"].isin(
        ALL_CAMERAS
    )
].copy()

df = df.sort_values(
    [
        "timestamp",
        "camera_id"
    ]
)

print(
    "Rows:",
    len(df)
)

print(
    "Cameras:",
    sorted(
        df.camera_id.unique()
    )
)


# ============================================================
# CORRECT CLUSTER VALUES
# ============================================================

print(
    "\nCluster values before correction:"
)

print(
    sorted(
        df["cluster"]
        .dropna()
        .unique()
        .tolist()
    )
)

# The clustered dataset already uses:
#
# 0 = Very Light
# 1 = Light
# 2 = Moderate
# 3 = Severe

df["cluster"] = (
    pd.to_numeric(
        df["cluster"],
        errors="coerce"
    )
    .round()
    .clip(
        0,
        3
    )
)

df = df.dropna(
    subset=["cluster"]
)

df["cluster"] = (
    df["cluster"]
    .astype(int)
)

print(
    "\nCluster values after correction:"
)

print(
    sorted(
        df["cluster"]
        .unique()
        .tolist()
    )
)


# ============================================================
# CURRENT CAMERA STATES
# ============================================================

print("\nFinding latest camera states...")

current_states = {}

for cam in ALL_CAMERAS:

    cam_data = df[
        df["camera_id"] == cam
    ]

    if len(cam_data) == 0:

        raise ValueError(
            f"No data found for camera {cam}"
        )

    cam_data = cam_data.sort_values(
        "timestamp"
    )

    latest_row = (
        cam_data.iloc[-1]
    )

    current_states[cam] = int(
        latest_row["cluster"]
    )


latest_raw_timestamp = (
    df["timestamp"].max()
)

print(
    "\nLatest raw timestamp:",
    latest_raw_timestamp
)


# ============================================================
# CURRENT CORRIDOR SITUATION
# ============================================================

print("\n")
print("=" * 65)
print("CURRENT CORRIDOR SITUATION")
print("=" * 65)

for cam in ALL_CAMERAS:

    print(
        f"Camera {cam}: "
        f"{state_name(current_states[cam])}"
    )


# ============================================================
# BUILD SYNCHRONIZED 5-MINUTE DATA
# ============================================================

print("\nPreparing model input...")

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

    c = c[
        [
            "total_vehicles",
            "vehicles_per_mpx",
            "cluster"
        ]
    ]

    c.columns = [
        f"vehicles_{cam}",
        f"density_{cam}",
        f"state_{cam}"
    ]

    # IMPORTANT:
    # clustered_traffic.csv can contain
    # duplicate timestamps.
    #
    # Group first so resampling never
    # encounters duplicate index labels.

    c = (
        c.groupby(
            level=0
        )
        .agg({
            f"vehicles_{cam}": "mean",
            f"density_{cam}": "mean",
            f"state_{cam}": "mean"
        })
    )

    # Resample to 5-minute intervals.

    c = (
        c.resample("5min")
        .mean()
    )

    tables.append(c)


# Merge all cameras.

data = pd.concat(
    tables,
    axis=1
)

print(
    "\nSynchronized shape:",
    data.shape
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print(
    "\nMissing values before filling:"
)

print(
    data.isna()
    .sum()
    .to_string()
)

# Interpolate short gaps.

data = data.interpolate(
    method="linear",
    limit=3
)

# Fill remaining edge values only.

data = data.ffill()
data = data.bfill()


# ============================================================
# ROUND STATE COLUMNS
# ============================================================

for cam in ALL_CAMERAS:

    state_col = (
        f"state_{cam}"
    )

    data[state_col] = (
        data[state_col]
        .round()
        .clip(
            0,
            3
        )
        .astype(int)
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
    2
    * np.pi
    * data["hour"]
    / 24
)

data["hour_cos"] = np.cos(
    2
    * np.pi
    * data["hour"]
    / 24
)

data["weekday_sin"] = np.sin(
    2
    * np.pi
    * data["weekday"]
    / 7
)

data["weekday_cos"] = np.cos(
    2
    * np.pi
    * data["weekday"]
    / 7
)


# ============================================================
# TRAFFIC CHANGE FEATURES
# ============================================================

# IMPORTANT:
# The original downstream model was trained with
# vehicle_change and density_change for ALL 4 cameras.

for cam in ALL_CAMERAS:

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


# ============================================================
# EXACT TRAINING FEATURE ORDER
# ============================================================

FEATURES = []

for cam in ALL_CAMERAS:

    FEATURES.extend([
        f"vehicles_{cam}",
        f"density_{cam}",
        f"state_{cam}",
        f"vehicle_change_{cam}",
        f"density_change_{cam}"
    ])


FEATURES.extend([
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos"
])


# ============================================================
# FEATURE CHECK
# ============================================================

print(
    "\nFeature count:",
    len(FEATURES)
)

print(
    "\nFeatures:"
)

for i, feature in enumerate(
    FEATURES,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# EXPECTED FEATURE COUNT
# ============================================================

EXPECTED_FEATURES = 24

if len(FEATURES) != EXPECTED_FEATURES:

    raise ValueError(
        f"Feature construction error.\n"
        f"Generated: {len(FEATURES)}\n"
        f"Expected: {EXPECTED_FEATURES}"
    )


# ============================================================
# REMOVE FIRST ROW WITH DIFF NaN
# ============================================================

data = data.dropna(
    subset=FEATURES
)

print(
    "\nData shape after feature creation:",
    data.shape
)


# ============================================================
# LOAD DOWNSTREAM MODEL
# ============================================================

print(
    "\nLoading downstream model..."
)

if not os.path.exists(
    MODEL_FILE
):

    raise FileNotFoundError(
        f"Model not found:\n"
        f"{MODEL_FILE}"
    )


model = load_model(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)


# ============================================================
# MODEL INPUT CHECK
# ============================================================

model_input_shape = (
    model.input_shape
)

print(
    "Model input shape:",
    model_input_shape
)

model_expected_features = (
    model.input_shape[-1]
)

print(
    "Model expects:",
    model_expected_features,
    "features"
)

if model_expected_features != len(
    FEATURES
):

    raise ValueError(
        "\nMODEL FEATURE MISMATCH\n"
        f"Generated features: "
        f"{len(FEATURES)}\n"
        f"Model expects: "
        f"{model_expected_features}"
    )


# ============================================================
# FIND LATEST CONTINUOUS 30-MINUTE WINDOW
# ============================================================

print(
    "\nSearching for latest continuous "
    "30-minute history..."
)

timestamps = (
    data.index
)

latest_window = None

for end in range(
    len(data),
    T_IN - 1,
    -1
):

    candidate = data.iloc[
        end - T_IN:end
    ]

    candidate_times = (
        candidate.index
    )

    gaps = (
        candidate_times.to_series()
        .diff()
        .dropna()
    )

    if (
        len(gaps) == T_IN - 1
        and
        (
            gaps
            == pd.Timedelta(
                minutes=5
            )
        ).all()
    ):

        latest_window = candidate

        break


if latest_window is None:

    raise ValueError(
        "Could not find a continuous "
        "30-minute input window."
    )


print(
    "\nModel input period:"
)

print(
    "Start:",
    latest_window.index[0]
)

print(
    "End:",
    latest_window.index[-1]
)


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

X_input = (
    latest_window[
        FEATURES
    ]
    .values
    .astype(
        np.float32
    )
)

X_input = np.expand_dims(
    X_input,
    axis=0
)

print(
    "\nInput shape:",
    X_input.shape
)


# ============================================================
# RUN MODEL
# ============================================================

print(
    "\nRunning downstream prediction..."
)

prediction_raw = model.predict(
    X_input,
    verbose=0
)

print(
    "Raw prediction shape:",
    prediction_raw.shape
)


# ============================================================
# DECODE MODEL OUTPUT
# ============================================================

if prediction_raw.ndim == 3:

    # Expected:
    #
    # (1, 6, 4)

    probabilities = (
        prediction_raw[0]
    )

elif prediction_raw.ndim == 2:

    # Possible:
    #
    # (1, 24)

    probabilities = (
        prediction_raw[0]
        .reshape(
            T_OUT,
            4
        )
    )

else:

    raise ValueError(
        "Unexpected model output shape: "
        + str(
            prediction_raw.shape
        )
    )


predicted_states = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# FUTURE PREDICTION
# ============================================================

print("\n")
print("=" * 65)
print("FUTURE DOWNSTREAM PREDICTION")
print("=" * 65)

prediction_records = []

for i in range(
    T_OUT
):

    horizon = (
        (i + 1)
        * 5
    )

    predicted_state = int(
        predicted_states[i]
    )

    confidence = float(
        probabilities[i][
            predicted_state
        ]
    )

    print(
        f"T+{horizon:02d} min : "
        f"{state_name(predicted_state):10s} "
        f"confidence={confidence:.3f}"
    )

    prediction_records.append({

        "horizon_min":
            horizon,

        "predicted_state":
            predicted_state,

        "predicted_state_name":
            state_name(
                predicted_state
            ),

        "confidence":
            round(
                confidence,
                4
            )
    })


prediction_df = pd.DataFrame(
    prediction_records
)


# ============================================================
# PROPAGATION RISK INFORMATION
# ============================================================

print("\n")
print("=" * 65)
print("PROPAGATION INFORMATION")
print("=" * 65)

strongest_upstream = None
strongest_downstream = None
strongest_risk = 0.0
strongest_lag = 0.0

if os.path.exists(
    RISK_FILE
):

    link_risk = pd.read_csv(
        RISK_FILE
    )

    if len(link_risk) > 0:

        link_risk = (
            link_risk
            .sort_values(
                "average_risk",
                ascending=False
            )
        )

        strongest = (
            link_risk.iloc[0]
        )

        strongest_upstream = int(
            strongest[
                "upstream_camera"
            ]
        )

        strongest_downstream = int(
            strongest[
                "downstream_camera"
            ]
        )

        strongest_risk = float(
            strongest[
                "average_risk"
            ]
        )

        strongest_lag = float(
            strongest[
                "average_lag"
            ]
        )

        print(
            "Strongest propagation link:"
        )

        print(
            f"{strongest_upstream} "
            f"-> "
            f"{strongest_downstream}"
        )

        print(
            f"Historical average risk: "
            f"{strongest_risk:.2f}/100"
        )

        print(
            f"Historical average lag: "
            f"{strongest_lag:.2f} minutes"
        )

    else:

        print(
            "Propagation risk file is empty."
        )

else:

    print(
        "Propagation risk file not found."
    )


# ============================================================
# FINAL CORRIDOR RISK
# ============================================================

predicted_states = (
    predicted_states.astype(int)
)

base_risk = calculate_base_risk(
    current_states,
    predicted_states
)

# Combine:
#
# 60% current/future model risk
# 40% historical propagation risk

final_risk = (
    0.60 * base_risk
    +
    0.40 * strongest_risk
)

final_risk = float(
    np.clip(
        final_risk,
        0,
        100
    )
)

final_level = risk_level(
    final_risk
)


# ============================================================
# TREND ANALYSIS
# ============================================================

current_target_state = (
    current_states[
        TARGET_CAMERA
    ]
)

future_target_state = int(
    predicted_states[-1]
)

if future_target_state > current_target_state:

    trend = "WORSENING"

elif future_target_state < current_target_state:

    trend = "IMPROVING"

else:

    trend = "STABLE"


# ============================================================
# CORRIDOR INSIGHT
# ============================================================

if (
    strongest_upstream is not None
    and
    final_level in [
        "HIGH",
        "MEDIUM"
    ]
):

    insight = (
        f"Congestion from upstream "
        f"camera {strongest_upstream} "
        f"is likely to propagate toward "
        f"camera {TARGET_CAMERA}."
    )

elif final_level == "HIGH":

    insight = (
        "High congestion risk detected "
        "across the corridor."
    )

elif final_level == "MEDIUM":

    insight = (
        "Moderate congestion propagation "
        "risk detected."
    )

elif trend == "WORSENING":

    insight = (
        "Traffic conditions are worsening "
        "toward the downstream camera."
    )

else:

    insight = (
        "No significant congestion "
        "propagation risk detected."
    )


# ============================================================
# FINAL TRAFLOW RESULT
# ============================================================

print("\n")
print("=" * 65)
print("FINAL TRAFLOW CORRIDOR INTELLIGENCE")
print("=" * 65)

print(
    "\nCurrent target state :",
    state_name(
        current_target_state
    )
)

print(
    "Predicted +30 min    :",
    state_name(
        future_target_state
    )
)

print(
    "Trend                :",
    trend
)

print(
    "Propagation risk     :",
    f"{final_risk:.2f}/100"
)

print(
    "Risk level           :",
    final_level
)

if strongest_upstream is not None:

    print(
        "Strongest upstream   :",
        strongest_upstream,
        "->",
        strongest_downstream
    )

    print(
        "Typical propagation  :",
        f"{strongest_lag:.2f} min"
    )

print(
    "\nCorridor insight:"
)

print(
    insight
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_output = os.path.join(
    OUTPUT_DIR,
    "future_predictions.csv"
)

prediction_df.to_csv(
    prediction_output,
    index=False
)


# ============================================================
# SAVE FINAL JSON
# ============================================================

final_result = {

    "current_timestamp":
        str(
            latest_window.index[-1]
        ),

    "current_states": {
        str(cam):
            state_name(
                current_states[cam]
            )
        for cam in ALL_CAMERAS
    },

    "target_camera":
        TARGET_CAMERA,

    "predictions":
        prediction_records,

    "predicted_states":
        [
            state_name(x)
            for x in predicted_states
        ],

    "predicted_state_values":
        predicted_states.tolist(),

    "propagation_risk":
        round(
            final_risk,
            2
        ),

    "risk_level":
        final_level,

    "trend":
        trend,

    "strongest_upstream_camera":
        strongest_upstream,

    "strongest_downstream_camera":
        strongest_downstream,

    "average_propagation_lag":
        round(
            strongest_lag,
            2
        ),

    "insight":
        insight
}


json_output = os.path.join(
    OUTPUT_DIR,
    "final_result.json"
)

with open(
    json_output,
    "w"
) as f:

    json.dump(
        final_result,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 65)
print("FINAL INFERENCE PIPELINE COMPLETE")
print("=" * 65)

print(
    "\nSaved:"
)

print(
    prediction_output
)

print(
    json_output
)

print(
    "\nDONE."
)