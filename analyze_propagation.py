import os
import numpy as np
import pandas as pd

# ============================================================
# TRAFLOW CONGESTION PROPAGATION ANALYSIS
# ============================================================

INPUT = "scripts/result_ml/clustered_traffic.csv"

OUTPUT_DIR = "scripts/artifacts/propagation_analysis"

CAMERAS = [2701, 2702, 2704, 2706]

TARGET_CAMERA = 2704

UPSTREAM_CAMERAS = [2701, 2702, 2706]

RESAMPLE_INTERVAL = "5min"

LAGS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("TRAFLOW CONGESTION PROPAGATION ANALYSIS")
print("=" * 65)

print("\nLoading:", INPUT)

df = pd.read_csv(
    INPUT,
    parse_dates=["timestamp"]
)

df = df[
    df["camera_id"].isin(CAMERAS)
].copy()

df = df.sort_values(
    ["timestamp", "camera_id"]
)

print("\nRows:", len(df))

print(
    "Cameras:",
    sorted(df["camera_id"].unique())
)


# ============================================================
# CHECK CLASSES
# ============================================================

print("\nCluster values before correction:")

print(
    sorted(
        df["cluster"]
        .dropna()
        .unique()
        .tolist()
    )
)


# Current dataset already uses:
#
# 0 = Very Light
# 1 = Light
# 2 = Moderate
# 3 = Severe
#

df["cluster"] = pd.to_numeric(
    df["cluster"],
    errors="coerce"
)

df = df.dropna(
    subset=["cluster"]
).copy()

df["cluster"] = (
    df["cluster"]
    .round()
    .astype(int)
)

df = df[
    df["cluster"].isin([0, 1, 2, 3])
].copy()


print("\nCluster values after correction:")

print(
    sorted(
        df["cluster"].unique().tolist()
    )
)


# ============================================================
# SYNCHRONIZE CAMERAS
# ============================================================

print("\nCreating synchronized data...")

tables = []


for cam in CAMERAS:

    print(
        f"Processing camera {cam}..."
    )

    c = df[
        df["camera_id"] == cam
    ].copy()

    if c.empty:
        raise RuntimeError(
            f"No data found for camera {cam}"
        )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # There may be duplicate timestamps.
    #
    # Aggregate them BEFORE setting the index.
    # --------------------------------------------------------

    c = c[
        [
            "timestamp",
            "cluster",
            "total_vehicles",
            "vehicles_per_mpx"
        ]
    ].copy()


    # Traffic measurements:
    # mean when multiple records exist
    #
    # State:
    # median/rounded state when duplicate records exist.
    #
    # Since cluster is categorical, median is safer than mean.
    # --------------------------------------------------------

    c = (
        c.groupby("timestamp")
        .agg(
            {
                "cluster": "median",
                "total_vehicles": "mean",
                "vehicles_per_mpx": "mean"
            }
        )
        .sort_index()
    )


    # Rename columns

    c.columns = [
        f"state_{cam}",
        f"vehicles_{cam}",
        f"density_{cam}"
    ]


    # --------------------------------------------------------
    # RESAMPLE TRAFFIC VALUES
    # --------------------------------------------------------

    traffic = c[
        [
            f"vehicles_{cam}",
            f"density_{cam}"
        ]
    ].resample(
        RESAMPLE_INTERVAL
    ).mean()


    # --------------------------------------------------------
    # RESAMPLE STATE
    #
    # Use nearest interpolation AFTER creating the grid.
    # No .asfreq(), therefore no duplicate-label problem.
    # --------------------------------------------------------

    state = c[
        [f"state_{cam}"]
    ].resample(
        RESAMPLE_INTERVAL
    ).mean()


    state = state.interpolate(
        method="nearest",
        limit_direction="both"
    )


    state[f"state_{cam}"] = (
        state[f"state_{cam}"]
        .round()
        .clip(0, 3)
        .astype(int)
    )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    camera_data = pd.concat(
        [
            state,
            traffic
        ],
        axis=1
    )

    tables.append(
        camera_data
    )


# ============================================================
# MERGE CAMERAS
# ============================================================

data = pd.concat(
    tables,
    axis=1
)

data = data.sort_index()


print(
    "\nSynchronized shape:",
    data.shape
)


# ============================================================
# MISSING VALUES
# ============================================================

print(
    "\nMissing values before filling:"
)

print(
    data.isna()
    .sum()
    .to_string()
)


# ------------------------------------------------------------
# Fill short traffic gaps
# ------------------------------------------------------------

for cam in CAMERAS:

    for prefix in [
        "vehicles",
        "density"
    ]:

        col = f"{prefix}_{cam}"

        data[col] = (
            data[col]
            .interpolate(
                method="linear",
                limit=3
            )
        )


# ------------------------------------------------------------
# Fill short state gaps
# ------------------------------------------------------------

for cam in CAMERAS:

    col = f"state_{cam}"

    data[col] = (
        data[col]
        .interpolate(
            method="nearest",
            limit=3
        )
    )


# ------------------------------------------------------------
# Remove remaining incomplete rows
# ------------------------------------------------------------

data = data.dropna()


# Convert states to integers

for cam in CAMERAS:

    data[f"state_{cam}"] = (
        data[f"state_{cam}"]
        .round()
        .clip(0, 3)
        .astype(int)
    )


print(
    "\nShape after cleaning:",
    data.shape
)


# ============================================================
# STATE VERIFICATION
# ============================================================

print(
    "\nState values after synchronization:"
)

for cam in CAMERAS:

    values = sorted(
        data[
            f"state_{cam}"
        ].unique().tolist()
    )

    print(
        f"Camera {cam}: {values}"
    )


# ============================================================
# TIMESTAMP CONTINUITY
# ============================================================

print(
    "\nChecking timestamp continuity..."
)

gaps = (
    data.index.to_series()
    .diff()
    .dt.total_seconds()
    .div(60)
)

print(
    gaps.describe()
)


# ============================================================
# PROPAGATION ANALYSIS
# ============================================================

print("\n")
print("=" * 65)
print("LAGGED STATE CORRELATION")
print("=" * 65)

print(
    "\nTarget camera:",
    TARGET_CAMERA
)

print(
    "Target state:",
    f"state_{TARGET_CAMERA}"
)

print(
    "\nTesting whether upstream congestion"
)

print(
    "appears before downstream congestion."
)


results = []

target_col = (
    f"state_{TARGET_CAMERA}"
)


# ============================================================
# TEST UPSTREAM CAMERAS
# ============================================================

for upstream in UPSTREAM_CAMERAS:

    upstream_col = (
        f"state_{upstream}"
    )

    print("\n")
    print("-" * 65)

    print(
        f"UPSTREAM CAMERA {upstream}"
    )

    print(
        f"DOWNSTREAM CAMERA {TARGET_CAMERA}"
    )

    print("-" * 65)


    for lag in LAGS:

        steps = lag // 5


        # Upstream state at time t
        upstream_state = (
            data[upstream_col]
        )


        # Downstream state at t + lag
        downstream_future = (
            data[target_col]
            .shift(-steps)
        )


        pair = pd.concat(
            [
                upstream_state,
                downstream_future
            ],
            axis=1
        ).dropna()


        if len(pair) < 10:

            correlation = np.nan

        else:

            correlation = (
                pair.iloc[:, 0]
                .corr(
                    pair.iloc[:, 1]
                )
            )


        results.append(
            {
                "upstream_camera": upstream,
                "downstream_camera": TARGET_CAMERA,
                "lag_minutes": lag,
                "correlation": correlation,
                "samples": len(pair)
            }
        )


        print(
            f"Lag +{lag:02d} min"
            f" | correlation = "
            f"{correlation:.4f}"
            f" | samples = "
            f"{len(pair)}"
        )


# ============================================================
# SAVE CORRELATION RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


lag_output = os.path.join(
    OUTPUT_DIR,
    "lagged_correlations.csv"
)


results_df.to_csv(
    lag_output,
    index=False
)


# ============================================================
# BEST PROPAGATION LAGS
# ============================================================

print("\n")
print("=" * 65)
print("BEST PROPAGATION LAGS")
print("=" * 65)


best_rows = []


for upstream in UPSTREAM_CAMERAS:

    subset = results_df[
        results_df[
            "upstream_camera"
        ] == upstream
    ].copy()


    subset = subset.dropna(
        subset=["correlation"]
    )


    if subset.empty:
        continue


    best = subset.loc[
        subset[
            "correlation"
        ].abs().idxmax()
    ]


    best_rows.append(
        best
    )


    print(
        f"\nCamera {upstream}"
        f" -> Camera {TARGET_CAMERA}"
    )

    print(
        f"Best lag: "
        f"+{int(best['lag_minutes'])} minutes"
    )

    print(
        f"Correlation: "
        f"{best['correlation']:.4f}"
    )


# ============================================================
# SAVE BEST LAGS
# ============================================================

best_output = os.path.join(
    OUTPUT_DIR,
    "best_propagation_lags.csv"
)


if best_rows:

    best_df = pd.DataFrame(
        best_rows
    )

    best_df.to_csv(
        best_output,
        index=False
    )


# ============================================================
# STATE TRANSITION ANALYSIS
# ============================================================

print("\n")
print("=" * 65)
print("STATE TRANSITION ANALYSIS")
print("=" * 65)


transition_results = []


for upstream in UPSTREAM_CAMERAS:

    upstream_col = (
        f"state_{upstream}"
    )


    print(
        f"\nCamera {upstream}"
        f" -> Camera {TARGET_CAMERA}"
    )


    for lag in LAGS:

        steps = lag // 5


        upstream_state = (
            data[upstream_col]
        )

        downstream_state = (
            data[target_col]
            .shift(-steps)
        )


        pair = pd.concat(
            [
                upstream_state,
                downstream_state
            ],
            axis=1
        ).dropna()


        if pair.empty:
            continue


        pair.columns = [
            "upstream",
            "downstream"
        ]


        same_state_rate = (
            pair["upstream"]
            == pair["downstream"]
        ).mean()


        increase_rate = (
            pair["downstream"]
            > pair["upstream"]
        ).mean()


        decrease_rate = (
            pair["downstream"]
            < pair["upstream"]
        ).mean()


        transition_results.append(
            {
                "upstream_camera": upstream,
                "downstream_camera": TARGET_CAMERA,
                "lag_minutes": lag,
                "same_state_rate": same_state_rate,
                "increase_rate": increase_rate,
                "decrease_rate": decrease_rate,
                "samples": len(pair)
            }
        )


transition_df = pd.DataFrame(
    transition_results
)


# ============================================================
# SAVE TRANSITION ANALYSIS
# ============================================================

transition_output = os.path.join(
    OUTPUT_DIR,
    "state_transition_analysis.csv"
)


transition_df.to_csv(
    transition_output,
    index=False
)


# ============================================================
# TRANSITION SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("TRANSITION SUMMARY")
print("=" * 65)


for upstream in UPSTREAM_CAMERAS:

    subset = transition_df[
        transition_df[
            "upstream_camera"
        ] == upstream
    ]


    if subset.empty:
        continue


    best_same = subset.loc[
        subset[
            "same_state_rate"
        ].idxmax()
    ]


    best_increase = subset.loc[
        subset[
            "increase_rate"
        ].idxmax()
    ]


    print(
        f"\nCamera {upstream}"
        f" -> Camera {TARGET_CAMERA}"
    )


    print(
        "Highest same-state rate:"
    )

    print(
        f"  +{int(best_same['lag_minutes'])} min"
        f" : "
        f"{best_same['same_state_rate']:.4f}"
    )


    print(
        "Highest downstream increase rate:"
    )

    print(
        f"  +{int(best_increase['lag_minutes'])} min"
        f" : "
        f"{best_increase['increase_rate']:.4f}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 65)
print("PROPAGATION ANALYSIS COMPLETE")
print("=" * 65)

print("\nFiles saved:")

print(
    lag_output
)

if best_rows:

    print(
        best_output
    )

print(
    transition_output
)

print("\nDONE.")