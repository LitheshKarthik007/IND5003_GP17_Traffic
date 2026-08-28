import os
import numpy as np
import pandas as pd

# ============================================================
# TRAFLOW - CONGESTION PROPAGATION EVENT DETECTION
# ============================================================

INPUT = "scripts/result_ml/clustered_traffic.csv"

OUTPUT_DIR = "scripts/artifacts/propagation_analysis"

OUTPUT_EVENTS = os.path.join(
    OUTPUT_DIR,
    "propagation_events.csv"
)

CAMERAS = [2701, 2702, 2704, 2706]

# Upstream -> downstream relationships
LINKS = [
    (2701, 2704),
    (2702, 2704),
    (2706, 2704)
]

# Maximum propagation window
MAX_LAG = 60

# Minimum state increase required
MIN_STATE_INCREASE = 1


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("TRAFLOW CONGESTION PROPAGATION EVENT DETECTION")
print("=" * 65)

print("\nLoading:", INPUT)

df = pd.read_csv(
    INPUT,
    parse_dates=["timestamp"]
)

df = df[df["camera_id"].isin(CAMERAS)].copy()

df = df.sort_values(
    ["camera_id", "timestamp"]
)

print("\nRows:", len(df))
print("Cameras:", sorted(df["camera_id"].unique()))


# ============================================================
# CORRECT STATE VALUES
# ============================================================

print("\nCluster values before correction:")
print(sorted(df["cluster"].dropna().unique()))

# cluster is already 0,1,2,3
# Keep it as the final state representation.

df["state"] = (
    pd.to_numeric(
        df["cluster"],
        errors="coerce"
    )
)

df = df.dropna(
    subset=["state"]
)

df["state"] = df["state"].round().astype(int)

# Keep only valid classes
df = df[
    df["state"].between(0, 3)
].copy()

print("\nCluster values after correction:")
print(sorted(df["state"].unique()))


# ============================================================
# CREATE SYNCHRONIZED 5-MINUTE TABLE
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

    c = c[
        [
            "timestamp",
            "state",
            "total_vehicles",
            "vehicles_per_mpx"
        ]
    ]

    # Remove duplicate timestamps
    c = (
        c.groupby("timestamp")
        .agg({
            "state": "mean",
            "total_vehicles": "mean",
            "vehicles_per_mpx": "mean"
        })
    )

    c.columns = [
        f"state_{cam}",
        f"vehicles_{cam}",
        f"density_{cam}"
    ]

    # 5-minute aggregation
    c = c.resample("5min").mean()

    tables.append(c)


# ============================================================
# MERGE CAMERAS
# ============================================================

data = pd.concat(
    tables,
    axis=1
)

data = data.sort_index()

print("\nSynchronized shape:", data.shape)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

state_columns = [
    f"state_{cam}"
    for cam in CAMERAS
]

print("\nMissing state values:")
print(
    data[state_columns]
    .isna()
    .sum()
    .to_string()
)

# IMPORTANT:
# We do NOT interpolate congestion states.
# We only forward/backward fill very small gaps.

data[state_columns] = (
    data[state_columns]
    .ffill(limit=2)
)

data[state_columns] = (
    data[state_columns]
    .bfill(limit=2)
)

# Remove rows where any camera state is still missing

data = data.dropna(
    subset=state_columns
)

# Convert states back to integers

for col in state_columns:

    data[col] = (
        data[col]
        .round()
        .astype(int)
    )

print(
    "\nShape after state cleaning:",
    data.shape
)


# ============================================================
# CHECK STATE VALUES
# ============================================================

print(
    "\nState values after synchronization:"
)

for cam in CAMERAS:

    print(
        f"Camera {cam}:",
        sorted(
            data[
                f"state_{cam}"
            ].unique()
        )
    )


# ============================================================
# DETECT CONTINUOUS 5-MINUTE SEQUENCES
# ============================================================

print(
    "\nChecking timestamp continuity..."
)

time_diff = (
    data.index.to_series()
    .diff()
)

data["sequence"] = (
    time_diff != pd.Timedelta(
        minutes=5
    )
).cumsum()

sequence_sizes = (
    data.groupby("sequence")
    .size()
)

print(
    "Number of sequences:",
    len(sequence_sizes)
)

print(
    "Longest sequence:",
    sequence_sizes.max()
)


# ============================================================
# PROPAGATION EVENT DETECTION
# ============================================================

events = []

print("\n")
print("=" * 65)
print("DETECTING PROPAGATION EVENTS")
print("=" * 65)

for upstream, downstream in LINKS:

    upstream_col = (
        f"state_{upstream}"
    )

    downstream_col = (
        f"state_{downstream}"
    )

    print(
        f"\n{upstream} -> {downstream}"
    )

    event_count = 0

    # Process each continuous sequence separately

    for seq_id, seq in data.groupby(
        "sequence"
    ):

        seq = seq.sort_index()

        if len(seq) < 2:
            continue

        timestamps = seq.index

        upstream_states = (
            seq[upstream_col]
            .to_numpy()
        )

        downstream_states = (
            seq[downstream_col]
            .to_numpy()
        )

        # Examine every point as a possible upstream event

        for i in range(
            len(seq) - 1
        ):

            upstream_state = (
                upstream_states[i]
            )

            upstream_time = (
                timestamps[i]
            )

            # Look ahead from +5 to +60 minutes

            for lag in range(
                5,
                MAX_LAG + 1,
                5
            ):

                future_time = (
                    upstream_time
                    + pd.Timedelta(
                        minutes=lag
                    )
                )

                # Locate exact future timestamp

                if future_time not in seq.index:
                    continue

                j = seq.index.get_loc(
                    future_time
                )

                downstream_state = (
                    downstream_states[j]
                )

                state_increase = (
                    downstream_state
                    - upstream_state
                )

                # Propagation event:
                #
                # upstream congestion state
                # is followed by
                # increased downstream state.

                if (
                    state_increase
                    >= MIN_STATE_INCREASE
                ):

                    # Check that this is the
                    # earliest downstream increase
                    # within the propagation window.

                    earlier_increase = False

                    for earlier_lag in range(
                        5,
                        lag,
                        5
                    ):

                        earlier_time = (
                            upstream_time
                            + pd.Timedelta(
                                minutes=earlier_lag
                            )
                        )

                        if (
                            earlier_time
                            not in seq.index
                        ):
                            continue

                        k = seq.index.get_loc(
                            earlier_time
                        )

                        earlier_state = (
                            downstream_states[k]
                        )

                        if (
                            earlier_state
                            > upstream_state
                        ):

                            earlier_increase = True
                            break

                    if earlier_increase:
                        continue

                    # Event detected

                    event_count += 1

                    events.append({

                        "upstream_camera":
                            upstream,

                        "downstream_camera":
                            downstream,

                        "upstream_time":
                            upstream_time,

                        "downstream_time":
                            future_time,

                        "propagation_lag_min":
                            lag,

                        "upstream_state":
                            int(upstream_state),

                        "downstream_state":
                            int(downstream_state),

                        "state_increase":
                            int(state_increase),

                        "sequence_id":
                            int(seq_id)

                    })

                    # One event per upstream timestamp

                    break

    print(
        "Propagation events:",
        event_count
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

events_df = pd.DataFrame(
    events
)


# ============================================================
# EVENT SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("PROPAGATION EVENT SUMMARY")
print("=" * 65)

print(
    "\nTotal propagation events:",
    len(events_df)
)

if len(events_df) > 0:

    print(
        "\nEvents by link:"
    )

    print(
        events_df
        .groupby(
            [
                "upstream_camera",
                "downstream_camera"
            ]
        )
        .size()
        .to_string()
    )

    print(
        "\nPropagation lag distribution:"
    )

    print(
        events_df[
            "propagation_lag_min"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nAverage propagation lag:"
    )

    print(
        events_df[
            "propagation_lag_min"
        ].mean()
    )

    print(
        "\nAverage state increase:"
    )

    print(
        events_df[
            "state_increase"
        ].mean()
    )

    print(
        "\nFirst 10 events:"
    )

    print(
        events_df
        .head(10)
        .to_string(index=False)
    )

else:

    print(
        "\nNo propagation events detected."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

events_df.to_csv(
    OUTPUT_EVENTS,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 65)
print("PROPAGATION EVENT DETECTION COMPLETE")
print("=" * 65)

print(
    "\nSaved:"
)

print(
    OUTPUT_EVENTS
)

print("\nDONE.")