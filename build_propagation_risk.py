import os
import numpy as np
import pandas as pd

# ============================================================
# TRAFLOW - PROPAGATION RISK ENGINE
# ============================================================

INPUT = (
    "scripts/artifacts/propagation_analysis/"
    "propagation_events.csv"
)

OUTPUT_DIR = (
    "scripts/artifacts/propagation_analysis"
)

OUTPUT_EVENTS = os.path.join(
    OUTPUT_DIR,
    "propagation_risk_events.csv"
)

OUTPUT_LINKS = os.path.join(
    OUTPUT_DIR,
    "propagation_link_risk.csv"
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "propagation_risk_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# State meanings
STATE_NAMES = {
    0: "Very Light",
    1: "Light",
    2: "Moderate",
    3: "Severe"
}

# Maximum propagation lag considered
MAX_LAG = 60

# ============================================================
# LOAD EVENTS
# ============================================================

print("=" * 65)
print("TRAFLOW PROPAGATION RISK ENGINE")
print("=" * 65)

print("\nLoading:", INPUT)

events = pd.read_csv(
    INPUT,
    parse_dates=[
        "upstream_time",
        "downstream_time"
    ]
)

print(
    "Rows:",
    len(events)
)

print(
    "Columns:",
    events.columns.tolist()
)


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "upstream_camera",
    "downstream_camera",
    "upstream_time",
    "downstream_time",
    "propagation_lag_min",
    "upstream_state",
    "downstream_state",
    "state_increase"
]

missing_columns = [
    c for c in required_columns
    if c not in events.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================================
# CLEAN DATA
# ============================================================

events = events.dropna(
    subset=required_columns
).copy()

# Ensure integer state values

events["upstream_state"] = (
    events["upstream_state"]
    .round()
    .astype(int)
)

events["downstream_state"] = (
    events["downstream_state"]
    .round()
    .astype(int)
)

events["state_increase"] = (
    events["state_increase"]
    .round()
    .astype(int)
)

events["propagation_lag_min"] = (
    events["propagation_lag_min"]
    .astype(float)
)

# Keep valid ranges

events = events[
    events["upstream_state"].between(0, 3)
    & events["downstream_state"].between(0, 3)
    & events["state_increase"].between(1, 3)
    & events["propagation_lag_min"].between(
        5,
        MAX_LAG
    )
].copy()

print(
    "\nValid propagation events:",
    len(events)
)


# ============================================================
# ADD STATE LABELS
# ============================================================

events["upstream_state_name"] = (
    events["upstream_state"]
    .map(STATE_NAMES)
)

events["downstream_state_name"] = (
    events["downstream_state"]
    .map(STATE_NAMES)
)


# ============================================================
# 1. SEVERITY SCORE
# ============================================================

# Upstream state:
#
# 0 -> 0.00
# 1 -> 0.33
# 2 -> 0.67
# 3 -> 1.00

events["severity_score"] = (
    events["upstream_state"] / 3.0
)


# ============================================================
# 2. STATE INCREASE SCORE
# ============================================================

# Increase:
#
# +1 -> 0.33
# +2 -> 0.67
# +3 -> 1.00

events["increase_score"] = (
    events["state_increase"] / 3.0
)


# ============================================================
# 3. SPEED SCORE
# ============================================================

# Faster propagation = greater immediate risk.
#
# 5 minutes -> 1.00
# 60 minutes -> 0.00

events["speed_score"] = (
    1.0
    - (
        events["propagation_lag_min"] - 5
    ) / (
        MAX_LAG - 5
    )
)

events["speed_score"] = (
    events["speed_score"]
    .clip(0, 1)
)


# ============================================================
# 4. LINK RELIABILITY
# ============================================================

print(
    "\nCalculating link reliability..."
)

link_counts = (
    events
    .groupby(
        [
            "upstream_camera",
            "downstream_camera"
        ]
    )
    .size()
    .reset_index(
        name="event_count"
    )
)

print(
    "\nPropagation event counts:"
)

print(
    link_counts.to_string(
        index=False
    )
)

# Normalize link event frequency

max_events = (
    link_counts["event_count"].max()
)

if max_events > 0:

    link_counts["link_reliability"] = (
        link_counts["event_count"]
        / max_events
    )

else:

    link_counts["link_reliability"] = 0.0


events = events.merge(
    link_counts[
        [
            "upstream_camera",
            "downstream_camera",
            "event_count",
            "link_reliability"
        ]
    ],
    on=[
        "upstream_camera",
        "downstream_camera"
    ],
    how="left"
)


# ============================================================
# 5. RECENCY SCORE
# ============================================================

latest_event_time = (
    events["upstream_time"].max()
)

events["age_minutes"] = (
    (
        latest_event_time
        - events["upstream_time"]
    )
    .dt.total_seconds()
    / 60.0
)

# Exponential decay.
#
# Recent events receive higher weight.

DECAY_MINUTES = 120.0

events["recency_score"] = np.exp(
    -events["age_minutes"]
    / DECAY_MINUTES
)

events["recency_score"] = (
    events["recency_score"]
    .clip(0, 1)
)


# ============================================================
# 6. PROPAGATION RISK SCORE
# ============================================================

# Weighted combination:
#
# Severity       = 30%
# State increase = 30%
# Speed          = 15%
# Reliability    = 15%
# Recency        = 10%

events["risk_raw"] = (

    0.30
    * events["severity_score"]

    + 0.30
    * events["increase_score"]

    + 0.15
    * events["speed_score"]

    + 0.15
    * events["link_reliability"]

    + 0.10
    * events["recency_score"]
)


# Convert to 0-100

events["risk_score"] = (
    events["risk_raw"] * 100
)

events["risk_score"] = (
    events["risk_score"]
    .round(2)
)


# ============================================================
# 7. RISK LEVEL
# ============================================================

def risk_level(score):

    if score >= 75:
        return "HIGH"

    elif score >= 50:
        return "MEDIUM"

    elif score >= 25:
        return "LOW"

    else:
        return "MINIMAL"


events["risk_level"] = (
    events["risk_score"]
    .apply(risk_level)
)


# ============================================================
# 8. SORT EVENTS
# ============================================================

events = events.sort_values(
    "risk_score",
    ascending=False
)


# ============================================================
# DISPLAY TOP EVENTS
# ============================================================

print("\n")
print("=" * 65)
print("TOP PROPAGATION RISK EVENTS")
print("=" * 65)

display_columns = [
    "upstream_camera",
    "downstream_camera",
    "upstream_time",
    "downstream_time",
    "propagation_lag_min",
    "upstream_state_name",
    "downstream_state_name",
    "state_increase",
    "risk_score",
    "risk_level"
]

print(
    events[
        display_columns
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# LINK-LEVEL RISK
# ============================================================

print("\n")
print("=" * 65)
print("LINK-LEVEL PROPAGATION RISK")
print("=" * 65)

link_risk = (
    events
    .groupby(
        [
            "upstream_camera",
            "downstream_camera"
        ]
    )
    .agg(
        event_count=(
            "risk_score",
            "count"
        ),

        average_risk=(
            "risk_score",
            "mean"
        ),

        maximum_risk=(
            "risk_score",
            "max"
        ),

        average_lag=(
            "propagation_lag_min",
            "mean"
        ),

        average_state_increase=(
            "state_increase",
            "mean"
        ),

        high_risk_events=(
            "risk_level",
            lambda x:
            (x == "HIGH").sum()
        )
    )
    .reset_index()
)

link_risk["average_risk"] = (
    link_risk["average_risk"]
    .round(2)
)

link_risk["maximum_risk"] = (
    link_risk["maximum_risk"]
    .round(2)
)

link_risk["average_lag"] = (
    link_risk["average_lag"]
    .round(2)
)

link_risk["average_state_increase"] = (
    link_risk["average_state_increase"]
    .round(2)
)

link_risk["link_risk_level"] = (
    link_risk["average_risk"]
    .apply(risk_level)
)

link_risk = link_risk.sort_values(
    "average_risk",
    ascending=False
)

print(
    link_risk.to_string(
        index=False
    )
)


# ============================================================
# CORRIDOR SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("CORRIDOR PROPAGATION SUMMARY")
print("=" * 65)

overall_risk = (
    events["risk_score"].mean()
)

maximum_risk = (
    events["risk_score"].max()
)

high_risk_count = (
    events["risk_level"]
    .eq("HIGH")
    .sum()
)

medium_risk_count = (
    events["risk_level"]
    .eq("MEDIUM")
    .sum()
)

low_risk_count = (
    events["risk_level"]
    .eq("LOW")
    .sum()
)

print(
    f"\nAverage propagation risk : "
    f"{overall_risk:.2f}/100"
)

print(
    f"Maximum propagation risk : "
    f"{maximum_risk:.2f}/100"
)

print(
    f"HIGH risk events          : "
    f"{high_risk_count}"
)

print(
    f"MEDIUM risk events        : "
    f"{medium_risk_count}"
)

print(
    f"LOW risk events           : "
    f"{low_risk_count}"
)


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({

    "metric": [
        "total_events",
        "average_risk",
        "maximum_risk",
        "high_risk_events",
        "medium_risk_events",
        "low_risk_events",
        "average_propagation_lag",
        "average_state_increase"
    ],

    "value": [
        len(events),

        round(
            overall_risk,
            2
        ),

        round(
            maximum_risk,
            2
        ),

        high_risk_count,

        medium_risk_count,

        low_risk_count,

        round(
            events[
                "propagation_lag_min"
            ].mean(),
            2
        ),

        round(
            events[
                "state_increase"
            ].mean(),
            2
        )
    ]
})


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

events.to_csv(
    OUTPUT_EVENTS,
    index=False
)

link_risk.to_csv(
    OUTPUT_LINKS,
    index=False
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 65)
print("PROPAGATION RISK ENGINE COMPLETE")
print("=" * 65)

print("\nFiles saved:")

print(
    OUTPUT_EVENTS
)

print(
    OUTPUT_LINKS
)

print(
    OUTPUT_SUMMARY
)

print("\nDONE.")