import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


DATASET = (
    "scripts/artifacts/downstream_prediction/"
    "causeway_downstream_dataset.npz"
)


print("=" * 60)
print("TRAFLOW DOWNSTREAM BASELINE")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(
    DATASET,
    allow_pickle=True
)

X_val = data["X_val"]
Y_val = data["Y_val"]


print("\nX_val:", X_val.shape)
print("Y_val:", Y_val.shape)


# ============================================================
# BASELINE
# ============================================================

# Feature layout:
#
# For each camera:
#
# vehicles
# density
# state
# vehicle_change
# density_change
#
# Therefore:
#
# 2701 -> indices 0..4
# 2702 -> indices 5..9
# 2704 -> indices 10..14
# 2706 -> indices 15..19
#
# Target camera = 2704
# state_2704 = index 12
#
# We use the LAST observed state of camera 2704
# as the prediction for all six future steps.

CURRENT_STATE_INDEX = 12


current_state = (
    X_val[:, -1, CURRENT_STATE_INDEX]
)


# Convert to integer
current_state = np.rint(
    current_state
).astype(int)


# ============================================================
# PREDICT ALL SIX FUTURE STEPS
# ============================================================

predictions = np.repeat(
    current_state[:, np.newaxis],
    6,
    axis=1
)


# ============================================================
# EVALUATE
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE RESULTS")
print("=" * 60)


overall_accuracy = accuracy_score(
    Y_val.flatten(),
    predictions.flatten()
)


print(
    f"\nOverall accuracy: "
    f"{overall_accuracy:.4f}"
)


for step in range(6):

    accuracy = accuracy_score(
        Y_val[:, step],
        predictions[:, step]
    )

    print(
        f"T+{(step + 1) * 5:02d} min accuracy: "
        f"{accuracy:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX - ALL HORIZONS")
print("=" * 60)


cm = confusion_matrix(
    Y_val.flatten(),
    predictions.flatten(),
    labels=[0, 1, 2, 3]
)

print(cm)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)


labels = [
    "Very Light",
    "Light",
    "Moderate",
    "Severe"
]


for i in range(
    min(10, len(Y_val))
):

    actual = [
        labels[x]
        for x in Y_val[i]
    ]

    predicted = [
        labels[x]
        for x in predictions[i]
    ]

    print(
        f"\nSample {i + 1}"
    )

    print(
        "Actual:   ",
        actual
    )

    print(
        "Baseline: ",
        predicted
    )


# ============================================================
# TRANSITION ANALYSIS
# ============================================================

transition_mask = np.any(
    Y_val != predictions,
    axis=1
)


transition_count = (
    transition_mask.sum()
)


if transition_count > 0:

    transition_accuracy = accuracy_score(
        Y_val[transition_mask].flatten(),
        predictions[transition_mask].flatten()
    )

else:

    transition_accuracy = 0.0


print("\n")
print("=" * 60)
print("TRANSITION ANALYSIS")
print("=" * 60)

print(
    "Samples with future state change:",
    transition_count
)

print(
    "Transition accuracy:",
    f"{transition_accuracy:.4f}"
)


print("\n")
print("=" * 60)
print("BASELINE COMPLETE")
print("=" * 60)