import os
import json
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


DATASET = (
    "scripts/artifacts/spatial_training/"
    "causeway_spatial_dataset.npz"
)

MODEL_DIR = "scripts/artifacts/spatial_training"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "causeway_xgboost.json"
)

RESULT_PATH = os.path.join(
    MODEL_DIR,
    "causeway_xgboost_results.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

data = np.load(DATASET)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ============================================================
# FLATTEN TIME WINDOW
# ============================================================

# 24 timesteps × 8 features
#
# becomes
#
# 192 features

X_train_flat = X_train.reshape(
    X_train.shape[0],
    -1
)

X_val_flat = X_val.reshape(
    X_val.shape[0],
    -1
)


print()
print("Flattened:")
print("X_train:", X_train_flat.shape)
print("X_val:", X_val_flat.shape)


# ============================================================
# TRAIN ONE MODEL PER FUTURE HORIZON
# ============================================================

T_OUT = y_train.shape[1]

predictions = np.zeros_like(y_val)

models = []


for step in range(T_OUT):

    minutes = (step + 1) * 5

    print()
    print("======================================")
    print(f"TRAINING T+{minutes} MIN MODEL")
    print("======================================")

    y_train_step = y_train[:, step]

    y_val_step = y_val[:, step]

    model = XGBClassifier(

        n_estimators=300,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="multi:softprob",

        num_class=4,

        eval_metric="mlogloss",

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_train_flat,
        y_train_step
    )

    pred = model.predict(
        X_val_flat
    )

    predictions[:, step] = pred

    accuracy = accuracy_score(
        y_val_step,
        pred
    )

    print(
        f"T+{minutes} min accuracy: "
        f"{accuracy:.4f}"
    )

    models.append(model)


# ============================================================
# OVERALL ACCURACY
# ============================================================

overall_accuracy = accuracy_score(
    y_val.flatten(),
    predictions.flatten()
)


print()
print("======================================")
print("XGBOOST SPATIAL RESULTS")
print("======================================")

print(
    f"Overall accuracy: "
    f"{overall_accuracy:.4f}"
)


# ============================================================
# PER HORIZON
# ============================================================

step_accuracy = []

for step in range(T_OUT):

    accuracy = accuracy_score(
        y_val[:, step],
        predictions[:, step]
    )

    step_accuracy.append(
        float(accuracy)
    )

    minutes = (step + 1) * 5

    print(
        f"T+{minutes:02d} min accuracy: "
        f"{accuracy:.4f}"
    )


# ============================================================
# TRANSITION ACCURACY
# ============================================================

transition_samples = []

for i in range(len(y_val)):

    actual = y_val[i]

    if len(np.unique(actual)) > 1:

        transition_samples.append(i)


if transition_samples:

    transition_actual = y_val[
        transition_samples
    ]

    transition_pred = predictions[
        transition_samples
    ]

    transition_accuracy = accuracy_score(
        transition_actual.flatten(),
        transition_pred.flatten()
    )

else:

    transition_accuracy = None


print()

print(
    "Transition samples:",
    len(transition_samples)
)

if transition_accuracy is not None:

    print(
        "Transition accuracy:",
        f"{transition_accuracy:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": "XGBoost spatial baseline",

    "overall_accuracy":
        float(overall_accuracy),

    "accuracy_per_horizon":
        step_accuracy,

    "transition_samples":
        len(transition_samples),

    "transition_accuracy":
        (
            float(transition_accuracy)
            if transition_accuracy is not None
            else None
        )
}


with open(
    RESULT_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print()
print("Results saved:")
print(RESULT_PATH)
