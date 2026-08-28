import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Reshape,
    Softmax,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET = (
    "scripts/artifacts/upstream_prediction/"
    "causeway_upstream_dataset.npz"
)

OUTPUT_DIR = "scripts/artifacts/upstream_prediction"

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "causeway_upstream_lstm.keras"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "causeway_upstream_lstm_results.json"
)

T_IN = 6
T_OUT = 6
N_CLASSES = 4

CLASS_NAMES = [
    "Very Light",
    "Light",
    "Moderate",
    "Severe"
]


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("TRAFLOW UPSTREAM-ONLY LSTM")
print("=" * 60)

print("\nLoading dataset:")
print(DATASET)

data = np.load(
    DATASET,
    allow_pickle=True
)

X_train = data["X_train"].astype(np.float32)
Y_train = data["Y_train"].astype(np.int64)

X_val = data["X_val"].astype(np.float32)
Y_val = data["Y_val"].astype(np.int64)

times_train = data["times_train"]
times_val = data["times_val"]

print("\nDataset:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)


# ============================================================
# VALIDATE DATA
# ============================================================

print("\nChecking data...")

print(
    "Training classes:",
    np.unique(Y_train)
)

print(
    "Validation classes:",
    np.unique(Y_val)
)

if not np.all(
    np.isin(Y_train, [0, 1, 2, 3])
):
    raise ValueError(
        "Invalid class in Y_train"
    )

if not np.all(
    np.isin(Y_val, [0, 1, 2, 3])
):
    raise ValueError(
        "Invalid class in Y_val"
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTraining target distribution:")

train_counts = np.bincount(
    Y_train.flatten(),
    minlength=N_CLASSES
)

for cls in range(N_CLASSES):
    print(
        f"{cls} = {CLASS_NAMES[cls]}: "
        f"{train_counts[cls]}"
    )


print("\nValidation target distribution:")

val_counts = np.bincount(
    Y_val.flatten(),
    minlength=N_CLASSES
)

for cls in range(N_CLASSES):
    print(
        f"{cls} = {CLASS_NAMES[cls]}: "
        f"{val_counts[cls]}"
    )


# ============================================================
# BUILD MODEL
# ============================================================

N_FEATURES = X_train.shape[2]

print("\n")
print("=" * 60)
print("BUILDING UPSTREAM-ONLY LSTM")
print("=" * 60)

print("Input shape:")
print(
    f"(batch, {T_IN}, {N_FEATURES})"
)


inputs = Input(
    shape=(T_IN, N_FEATURES),
    name="upstream_traffic_input"
)


# LSTM learns how upstream traffic evolves
x = LSTM(
    64,
    return_sequences=False,
    name="upstream_temporal_lstm"
)(inputs)


x = Dropout(
    0.20,
    name="regularization"
)(x)


x = Dense(
    32,
    activation="relu",
    name="upstream_features"
)(x)


# 6 future steps × 4 congestion classes
x = Dense(
    T_OUT * N_CLASSES,
    name="future_logits"
)(x)


x = Reshape(
    (T_OUT, N_CLASSES),
    name="prediction_reshape"
)(x)


outputs = Softmax(
    axis=-1,
    name="future_congestion_probabilities"
)(x)


model = Model(
    inputs=inputs,
    outputs=outputs,
    name="TrafFlow_Upstream_Only_LSTM"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel summary:")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING UPSTREAM-ONLY LSTM")
print("=" * 60)

history = model.fit(
    X_train,
    Y_train,
    validation_data=(
        X_val,
        Y_val
    ),
    epochs=100,
    batch_size=32,
    callbacks=[
        checkpoint,
        early_stopping
    ],
    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATING UPSTREAM-ONLY MODEL")
print("=" * 60)

probabilities = model.predict(
    X_val,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=-1
)


print(
    "Prediction shape:",
    predictions.shape
)


# ============================================================
# ACCURACY BY HORIZON
# ============================================================

horizon_accuracies = []

for h in range(T_OUT):

    accuracy = np.mean(
        predictions[:, h]
        ==
        Y_val[:, h]
    )

    horizon_accuracies.append(
        float(accuracy)
    )

    print(
        f"T+{(h + 1) * 5:02d} min accuracy: "
        f"{accuracy:.4f}"
    )


# ============================================================
# OVERALL ACCURACY
# ============================================================

overall_accuracy = np.mean(
    predictions == Y_val
)

print("\n")
print("=" * 60)
print("UPSTREAM-ONLY RESULTS")
print("=" * 60)

print(
    f"Overall accuracy: "
    f"{overall_accuracy:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (N_CLASSES, N_CLASSES),
    dtype=int
)

for actual, predicted in zip(
    Y_val.flatten(),
    predictions.flatten()
):

    confusion[
        actual,
        predicted
    ] += 1


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(confusion)


# ============================================================
# TRANSITION ANALYSIS
# ============================================================

future_changes = np.any(
    Y_val[:, 1:]
    !=
    Y_val[:, :-1],
    axis=1
)

transition_count = np.sum(
    future_changes
)

print("\n")
print("=" * 60)
print("TRANSITION ANALYSIS")
print("=" * 60)

print(
    "Samples with future state change:",
    transition_count
)

if transition_count > 0:

    transition_correct = np.mean(
        predictions[future_changes]
        ==
        Y_val[future_changes]
    )

    print(
        "Transition accuracy:",
        f"{transition_correct:.4f}"
    )

else:

    transition_correct = None

    print(
        "No transition samples found."
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(
    min(10, len(X_val))
):

    actual_names = [
        CLASS_NAMES[c]
        for c in Y_val[i]
    ]

    predicted_names = [
        CLASS_NAMES[c]
        for c in predictions[i]
    ]

    confidence = [
        float(
            probabilities[
                i,
                h,
                predictions[i, h]
            ]
        )
        for h in range(T_OUT)
    ]

    print(
        f"\nSample {i + 1}"
    )

    print(
        "Actual:   ",
        actual_names
    )

    print(
        "Predicted:",
        predicted_names
    )

    print(
        "Confidence:",
        np.round(
            confidence,
            3
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": "TrafFlow_Upstream_Only_LSTM",

    "input_cameras": [
        2701,
        2702,
        2706
    ],

    "target_camera": 2704,

    "input_history_minutes": 30,

    "prediction_horizons_minutes": [
        5,
        10,
        15,
        20,
        25,
        30
    ],

    "number_of_features": int(
        N_FEATURES
    ),

    "training_samples": int(
        len(X_train)
    ),

    "validation_samples": int(
        len(X_val)
    ),

    "overall_accuracy": float(
        overall_accuracy
    ),

    "horizon_accuracy": {
        f"T+{(i + 1) * 5}": float(
            horizon_accuracies[i]
        )
        for i in range(T_OUT)
    },

    "confusion_matrix":
        confusion.tolist(),

    "transition_samples":
        int(transition_count),

    "transition_accuracy":
        (
            float(transition_correct)
            if transition_correct is not None
            else None
        )
}


with open(
    RESULTS_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("UPSTREAM-ONLY TRAINING COMPLETE")
print("=" * 60)

print(
    "Model saved:"
)

print(
    MODEL_PATH
)

print(
    "\nResults saved:"
)

print(
    RESULTS_PATH
)

print("\nDONE.")