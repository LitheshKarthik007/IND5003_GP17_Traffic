import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = (
    "scripts/artifacts/spatial_training/"
    "causeway_spatial_dataset.npz"
)

MODEL_DIR = "scripts/artifacts/spatial_training"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "causeway_spatial_lstm.keras"
)

LOG_PATH = os.path.join(
    MODEL_DIR,
    "causeway_spatial_lstm_log.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("======================================")
print("LOADING SPATIAL DATASET")
print("======================================")

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
# MODEL PARAMETERS
# ============================================================

T_IN = X_train.shape[1]
N_FEATURES = X_train.shape[2]
T_OUT = y_train.shape[1]

N_CLASSES = 4

print()
print("======================================")
print("MODEL CONFIGURATION")
print("======================================")

print("Input timesteps :", T_IN)
print("Features        :", N_FEATURES)
print("Output steps    :", T_OUT)
print("Classes         :", N_CLASSES)


# ============================================================
# BUILD SPATIAL-TEMPORAL LSTM
# ============================================================

inp = layers.Input(
    shape=(T_IN, N_FEATURES),
    name="traffic_input"
)

# Learn temporal patterns from all camera features
x = layers.LSTM(
    64,
    return_sequences=False,
    name="spatial_temporal_lstm"
)(inp)

x = layers.Dense(
    32,
    activation="relu",
    name="dense_features"
)(x)

# 6 future steps × 4 congestion classes
x = layers.Dense(
    T_OUT * N_CLASSES,
    name="future_predictions"
)(x)

x = layers.Reshape(
    (T_OUT, N_CLASSES),
    name="prediction_reshape"
)(x)

# Probability for each congestion class
out = layers.Softmax(
    axis=-1,
    name="congestion_probabilities"
)(x)


model = models.Model(
    inputs=inp,
    outputs=out,
    name="TrafFlow_Spatial_Temporal_LSTM"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print()
print("======================================")
print("TRAINING SPATIAL-TEMPORAL LSTM")
print("======================================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=100,

    batch_size=32,

    callbacks=[
        checkpoint,
        early_stop
    ],

    # IMPORTANT:
    # Do not randomly shuffle time-series samples.
    shuffle=False,

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("Loading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("======================================")
print("EVALUATING MODEL")
print("======================================")

probabilities = best_model.predict(
    X_val,
    verbose=0
)

# probabilities:
# (samples, 6 future steps, 4 classes)

predictions = np.argmax(
    probabilities,
    axis=-1
)


print("Prediction shape:", predictions.shape)


# ============================================================
# OVERALL ACCURACY
# ============================================================

overall_accuracy = accuracy_score(
    y_val.flatten(),
    predictions.flatten()
)


# ============================================================
# PER-HORIZON ACCURACY
# ============================================================

step_accuracy = []

for step in range(T_OUT):

    acc = accuracy_score(
        y_val[:, step],
        predictions[:, step]
    )

    step_accuracy.append(float(acc))


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("======================================")
print("SPATIAL LSTM RESULTS")
print("======================================")

print(
    f"Overall accuracy: "
    f"{overall_accuracy:.4f}"
)

print()

for i, acc in enumerate(step_accuracy):

    minutes = (i + 1) * 5

    print(
        f"T+{minutes:02d} min accuracy: "
        f"{acc:.4f}"
    )


# ============================================================
# CLASS NAMES
# ============================================================

class_names = {
    0: "Very Light",
    1: "Light",
    2: "Moderate",
    3: "Severe"
}


# ============================================================
# SHOW SAMPLE PREDICTIONS
# ============================================================

print()
print("======================================")
print("SAMPLE PREDICTIONS")
print("======================================")

NUM_SAMPLES = min(10, len(X_val))

for i in range(NUM_SAMPLES):

    actual = y_val[i]

    predicted = predictions[i]

    confidence = np.max(
        probabilities[i],
        axis=-1
    )

    print()
    print(f"Sample {i + 1}")

    print(
        "Actual:   ",
        [class_names[int(x)] for x in actual]
    )

    print(
        "Predicted:",
        [class_names[int(x)] for x in predicted]
    )

    print(
        "Confidence:",
        np.round(confidence, 3)
    )


# ============================================================
# TRANSITION TEST
# ============================================================

print()
print("======================================")
print("TRANSITION CASE ANALYSIS")
print("======================================")

transition_count = 0

transition_correct = []

for i in range(len(y_val)):

    actual = y_val[i]

    predicted = predictions[i]

    # Check whether actual traffic changes
    # during the prediction horizon

    if len(np.unique(actual)) > 1:

        transition_count += 1

        acc = accuracy_score(
            actual,
            predicted
        )

        transition_correct.append(
            float(acc)
        )


if transition_count > 0:

    transition_accuracy = np.mean(
        transition_correct
    )

    print(
        "Transition samples:",
        transition_count
    )

    print(
        "Transition accuracy:",
        f"{transition_accuracy:.4f}"
    )

else:

    transition_accuracy = None

    print(
        "No transition samples found."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

result = {

    "model":
        "TrafFlow Spatial-Temporal LSTM",

    "input_timesteps":
        int(T_IN),

    "input_history_minutes":
        int(T_IN * 5),

    "features_per_timestep":
        int(N_FEATURES),

    "output_timesteps":
        int(T_OUT),

    "prediction_horizon_minutes":
        int(T_OUT * 5),

    "classes":
        class_names,

    "training_samples":
        int(len(X_train)),

    "validation_samples":
        int(len(X_val)),

    "overall_accuracy":
        float(overall_accuracy),

    "accuracy_per_horizon":
        step_accuracy,

    "transition_samples":
        int(transition_count),

    "transition_accuracy":
        (
            float(transition_accuracy)
            if transition_accuracy is not None
            else None
        ),

    "model_path":
        MODEL_PATH
}


with open(
    LOG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")

print("Best model:")
print(MODEL_PATH)

print()
print("Results:")
print(LOG_PATH)