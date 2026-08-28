import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# CONFIG
# ============================================================

DATASET = (
    "scripts/artifacts/downstream_prediction/"
    "causeway_downstream_dataset.npz"
)

OUTPUT_DIR = "scripts/artifacts/downstream_prediction"

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "causeway_downstream_lstm.keras"
)

RESULT_PATH = os.path.join(
    OUTPUT_DIR,
    "causeway_downstream_lstm_results.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "Very Light",
    "Light",
    "Moderate",
    "Severe"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TRAFLOW DOWNSTREAM LSTM")
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

print("\nDataset:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)


# ============================================================
# CHECK LABELS
# ============================================================

print("\nUnique training labels:")
print(np.unique(Y_train))

print("Unique validation labels:")
print(np.unique(Y_val))


# ============================================================
# MODEL
# ============================================================

T_IN = X_train.shape[1]
N_FEATURES = X_train.shape[2]
T_OUT = Y_train.shape[1]
N_CLASSES = 4


def build_model():

    inputs = layers.Input(
        shape=(T_IN, N_FEATURES),
        name="traffic_input"
    )

    x = layers.LSTM(
        64,
        return_sequences=False,
        name="temporal_lstm"
    )(inputs)

    x = layers.Dense(
        32,
        activation="relu",
        name="traffic_features"
    )(x)

    x = layers.Dropout(
        0.20,
        name="dropout"
    )(x)

    x = layers.Dense(
        T_OUT * N_CLASSES,
        name="future_states"
    )(x)

    x = layers.Reshape(
        (T_OUT, N_CLASSES),
        name="prediction_reshape"
    )(x)

    outputs = layers.Softmax(
        axis=-1,
        name="congestion_probabilities"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="TrafFlow_Downstream_LSTM"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )

    return model


model = build_model()

print("\n")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
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
print("TRAINING DOWNSTREAM LSTM")
print("=" * 60)

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=100,
    batch_size=32,
    shuffle=True,
    callbacks=[
        checkpoint,
        early_stopping
    ],
    verbose=1
)


# ============================================================
# PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATING DOWNSTREAM LSTM")
print("=" * 60)

probabilities = model.predict(
    X_val,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=-1
)

print("Prediction shape:", predictions.shape)


# ============================================================
# OVERALL ACCURACY
# ============================================================

overall_accuracy = accuracy_score(
    Y_val.flatten(),
    predictions.flatten()
)

print("\nOverall accuracy:")
print(f"{overall_accuracy:.4f}")


# ============================================================
# PER-HORIZON ACCURACY
# ============================================================

horizon_accuracy = []

print("\nHorizon accuracy:")

for step in range(T_OUT):

    acc = accuracy_score(
        Y_val[:, step],
        predictions[:, step]
    )

    horizon_accuracy.append(
        float(acc)
    )

    print(
        f"T+{(step + 1) * 5:02d} min: "
        f"{acc:.4f}"
    )


# ============================================================
# TRANSITION DETECTION
# ============================================================

# A sample is considered a transition case when
# the future target state changes at least once.

transition_mask = np.any(
    Y_val[:, 1:] != Y_val[:, :-1],
    axis=1
)

transition_count = int(
    np.sum(transition_mask)
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
    "Transition samples:",
    transition_count
)

print(
    "Transition accuracy:",
    f"{transition_accuracy:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    Y_val.flatten(),
    predictions.flatten(),
    labels=[0, 1, 2, 3]
)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


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

    actual = [
        CLASS_NAMES[x]
        for x in Y_val[i]
    ]

    predicted = [
        CLASS_NAMES[x]
        for x in predictions[i]
    ]

    confidence = np.max(
        probabilities[i],
        axis=-1
    )

    print("\nSample", i + 1)

    print(
        "Actual:   ",
        actual
    )

    print(
        "Predicted:",
        predicted
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

    "model": "TrafFlow Downstream LSTM",

    "dataset": DATASET,

    "input_shape": list(X_train.shape),

    "output_shape": list(Y_train.shape),

    "overall_accuracy":
        float(overall_accuracy),

    "horizon_accuracy":
        horizon_accuracy,

    "transition_samples":
        transition_count,

    "transition_accuracy":
        float(transition_accuracy),

    "confusion_matrix":
        cm.tolist(),

    "best_model":
        MODEL_PATH
}


with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Model saved:",
    MODEL_PATH
)

print(
    "Results saved:",
    RESULT_PATH
)