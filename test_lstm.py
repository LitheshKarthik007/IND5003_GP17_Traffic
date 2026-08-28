import numpy as np
import tensorflow as tf

# -----------------------------
# 1. Load validation dataset
# -----------------------------
data = np.load(
    "scripts/artifacts/phase_cluster/phase_cluster_causeway_dataset.npz"
)

X_val = data["X_val"]
y_val = data["y_val"]

# -----------------------------
# 2. Load pretrained model
# -----------------------------
model = tf.keras.models.load_model(
    "scripts/artifacts/model_cluster/causeway_lstm_best.h5",
    compile=False
)

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# -----------------------------
# 3. Make predictions
# -----------------------------
pred_prob = model.predict(X_val, verbose=0)

# Convert probabilities to class 1-4
pred = np.argmax(pred_prob, axis=-1) + 1

# -----------------------------
# 4. Calculate accuracy
# -----------------------------
accuracy = np.mean(pred == y_val)

print("\n==============================")
print("      CAUSEWAY LSTM")
print("==============================")

print("Validation samples:", len(X_val))
print("Prediction shape:", pred.shape)
print("Accuracy:", accuracy)

# -----------------------------
# 5. Show predictions
# -----------------------------
for i in range(10):

    print("\nSample", i + 1)

    print("Actual:   ", y_val[i])
    print("Predicted:", pred[i])

    # confidence for each prediction
    confidence = np.max(pred_prob[i], axis=-1)

    print(
        "Confidence:",
        np.round(confidence, 3)
    )