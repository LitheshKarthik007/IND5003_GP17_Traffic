import numpy as np
import tensorflow as tf

data = np.load(
    "scripts/artifacts/phase_cluster/phase_cluster_causeway_dataset.npz"
)

X = data["X_val"]
y = data["y_val"]

model = tf.keras.models.load_model(
    "scripts/artifacts/model_cluster/causeway_lstm_best.h5",
    compile=False
)

pred_prob = model.predict(X, verbose=0)
pred = np.argmax(pred_prob, axis=-1) + 1

# Find samples where actual future traffic changes
transition_count = 0

print("\n========== TRANSITION CASES ==========\n")

for i in range(len(y)):

    actual = y[i]
    predicted = pred[i]

    # Check if actual future contains a change
    if len(set(actual)) > 1:

        transition_count += 1

        accuracy = np.mean(actual == predicted)

        print("Sample:", i)
        print("Actual:   ", actual)
        print("Predicted:", predicted)
        print("Accuracy: ", round(accuracy, 3))
        print()

        if transition_count >= 20:
            break

print("Transition samples shown:", transition_count)