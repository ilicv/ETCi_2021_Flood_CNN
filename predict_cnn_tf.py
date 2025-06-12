#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
import tensorflow as tf
from train_cnn_tf import find_samples, load_and_preprocess, IMAGE_SIZE, BATCH_SIZE

# Update MODEL_PATH to point to your saved model file or directory.
# If you saved in HDF5 format:
#   model.save('best_cnn_tf.h5')
# If using SavedModel format:
#   model.save('best_cnn_tf')  # directory
MODEL_PATH = 'best_cnn_tf.h5'
CSV_OUTPUT = 'predictions_tf.csv'
REPORT_PATH = 'prediction_report_tf.csv'


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    print(f"Loading model from '{MODEL_PATH}'...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.\n"
              "Please ensure you've saved your TensorFlow model as an HDF5 file ('.h5') or a SavedModel directory.")
        sys.exit(1)
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Finding samples in '{root_dir}'...")
    samples = find_samples(root_dir)
    print(f"Found {len(samples)} samples for prediction.")

    # Load and preprocess images
    X, y_true = load_and_preprocess(samples)

    print("Predicting probabilities...")
    logits = model.predict(X, batch_size=BATCH_SIZE)
    probs = tf.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)

    print(f"Writing predictions to '{CSV_OUTPUT}'...")
    with open(CSV_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'flood_prob', 'water_prob', 'flood_pred', 'water_pred', 'flood_true', 'water_true'])
        for (vh, vv, _), (fp, wp), (p0, p1), (t0, t1) in zip(samples, probs, preds, y_true.astype(int)):
            fname = os.path.splitext(os.path.basename(vh))[0].replace('_vh', '')
            writer.writerow([fname, f"{fp:.4f}", f"{wp:.4f}", p0, p1, t0, t1])
    print("Predictions saved.")

    # Compute statistics
    true = y_true.astype(int)
    total = len(true)
    overall_acc = np.mean((preds == true).all(axis=1))
    flood_acc = np.mean(preds[:, 0] == true[:, 0])
    water_acc = np.mean(preds[:, 1] == true[:, 1])
    flood_pos = preds[:, 0].sum()
    water_pos = preds[:, 1].sum()

    print(f"Writing report to '{REPORT_PATH}'...")
    with open(REPORT_PATH, 'w') as f:
        f.write(f"Total samples: {total}\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")
        f.write(f"Flood accuracy: {flood_acc:.4f}\n")
        f.write(f"Water accuracy: {water_acc:.4f}\n")
        f.write(f"Predicted flood positives: {flood_pos}\n")
        f.write(f"Predicted water positives: {water_pos}\n\n")
        f.write("filename,flood_pred,flood_true,water_pred,water_true\n")
        for (vh, vv, _), pred, true_vals in zip(samples, preds, true):
            fname = os.path.splitext(os.path.basename(vh))[0].replace('_vh', '')
            p0, p1 = pred
            t0, t1 = true_vals
            f.write(f"{fname},{p0},{t0},{p1},{t1}\n")
    print("Report saved.")
    print("Done.")


if __name__ == '__main__':
    main()
