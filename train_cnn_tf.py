#!/usr/bin/env python3
import os
import csv
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# ==================== Configuration ====================
BATCH_SIZE    = 16        # Number of samples per gradient update
IMAGE_SIZE    = (128, 128)  # (height, width) to resize input images
NUM_EPOCHS    = 50        # Maximum number of epochs to train
PATIENCE      = 7         # Epochs with no improvement before stopping
LEARNING_RATE = 1e-3      # Initial learning rate for optimizer
ROOT_DIR      = os.getcwd()  # Working directory where samples reside
CHART_PATH    = 'training_history_TF.png'  # Path to save the loss/accuracy chart


def find_samples(root_dir):
    """
    Recursively find all 'labels.csv' files under root_dir.
    Each labels.csv contains rows: filename.png, flood_label, water_body_label.
    Returns a list of tuples: (vh_image_path, vv_image_path, [flood_label, water_label]).
    Skips nested '_reduced' subfolders except root.
    """
    samples = []  # will collect all sample tuples

    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip entering other reduced dataset folders
        if os.path.basename(dirpath).startswith('_reduced') and dirpath != root_dir:
            dirnames[:] = []
            continue

        # Look for labels.csv in this folder
        if 'labels.csv' in filenames:
            # Paths to polarization image subdirectories
            vh_dir = os.path.join(dirpath, 'vh')
            vv_dir = os.path.join(dirpath, 'vv')
            # Only proceed if both vh and vv exist
            if not (os.path.isdir(vh_dir) and os.path.isdir(vv_dir)):
                continue

            # Open and read the CSV file
            csv_path = os.path.join(dirpath, 'labels.csv')
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header row
                for fname, flood_lbl, water_lbl in reader:
                    base, _ = os.path.splitext(fname)
                    vh_path = os.path.join(vh_dir, f"{base}_vh.png")
                    vv_path = os.path.join(vv_dir, f"{base}_vv.png")
                    # Only add if both image files exist
                    if os.path.isfile(vh_path) and os.path.isfile(vv_path):
                        samples.append((vh_path, vv_path,
                                        [int(flood_lbl), int(water_lbl)]))

    # Ensure we found samples
    if not samples:
        raise RuntimeError(f"No samples found in {root_dir}. Check labels.csv and vh/vv folders.")
    return samples


def load_and_preprocess(samples):
    """
    Load image data and labels from the sample list.
    - Reads each VH/VV PNG, converts to grayscale, resizes.
    - Stacks into a 2-channel array.
    - Normalizes pixel values to [0,1].
    Returns:
      X: numpy array of shape (N, H, W, 2)
      y: numpy array of shape (N, 2)
    """
    images = []  # to collect image arrays
    labels = []  # to collect label vectors

    for vh_path, vv_path, lbl in samples:
        # Load and resize each polarization channel
        vh = np.array(Image.open(vh_path).convert('L').resize(IMAGE_SIZE))
        vv = np.array(Image.open(vv_path).convert('L').resize(IMAGE_SIZE))
        # Stack channels to shape (H, W, 2)
        img = np.stack([vh, vv], axis=-1)
        images.append(img)
        labels.append(lbl)

    # Convert to arrays and normalize
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.float32)
    return X, y


def build_model(input_shape):
    """
    Build a CNN model with the following architecture:
      Input -> [Conv(16)-ReLU-Pool] -> [Conv(32)-ReLU-Pool]
            -> [Conv(64)-ReLU-GlobalAvgPool] -> Dense(32)-ReLU -> Dense(2)
    Outputs raw logits for two binary labels.
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(2, activation=None)(x)  # two logits
    model = models.Model(inputs=inp, outputs=out)
    return model


def plot_history(hist):
    """
    Plot training and validation curves:
      - Left: Loss
      - Right: Combined binary accuracy
    """
    epochs = range(1, len(hist.history['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss
    axes[0].plot(epochs, hist.history['loss'], label='Train Loss')
    axes[0].plot(epochs, hist.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Binary Crossentropy')
    axes[0].legend(); axes[0].grid(True)

    # Plot accuracy
    axes[1].plot(epochs, hist.history['binary_accuracy'], label='Train Acc')
    axes[1].plot(epochs, hist.history['val_binary_accuracy'], label='Val Acc')
    axes[1].set_title('Overall Binary Accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout(); plt.show()
    # Save chart to file
    fig.savefig(CHART_PATH)
    print(f"Training history chart saved to {CHART_PATH}")


def main():
    """
    Orchestrate the full training pipeline:
      1. Sample discovery
      2. Data loading/preprocessing
      3. Train/val/test split
      4. Model construction
      5. Training with callbacks
      6. Evaluation on test set (combined & per-variable)
      7. Plotting results
    """
    print(f"Gathering samples from '{ROOT_DIR}'...")
    samples = find_samples(ROOT_DIR)
    print(f"Found {len(samples)} total samples.")

    # Loading and preprocessing
    print("Loading and preprocessing images...")
    X, y = load_and_preprocess(samples)
    # Loading and preprocessing

    # Split dataset
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    n_test = int(0.2 * N)
    n_val = int(0.2 * N)
    n_train = N - n_val - n_test
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    print(f"Data split: {n_train} train, {n_val} val, {n_test} test samples.")
    # Split dataset

    # Build and compile model
    model = build_model(input_shape=IMAGE_SIZE + (2,))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')]
    )
    model.summary()
    # Build and compile model

    # Set up callbacks
    ckpt = callbacks.ModelCheckpoint(
        'best_cnn_tf.h5', save_best_only=True,
        monitor='val_loss', verbose=1
    )
    es = callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE,
        restore_best_weights=True, verbose=1
    )
    rlr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, verbose=1
    )

    # Train model
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[ckpt, es, rlr],
        verbose=2
    )
    # Train model

    # Evaluate on test set (combined)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}, Overall Test Acc: {acc:.4f}")
    # Evaluate on test set (combined)

    # Per-variable accuracy
    logits = model.predict(X_test)
    probs = tf.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    true = y_test.astype(int)
    flood_acc = np.mean(preds[:, 0] == true[:, 0])
    water_acc = np.mean(preds[:, 1] == true[:, 1])
    print(f"Flood Accuracy: {flood_acc:.4f}")
    print(f"Water Accuracy: {water_acc:.4f}")
    # Write accuracy report to text file
    report_path = 'accuracy_report_tf.txt'
    with open(report_path, 'w') as f:
        f.write(f"Flood Accuracy: {flood_acc:.4f}\n")
        f.write(f"Water Accuracy: {water_acc:.4f}")
    print(f"Accuracy report saved to {report_path}")


    # Plot training history
    plot_history(hist)


if __name__ == '__main__':
    main()
