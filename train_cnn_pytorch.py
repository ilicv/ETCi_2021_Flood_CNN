#!/usr/bin/env python3
import os
import csv
import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# ==================== Configuration ====================
BATCH_SIZE = 16        # Number of samples per gradient update
NUM_EPOCHS = 50        # Maximum number of training epochs
PATIENCE = 7           # Early stopping patience (epochs without improvement)
LEARNING_RATE = 1e-3   # Initial learning rate for optimizer
IMAGE_SIZE = (128, 128)  # Resize input images to this size (H, W)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHART_PATH    = 'training_history_PyTorch.png'  # Path to save the loss/accuracy chart


class FloodDataset(Dataset):
    """
    A PyTorch Dataset that scans a directory tree for labels.csv files
    and loads corresponding VH/VV image pairs with binary labels.

    Directory structure assumption:
      root_dir/.../labels.csv
      root_dir/.../vh/<basename>_vh.png
      root_dir/.../vv/<basename>_vv.png

    Each row in labels.csv: filename.png, flood_label, water_body_label
    """
    def __init__(self, root_dir, transform=None):
        print(f"Scanning for labels in '{root_dir}'...")
        self.samples = []     # Will hold tuples: (vh_path, vv_path, [flood, water])
        self.transform = transform

        # Recursively traverse root_dir
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Prevent descending into other reduced folders
            if os.path.basename(dirpath).startswith('_reduced') and dirpath != root_dir:
                dirnames[:] = []
                continue

            # When a labels.csv is found
            if 'labels.csv' in filenames:
                print(f"Found labels.csv in: {dirpath}")
                csv_path = os.path.join(dirpath, 'labels.csv')
                vh_dir = os.path.join(dirpath, 'vh')
                vv_dir = os.path.join(dirpath, 'vv')
                # Validate that both vh/ and vv/ exist
                if not (os.path.isdir(vh_dir) and os.path.isdir(vv_dir)):
                    print(f"Warning: Missing 'vh' or 'vv' in {dirpath}, skipping.")
                    continue

                # Read CSV and build sample list
                with open(csv_path, newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    print(f"Reading {csv_path}, header: {header}")
                    for row in reader:
                        if len(row) < 3:
                            continue
                        fname, flood_lbl, water_lbl = row[0], row[1], row[2]
                        base, _ = os.path.splitext(fname)
                        vh_path = os.path.join(vh_dir, f"{base}_vh.png")
                        vv_path = os.path.join(vv_dir, f"{base}_vv.png")
                        # Only include if both channels exist
                        if os.path.isfile(vh_path) and os.path.isfile(vv_path):
                            labels = [int(flood_lbl), int(water_lbl)]
                            self.samples.append((vh_path, vv_path, labels))

        total = len(self.samples)
        if total == 0:
            raise RuntimeError(
                f"No samples found in '{root_dir}'. Ensure labels.csv and vh/vv folders are correct."
            )
        print(f"Total samples found: {total}")

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve one sample:
          - Load VH and VV images, apply transforms
          - Stack into a 2-channel tensor
          - Return (input_tensor, label_tensor)
        """
        vh_path, vv_path, labels = self.samples[idx]
        img_vh = Image.open(vh_path).convert('L')
        img_vv = Image.open(vv_path).convert('L')
        if self.transform:
            img_vh = self.transform(img_vh)
            img_vv = self.transform(img_vv)
        # Combine channels: tensor shape [2, H, W]
        x = torch.cat([img_vh, img_vv], dim=0)
        y = torch.tensor(labels, dtype=torch.float32)
        return x, y


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture:
      - 3 convolutional layers with ReLU + pooling/adaptive pooling
      - Flatten + 2-layer MLP classifier outputting 2 logits
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        """Forward pass through feature extractor and classifier."""
        x = self.features(x)
        return self.classifier(x)


def plot_metrics(history):
    """
    Plot training history:
      - Left: Loss curves
      - Right: Accuracy curves for flood & water (combined all-label accuracy)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot overall Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], label='Val Acc')
    axes[1].set_title('Overall Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    # Save chart to file
    fig.savefig(CHART_PATH)
    print(f"Training history chart saved to {CHART_PATH}")


def train():
    """
    Main training routine:
      1. Load and preprocess data
      2. Split into train/val/test
      3. Train CNN with early stopping & LR scheduling
      4. Evaluate on test set per-label and overall
      5. Plot training metrics
    """
    print(f"Starting training in '{os.getcwd()}' on {DEVICE}")

    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # resize images
        transforms.ToTensor()           # convert to [0,1] tensor
    ])

    # Load dataset
    try:
        dataset = FloodDataset(os.getcwd(), transform)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    # Load dataset

    # Split into train/val/test (60/20/20)
    total = len(dataset)
    test_size = int(0.2 * total)
    train_val_size = total - test_size
    train_val_ds, test_ds = random_split(dataset, [train_val_size, test_size])
    train_size = int(0.8 * train_val_size)
    val_size = train_val_size - train_size
    train_ds, val_ds = random_split(train_val_ds, [train_size, val_size])
    print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test samples")
    # Split into train/val/test (60/20/20)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model, loss, optimizer, scheduler
    model = SimpleCNN().to(DEVICE)
    print(model)
    criterion = nn.BCEWithLogitsLoss()  # for multi-label binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    # Model, loss, optimizer, scheduler

    # History containers
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_flood_acc': [], 'val_flood_acc': [],
        'train_water_acc': [], 'val_water_acc': []
    }
    best_loss = float('inf')
    wait = 0
    # History containers

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # ------------- Training Phase -------------
        model.train()
        running_loss = 0.0
        correct_overall = 0
        correct_flood = 0
        correct_water = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            # Per-sample overall correct if both labels match
            correct_overall += (preds == y_batch).all(dim=1).sum().item()
            # Per-label correct counts
            correct_flood += (preds[:, 0] == y_batch[:, 0]).sum().item()
            correct_water += (preds[:, 1] == y_batch[:, 1]).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_overall / len(train_loader.dataset)
        train_flood_acc = correct_flood / len(train_loader.dataset)
        train_water_acc = correct_water / len(train_loader.dataset)

        # ------------- Validation Phase -------------
        model.eval()
        val_loss_sum = 0.0
        val_correct_overall = 0
        val_correct_flood = 0
        val_correct_water = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(x_batch)
                val_loss_sum += criterion(logits, y_batch).item() * x_batch.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_correct_overall += (preds == y_batch).all(dim=1).sum().item()
                val_correct_flood += (preds[:, 0] == y_batch[:, 0]).sum().item()
                val_correct_water += (preds[:, 1] == y_batch[:, 1]).sum().item()

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_acc = val_correct_overall / len(val_loader.dataset)
        val_flood_acc = val_correct_flood / len(val_loader.dataset)
        val_water_acc = val_correct_water / len(val_loader.dataset)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_flood_acc'].append(train_flood_acc)
        history['val_flood_acc'].append(val_flood_acc)
        history['train_water_acc'].append(train_water_acc)
        history['val_water_acc'].append(val_water_acc)

        # Print epoch summary
        print(f"Epoch {epoch}/{NUM_EPOCHS} - "
              f"TLoss: {train_loss:.4f}, VLoss: {val_loss:.4f}, "
              f"TAcc: {train_acc:.4f} (F:{train_flood_acc:.4f},W:{train_water_acc:.4f}), "
              f"VAcc: {val_acc:.4f} (F:{val_flood_acc:.4f},W:{val_water_acc:.4f}), "
              f"LR: {current_lr:.6f}")

        # Early stopping & checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), 'best_cnn.pt')
            print("--> Saved new best model")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete. Best model saved as 'best_cnn.pt'.")

    # ------------- Test Evaluation -------------
    model.eval()
    test_loss_sum = 0.0
    test_correct_overall = 0
    test_correct_flood = 0
    test_correct_water = 0
    # Create DataLoader for test dataset
    _, test_ds = random_split(dataset, [train_val_size, test_size])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(x_batch)
            test_loss_sum += criterion(logits, y_batch).item() * x_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            test_correct_overall += (preds == y_batch).all(dim=1).sum().item()
            test_correct_flood += (preds[:, 0] == y_batch[:, 0]).sum().item()
            test_correct_water += (preds[:, 1] == y_batch[:, 1]).sum().item()

    test_loss = test_loss_sum / len(test_loader.dataset)
    test_acc = test_correct_overall / len(test_loader.dataset)
    test_flood_acc = test_correct_flood / len(test_loader.dataset)
    test_water_acc = test_correct_water / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} "
          f"(Flood: {test_flood_acc:.4f}, Water: {test_water_acc:.4f})")

    # Write accuracy report to text file
    report_path = 'accuracy_report_PyTorch.txt'
    with open(report_path, 'w') as f:
        f.write(f"Flood Accuracy: {test_flood_acc:.4f}\n")
        f.write(f"Water Accuracy: {test_water_acc:.4f}")
    print(f"Accuracy report saved to {report_path}")

    # Plot training metrics
    plot_metrics(history)


if __name__=='__main__':
    train()
