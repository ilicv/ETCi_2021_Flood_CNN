#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from train_cnn_pytorch import FloodDataset, SimpleCNN, IMAGE_SIZE, DEVICE

MODEL_PATH = 'best_cnn.pt'
CSV_OUTPUT = 'predictions_torch.csv'
REPORT_PATH = 'prediction_report_torch.csv'


def main():
    # Determine root directory for samples
    root_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    print(f"Loading model from '{MODEL_PATH}'...")
    # Load model architecture and weights
    model = SimpleCNN().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Prepare image transform
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    # Scan for samples
    print(f"Scanning for samples in '{root_dir}'...")
    dataset = FloodDataset(root_dir, transform)
    samples = dataset.samples
    print(f"Found {len(samples)} samples for prediction.")

    filenames = []
    probs_list = []
    preds_list = []
    true_list = []

    # Prediction loop
    with torch.no_grad():
        for vh_path, vv_path, label in samples:
            # Load and preprocess images
            img_vh = Image.open(vh_path).convert('L')
            img_vv = Image.open(vv_path).convert('L')
            img_vh = transform(img_vh)
            img_vv = transform(img_vv)
            x = torch.cat([img_vh, img_vv], dim=0).unsqueeze(0).to(DEVICE)

            # Forward pass
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            pred = (probs > 0.5).astype(int)
            true = np.array(label, dtype=int)

            # Extract basename
            fname = os.path.splitext(os.path.basename(vh_path))[0].replace('_vh', '')

            filenames.append(fname)
            probs_list.append(probs)
            preds_list.append(pred)
            true_list.append(true)

    # Write predictions CSV
    print(f"Writing predictions to '{CSV_OUTPUT}'...")
    with open(CSV_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'flood_prob', 'water_prob', 'flood_pred', 'water_pred', 'flood_true', 'water_true'])
        for fname, probs, pred, true in zip(filenames, probs_list, preds_list, true_list):
            writer.writerow([fname, f"{probs[0]:.4f}", f"{probs[1]:.4f}", pred[0], pred[1], true[0], true[1]])
    print("Predictions saved.")

    # Compute statistics
    preds_arr = np.array(preds_list)
    true_arr = np.array(true_list)
    total = len(true_arr)
    overall_acc = np.mean(np.all(preds_arr == true_arr, axis=1))
    flood_acc = np.mean(preds_arr[:, 0] == true_arr[:, 0])
    water_acc = np.mean(preds_arr[:, 1] == true_arr[:, 1])
    flood_pos = preds_arr[:, 0].sum()
    water_pos = preds_arr[:, 1].sum()

    # Write report
    print(f"Writing report to '{REPORT_PATH}'...")
    with open(REPORT_PATH, 'w') as f:
        f.write(f"Total samples: {total}\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")
        f.write(f"Flood accuracy: {flood_acc:.4f}\n")
        f.write(f"Water accuracy: {water_acc:.4f}\n")
        f.write(f"Predicted flood positives: {flood_pos}\n")
        f.write(f"Predicted water positives: {water_pos}\n\n")
        f.write("Sample-wise predictions (predicted vs expected):\n")
        f.write("filename,flood_pred,flood_true,water_pred,water_true\n")
        for fname, pred, true in zip(filenames, preds_list, true_list):
            f.write(f"{fname},{pred[0]},{true[0]},{pred[1]},{true[1]}\n")
    print("Report saved.")
    print("Done.")


if __name__ == '__main__':
    main()
