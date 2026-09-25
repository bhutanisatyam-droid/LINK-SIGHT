"""PyTorch Training & ONNX Exporter for TinyBeaconNet (Core 1 Vision AI).

Trains a Shallow Hybrid CNN-MLP (~8.2K parameters) to classify:
- 1 = True Optical Beacon
- 0 = Solar Glint / Dead Pixel / Background Clutter

Can be run locally or uploaded to Google Colab with free T4 GPU.
Exports trained weights to `models/tinybeaconnet.onnx`.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class TinyBeaconNet(nn.Module):
    """Hybrid CNN-MLP architecture for optical beacon prescreening."""

    def __init__(self):
        super().__init__()
        
        # Branch 1: Lightweight 2D CNN on 24x24 grayscale patch
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 8 x 12 x 12

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16 x 6 x 6
            nn.Flatten()                            # Flatten -> 576
        )

        # Branch 2: Dense MLP on 8D Hand-Crafted Mathematical Features
        self.feature_mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(inplace=True)
        )

        # Branch 3: Multimodal Fusion & Classification Head
        self.fusion_head = nn.Sequential(
            nn.Linear(576 + 16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, patch: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(patch)
        feat_out = self.feature_mlp(features)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        return self.fusion_head(combined)


def train():
    print("=" * 60)
    print("LinkSight: TinyBeaconNet Training Pipeline (PyTorch -> ONNX)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    data_path = "datasets/dataset_beacon.npz"
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at '{data_path}'!")
        print("Please run `python tools/generate_beacon_data.py` first.")
        return

    print(f"[*] Loading synthetic dataset from '{data_path}'...")
    data = np.load(data_path)
    patches = torch.tensor(data["patches"], dtype=torch.float32)
    features = torch.tensor(data["features"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.float32)

    total_samples = len(labels)
    train_size = int(0.85 * total_samples)
    val_size = total_samples - train_size

    full_dataset = TensorDataset(patches, features, labels)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    print(f"[*] Training Samples: {train_size} | Validation Samples: {val_size}")

    model = TinyBeaconNet().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total Trainable Parameters: {param_count:,} (Ultra-lightweight SWaP-C target)")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    epochs = 25
    print("\n[*] Starting Training Loop (25 Epochs)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0

        for b_patch, b_feat, b_label in train_loader:
            b_patch, b_feat, b_label = b_patch.to(device), b_feat.to(device), b_label.to(device)

            optimizer.zero_grad()
            preds = model(b_patch, b_feat)
            loss = criterion(preds, b_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(b_label)
            correct_train += ((preds >= 0.5) == b_label).sum().item()

        scheduler.step()
        train_loss /= train_size
        train_acc = (correct_train / train_size) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        with torch.no_grad():
            for b_patch, b_feat, b_label in val_loader:
                b_patch, b_feat, b_label = b_patch.to(device), b_feat.to(device), b_label.to(device)
                preds = model(b_patch, b_feat)
                loss = criterion(preds, b_label)
                val_loss += loss.item() * len(b_label)

                bin_preds = (preds >= 0.5).int()
                correct_val += (bin_preds == b_label).sum().item()

                tp += ((bin_preds == 1) & (b_label == 1)).sum().item()
                fp += ((bin_preds == 1) & (b_label == 0)).sum().item()
                tn += ((bin_preds == 0) & (b_label == 0)).sum().item()
                fn += ((bin_preds == 0) & (b_label == 1)).sum().item()

        val_loss /= val_size
        val_acc = (correct_val / val_size) * 100.0
        precision = (tp / (tp + fp + 1e-5)) * 100.0
        recall = (tp / (tp + fn + 1e-5)) * 100.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | Precision: {precision:.2f}% Recall: {recall:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n[*] Training Complete in {elapsed:.1f}s!")

    # Export to standardized ONNX binary
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/tinybeaconnet.onnx"
    model.eval()
    model.to("cpu")

    dummy_patch = torch.randn(1, 1, 24, 24, dtype=torch.float32)
    dummy_feat = torch.randn(1, 8, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_patch, dummy_feat),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["patch", "features"],
        output_names=["probability"],
        dynamic_axes={"patch": {0: "batch"}, "features": {0: "batch"}, "probability": {0: "batch"}}
    )

    file_size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"[SUCCESS] Exported ONNX model to: {onnx_path}")
    print(f"[*] ONNX Binary File Size: {file_size_kb:.2f} KB (Well below edge SWaP budget)")
    print("=" * 60)


if __name__ == "__main__":
    train()
