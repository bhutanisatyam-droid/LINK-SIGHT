"""PyTorch Training & ONNX Exporter for MicroGRU-Coast (Core 2 State AI).

Trains a 2-Layer Gated Recurrent Unit (~8.9K parameters) to predict:
- Non-linear UAV trajectory during a 200ms optical occlusion window

Can be run locally on CPU (~15-20 mins) or Colab.
Exports trained weights to `models/microgru_coast.onnx`.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MicroGRUCoaster(nn.Module):
    """2-Layer GRU for blind trajectory coasting under optical occlusion."""

    def __init__(self, input_size: int = 4, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2) # Predicts next (dx, dy)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :] # Take state at last time step
        return self.head(last_hidden)


def train():
    print("=" * 60)
    print("LinkSight: MicroGRU-Coast Training Pipeline (PyTorch -> ONNX)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    data_path = "datasets/dataset_trajectory.npz"
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at '{data_path}'!")
        print("Please run `python tools/generate_trajectory_data.py` first.")
        return

    print(f"[*] Loading trajectory dataset from '{data_path}'...")
    data = np.load(data_path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)

    total_samples = len(X)
    train_size = int(0.85 * total_samples)
    val_size = total_samples - train_size

    full_ds = TensorDataset(X, Y)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    print(f"[*] Training Sequences: {train_size:,} | Validation Sequences: {val_size:,}")

    model = MicroGRUCoaster().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Trainable Parameters: {param_count:,} (Ultra-lightweight GRU)")

    criterion = nn.SmoothL1Loss() # Huber loss
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    epochs = 20
    print("\n[*] Starting Training Loop (20 Epochs)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)

            optimizer.zero_grad()
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(b_x)

        train_loss /= train_size

        # Validation
        model.eval()
        val_loss = 0.0
        total_drift_error = 0.0

        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                preds = model(b_x)
                loss = criterion(preds, b_y)
                val_loss += loss.item() * len(b_x)

                # Euclidean pixel drift error: sqrt( (pred_dx - true_dx)^2 + (pred_dy - true_dy)^2 )
                euclid_err = torch.sqrt(torch.sum((preds - b_y) ** 2, dim=1))
                total_drift_error += torch.sum(euclid_err).item()

        val_loss /= val_size
        mean_drift_px = total_drift_error / val_size
        scheduler.step(val_loss)

        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | 1-Step Drift Error: {mean_drift_px:.3f} px")

    elapsed = time.time() - start_time
    print(f"\n[*] Training Complete in {elapsed:.1f}s!")

    # Export to standardized ONNX binary
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/microgru_coast.onnx"
    model.eval()
    model.to("cpu")

    dummy_input = torch.randn(1, 30, 4, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["state_history"],
        output_names=["predicted_delta"],
        dynamic_axes={"state_history": {0: "batch"}, "predicted_delta": {0: "batch"}}
    )

    file_size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"[SUCCESS] Exported ONNX model to: {onnx_path}")
    print(f"[*] ONNX Binary File Size: {file_size_kb:.2f} KB")
    print("=" * 60)


if __name__ == "__main__":
    train()
