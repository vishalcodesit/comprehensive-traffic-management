"""
Trains the CongestionLSTM multi-horizon forecaster on the preprocessed windows.

Run:  python -m src.congestion.train
"""
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.congestion.dataset import load_split
from src.congestion.model import CongestionLSTM

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def run(epochs=config.EPOCHS, patience=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = load_split(config.PROCESSED_NPZ, "train")
    val_ds = load_split(config.PROCESSED_NPZ, "val")
    n_features = train_ds.X.shape[-1]
    n_horizons = train_ds.y_volume.shape[-1]
    print(f"Train windows: {len(train_ds)}  Val windows: {len(val_ds)}  "
          f"features: {n_features}  horizons: {n_horizons}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = CongestionLSTM(
        n_features=n_features, n_horizons=n_horizons,
        hidden_size=config.LSTM_HIDDEN_SIZE, num_layers=config.LSTM_NUM_LAYERS,
        dropout=config.LSTM_DROPOUT,
    ).to(device)

    # Normalize volume targets for a well-scaled regression loss; the raw
    # scale (~5000 veh/hr) would otherwise dominate the classification loss.
    y_train_all = train_ds.y_volume.numpy()
    vol_mean, vol_std = y_train_all.mean(), y_train_all.std()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val = float("inf")
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(config.CONGESTION_MODEL_PATH), exist_ok=True)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, y_vol, y_lvl in train_loader:
            X, y_vol, y_lvl = X.to(device), y_vol.to(device), y_lvl.to(device)
            y_vol_norm = (y_vol - vol_mean) / vol_std

            optimizer.zero_grad()
            pred_vol, pred_logits = model(X)
            loss_reg = mse(pred_vol, y_vol_norm)
            loss_cls = ce(pred_logits.reshape(-1, pred_logits.shape[-1]), y_lvl.reshape(-1))
            loss = loss_reg + 0.3 * loss_cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, abs_errors = [], []
        with torch.no_grad():
            for X, y_vol, y_lvl in val_loader:
                X, y_vol, y_lvl = X.to(device), y_vol.to(device), y_lvl.to(device)
                y_vol_norm = (y_vol - vol_mean) / vol_std
                pred_vol, pred_logits = model(X)
                loss_reg = mse(pred_vol, y_vol_norm)
                loss_cls = ce(pred_logits.reshape(-1, pred_logits.shape[-1]), y_lvl.reshape(-1))
                val_losses.append((loss_reg + 0.3 * loss_cls).item())
                pred_vol_real = pred_vol * vol_std + vol_mean
                abs_errors.append(torch.abs(pred_vol_real - y_vol).cpu().numpy())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_mae = float(np.mean(np.concatenate(abs_errors)))
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_MAE={val_mae:.1f} veh/hr")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "n_features": n_features,
                "n_horizons": n_horizons,
                "vol_mean": vol_mean,
                "vol_std": vol_std,
                "hidden_size": config.LSTM_HIDDEN_SIZE,
                "num_layers": config.LSTM_NUM_LAYERS,
                "dropout": config.LSTM_DROPOUT,
                "horizons": config.HORIZONS,
            }, config.CONGESTION_MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s. Best val_loss={best_val:.4f}. "
          f"Model saved to {config.CONGESTION_MODEL_PATH}")

    with open(os.path.join(config.RESULTS_DIR, "congestion_train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if HAS_PLT:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(history["train_loss"], label="train")
        ax[0].plot(history["val_loss"], label="val")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(history["val_mae"], color="tab:orange")
        ax[1].set_title("Validation MAE (veh/hr)"); ax[1].set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(os.path.join(config.RESULTS_DIR, "congestion_training_curves.png"), dpi=140)
        print("Saved training curves plot.")


if __name__ == "__main__":
    run()
