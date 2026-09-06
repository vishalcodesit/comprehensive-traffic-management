"""
Evaluates the trained CongestionLSTM on the held-out test split: regression
error per horizon, congestion-level classification report, and plots.

Run:  python -m src.congestion.evaluate
"""
import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, accuracy_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.congestion.dataset import load_split
from src.congestion.model import CongestionLSTM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(path=config.CONGESTION_MODEL_PATH, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CongestionLSTM(
        n_features=ckpt["n_features"], n_horizons=ckpt["n_horizons"],
        hidden_size=ckpt["hidden_size"], num_layers=ckpt["num_layers"], dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(device=device)
    vol_mean, vol_std = ckpt["vol_mean"], ckpt["vol_std"]
    horizons = ckpt["horizons"]

    test_ds = load_split(config.PROCESSED_NPZ, "test")
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    all_pred_vol, all_true_vol, all_pred_lvl, all_true_lvl = [], [], [], []
    with torch.no_grad():
        for X, y_vol, y_lvl in loader:
            X = X.to(device)
            pred_vol, pred_logits = model(X)
            pred_vol = (pred_vol.cpu().numpy() * vol_std) + vol_mean
            pred_lvl = pred_logits.argmax(dim=-1).cpu().numpy()
            all_pred_vol.append(pred_vol)
            all_true_vol.append(y_vol.numpy())
            all_pred_lvl.append(pred_lvl)
            all_true_lvl.append(y_lvl.numpy())

    pred_vol = np.concatenate(all_pred_vol)
    true_vol = np.concatenate(all_true_vol)
    pred_lvl = np.concatenate(all_pred_lvl)
    true_lvl = np.concatenate(all_true_lvl)

    metrics = {}
    for i, h in enumerate(horizons):
        mae = mean_absolute_error(true_vol[:, i], pred_vol[:, i])
        rmse = float(np.sqrt(mean_squared_error(true_vol[:, i], pred_vol[:, i])))
        mape = float(np.mean(np.abs((true_vol[:, i] - pred_vol[:, i]) /
                                     np.clip(true_vol[:, i], 100, None))) * 100)
        acc = accuracy_score(true_lvl[:, i], pred_lvl[:, i])
        f1 = f1_score(true_lvl[:, i], pred_lvl[:, i], average="macro")
        metrics[f"h{h}"] = {"MAE": mae, "RMSE": rmse, "MAPE_%": mape,
                             "level_accuracy": acc, "level_macro_f1": f1}
        print(f"Horizon +{h}h  MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.2f}%  "
              f"level_acc={acc:.3f}  macro_F1={f1:.3f}")

    with open(os.path.join(config.RESULTS_DIR, "congestion_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Classification report + confusion matrix for the nearest horizon (most actionable).
    report = classification_report(true_lvl[:, 0], pred_lvl[:, 0],
                                     target_names=config.CONGESTION_LEVELS)
    print(f"\nCongestion-level report (horizon +{horizons[0]}h):\n{report}")

    cm = confusion_matrix(true_lvl[:, 0], pred_lvl[:, 0])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CONGESTION_LEVELS, yticklabels=config.CONGESTION_LEVELS, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Congestion level confusion matrix (+{horizons[0]}h)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "congestion_confusion_matrix.png"), dpi=140)

    # Predicted vs actual volume over a slice of the test set, per horizon.
    n_show = min(400, len(true_vol))
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 3 * len(horizons)), sharex=True)
    if len(horizons) == 1:
        axes = [axes]
    for i, h in enumerate(horizons):
        axes[i].plot(true_vol[:n_show, i], label="actual", linewidth=1.2)
        axes[i].plot(pred_vol[:n_show, i], label="predicted", linewidth=1.2, alpha=0.8)
        axes[i].set_title(f"+{h}h ahead")
        axes[i].set_ylabel("veh/hr")
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("test sample index (hours)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "congestion_pred_vs_actual.png"), dpi=140)
    print("Saved evaluation plots to results/.")


if __name__ == "__main__":
    run()
