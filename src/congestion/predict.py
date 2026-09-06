"""
Inference-time wrapper used by the integration layer / dashboard: loads the
trained model once and turns a window of recent traffic data into a
multi-horizon volume forecast + congestion-level forecast.

Run directly for a quick smoke test:  python -m src.congestion.predict
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.congestion.model import CongestionLSTM


class CongestionPredictor:
    def __init__(self, model_path=config.CONGESTION_MODEL_PATH, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = CongestionLSTM(
            n_features=ckpt["n_features"], n_horizons=ckpt["n_horizons"],
            hidden_size=ckpt["hidden_size"], num_layers=ckpt["num_layers"], dropout=ckpt["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.vol_mean = ckpt["vol_mean"]
        self.vol_std = ckpt["vol_std"]
        self.horizons = list(ckpt["horizons"])

    @torch.no_grad()
    def predict(self, window: np.ndarray):
        """window: (lookback, n_features) already-scaled feature array (see preprocess.py)."""
        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(self.device)
        pred_vol, pred_logits = self.model(x)
        pred_vol = (pred_vol.cpu().numpy()[0] * self.vol_std) + self.vol_mean
        pred_level = pred_logits.argmax(dim=-1).cpu().numpy()[0]
        probs = torch.softmax(pred_logits, dim=-1).cpu().numpy()[0]
        return {
            "horizons_hours": self.horizons,
            "predicted_volume": pred_vol.tolist(),
            "predicted_level": [config.CONGESTION_LEVELS[i] for i in pred_level],
            "level_probabilities": probs.tolist(),
        }


def _demo():
    """Runs a prediction on the last window of the cached test split, purely to
    verify the trained model loads and produces sane output end to end."""
    data = np.load(config.PROCESSED_NPZ, allow_pickle=True)
    X_test = data["X_test"]
    predictor = CongestionPredictor()
    result = predictor.predict(X_test[-1])
    print("Sample forecast from the most recent test window:")
    for h, v, lvl in zip(result["horizons_hours"], result["predicted_volume"], result["predicted_level"]):
        print(f"  +{h}h -> {v:.0f} veh/hr  [{lvl}]")


if __name__ == "__main__":
    _demo()
