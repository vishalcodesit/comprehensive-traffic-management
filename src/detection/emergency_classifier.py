"""
Emergency-vehicle classification.

We do not have a licensed, labeled dataset of ambulances/fire trucks/police
cars bundled with this project (see docs/report.md, Research Gap notes on
data availability), so this module ships two complementary signals that
work out of the box on *any* video, plus a pluggable CNN head for when a
labeled dataset is added:

1. Siren light-bar flicker heuristic (`_flicker_score`): emergency vehicles
   in most jurisdictions carry a roof-mounted light bar that strobes
   red/blue/amber at several Hz. We crop the top band of each tracked
   vehicle's bounding box, measure the red/blue pixel ratio in HSV space
   every frame, and look at how much that ratio oscillates over a short
   rolling window (a normal vehicle's roof color is ~static; a strobing
   light bar is not). This needs no training data and works on live video.

2. High-visibility livery heuristic (`_marking_score`): ambulances/fire
   trucks are disproportionately white/red/orange with high edge density
   (text, stripes, reflective chevrons) compared to ordinary traffic.

3. `EmergencyCNN` (optional): a MobileNetV3-Small transfer-learning head
   for emergency vs. normal vehicle crops. If a checkpoint exists at
   config.MODELS_DIR/emergency_cnn.pt it is loaded and ensembled with the
   heuristics; otherwise the module falls back to heuristics only. See
   train_emergency_cnn.py for how to fine-tune it once a labeled dataset
   (e.g. a Kaggle "emergency vehicle identification" set) is available.

Per-track evidence is accumulated over several frames (`history`) before a
vehicle is flagged, since a single frame is not enough to see a strobe.
"""
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

RED_LOWER1, RED_UPPER1 = (0, 90, 90), (10, 255, 255)
RED_LOWER2, RED_UPPER2 = (170, 90, 90), (180, 255, 255)
BLUE_LOWER, BLUE_UPPER = (100, 90, 90), (130, 255, 255)


def _color_ratios(hsv_crop):
    total = hsv_crop.shape[0] * hsv_crop.shape[1]
    if total == 0:
        return 0.0, 0.0
    red_mask = cv2.inRange(hsv_crop, RED_LOWER1, RED_UPPER1) | cv2.inRange(hsv_crop, RED_LOWER2, RED_UPPER2)
    blue_mask = cv2.inRange(hsv_crop, BLUE_LOWER, BLUE_UPPER)
    return float(np.count_nonzero(red_mask)) / total, float(np.count_nonzero(blue_mask)) / total


class EmergencyCNN:
    """Lazy-loaded transfer-learning classifier; no-op if no checkpoint exists."""

    def __init__(self, ckpt_path=None):
        ckpt_path = ckpt_path or os.path.join(config.MODELS_DIR, "emergency_cnn.pt")
        self.available = os.path.exists(ckpt_path)
        self.model = None
        if self.available:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            net = models.mobilenet_v3_small(weights=None)
            net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 2)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            if "model_state" in ckpt:
                state, self.emergency_idx = ckpt["model_state"], ckpt["emergency_idx"]
            else:
                # Legacy bare state_dict: ImageFolder's alphabetical order
                # puts 'emergency' at index 0.
                state, self.emergency_idx = ckpt, 0
            net.load_state_dict(state)
            net.eval().to(self.device)
            self.model = net
            self.transform = T.Compose([
                T.ToPILImage(), T.Resize((160, 160)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def score(self, bgr_crop) -> float:
        """Returns P(emergency) in [0, 1], or 0.5 (neutral) if unavailable."""
        if not self.available or bgr_crop.size == 0:
            return 0.5
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(x)
            prob = self.torch.softmax(logits, dim=-1)[0, self.emergency_idx].item()
        return prob


class EmergencyVehicleDetector:
    def __init__(self, history_len=15, min_frames=6, flicker_weight=0.5,
                 marking_weight=0.2, cnn_weight=0.3, threshold=0.55):
        self.history = defaultdict(lambda: deque(maxlen=history_len))
        self.min_frames = min_frames
        self.weights = (flicker_weight, marking_weight, cnn_weight)
        self.threshold = threshold
        self.cnn = EmergencyCNN()

    def _crop(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        full = frame[y1:y2, x1:x2]
        roof_h = max(1, int((y2 - y1) * 0.35))
        roof = frame[y1:y1 + roof_h, x1:x2]
        return full, roof

    def _marking_score(self, full_crop):
        if full_crop is None or full_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(full_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = np.count_nonzero(edges) / edges.size
        hsv = cv2.cvtColor(full_crop, cv2.COLOR_BGR2HSV)
        red_ratio, _ = _color_ratios(hsv)
        white_ratio = float(np.count_nonzero(
            (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 170)
        )) / hsv.shape[0] / hsv.shape[1]
        return float(np.clip(edge_density * 2.0 + red_ratio * 1.5 + white_ratio * 0.8, 0, 1))

    def _flicker_score(self, track_id, roof_crop):
        if roof_crop is None or roof_crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roof_crop, cv2.COLOR_BGR2HSV)
        red_ratio, blue_ratio = _color_ratios(hsv)
        signal = red_ratio + blue_ratio
        hist = self.history[track_id]
        hist.append(signal)
        if len(hist) < self.min_frames:
            return 0.0
        arr = np.array(hist)
        # Normalized variance: a strobing light bar oscillates strongly frame
        # to frame; a static-colored roof stays near-constant.
        variance = float(np.var(arr))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(arr - arr.mean()))) > 0)
        flicker = np.clip(variance * 40 + zero_crossings / len(hist), 0, 1)
        return float(flicker)

    def evaluate(self, frame, track):
        """track: {"track_id", "bbox", "class_name"}. Returns augmented dict with
        'emergency_score' in [0,1] and 'is_emergency' bool (False until enough
        frames of evidence have been collected for that track)."""
        full_crop, roof_crop = self._crop(frame, track["bbox"])
        flicker = self._flicker_score(track["track_id"], roof_crop)
        marking = self._marking_score(full_crop)
        cnn_score = self.cnn.score(full_crop) if full_crop is not None else 0.5

        w_flicker, w_marking, w_cnn = self.weights
        score = w_flicker * flicker + w_marking * marking + w_cnn * cnn_score
        enough_evidence = len(self.history[track["track_id"]]) >= self.min_frames

        out = dict(track)
        out["emergency_score"] = round(float(score), 3)
        out["is_emergency"] = bool(enough_evidence and score >= self.threshold)
        return out
