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
                 marking_weight=0.2, cnn_weight=0.3, threshold=0.55,
                 min_flash=0.25, full_flicker_swings=4, min_flicker=0.75,
                 roof_extend=0.3, cnn_confident=0.9):
        self.history = defaultdict(lambda: deque(maxlen=history_len))
        self.min_frames = min_frames
        self.weights = (flicker_weight, marking_weight, cnn_weight)
        self.threshold = threshold
        self.min_flash = min_flash                      # min share of the roof region a lit light must cover
        self.full_flicker_swings = full_flicker_swings  # on<->off swings in the window for flicker = 1.0
        self.roof_extend = roof_extend                  # roof band reaches this fraction of box height above it
        self.min_flicker = min_flicker
        self.cnn_confident = cnn_confident
        self.cnn = EmergencyCNN()

    def _crop(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        full = frame[y1:y2, x1:x2]
        # Light bars sit on top of the roof and detectors often leave them
        # out of the vehicle box, so the roof band extends above the box.
        roof_h = max(1, int((y2 - y1) * 0.35))
        roof_top = max(0, y1 - int((y2 - y1) * self.roof_extend))
        roof = frame[roof_top:y1 + roof_h, x1:x2]
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

    def _swings(self, series):
        """Counts on<->off flashes of one colour's share of the roof region.
        A strobe repeatedly drops to near zero and comes back; a red car's
        roof only drifts up and down while staying lit, so 'on' and 'off'
        are judged relative to the window's own peak (with hysteresis), and
        a colour that never covers min_flash of the region isn't a light."""
        peak = series.max()
        if peak < self.min_flash:
            return 0
        on, off = peak * 0.6, peak * 0.25
        state, swings = 0, 0
        for v in series:
            s = 1 if v >= on else (-1 if v <= off else 0)
            if s and s != state:
                swings += state != 0
                state = s
        return swings

    def _flicker_score(self, track_id, roof_crop):
        if roof_crop is None or roof_crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roof_crop, cv2.COLOR_BGR2HSV)
        hist = self.history[track_id]
        hist.append(_color_ratios(hsv))  # (red_ratio, blue_ratio)
        if len(hist) < self.min_frames:
            return 0.0
        arr = np.array(hist)
        # A strobing light bar repeatedly swings its red and/or blue share of
        # the roof far up and down (alternating red/blue, or one colour
        # flashing on/off); a static roof -- even a red one -- does not.
        swings = max(self._swings(arr[:, 0]), self._swings(arr[:, 1]))
        return float(np.clip(swings / self.full_flicker_swings, 0, 1))

    def evaluate(self, frame, track):
        """track: {"track_id", "bbox", "class_name"}. Returns augmented dict with
        'emergency_score' in [0,1] and 'is_emergency' bool (False until enough
        frames of evidence have been collected for that track)."""
        full_crop, roof_crop = self._crop(frame, track["bbox"])
        flicker = self._flicker_score(track["track_id"], roof_crop)
        marking = self._marking_score(full_crop)

        w_flicker, w_marking, w_cnn = self.weights
        if self.cnn.available:
            cnn_score = self.cnn.score(full_crop) if full_crop is not None else 0.0
            score = w_flicker * flicker + w_marking * marking + w_cnn * cnn_score
            cnn_confident = cnn_score >= self.cnn_confident
        else:
            # No CNN: renormalize over the heuristics instead of crediting
            # every vehicle a free "neutral" CNN score.
            score = (w_flicker * flicker + w_marking * marking) / (w_flicker + w_marking)
            cnn_confident = False
        enough_evidence = len(self.history[track["track_id"]]) >= self.min_frames
        # Livery alone matches every white van and red car, so a vehicle is
        # only flagged with flashing-light evidence (or a confident CNN).
        has_primary_evidence = flicker >= self.min_flicker or cnn_confident

        out = dict(track)
        out["emergency_score"] = round(float(score), 3)
        out["is_emergency"] = bool(enough_evidence and has_primary_evidence and score >= self.threshold)
        return out
