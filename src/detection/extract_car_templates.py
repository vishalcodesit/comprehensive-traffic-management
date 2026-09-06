"""
Extracts a handful of real, high-confidence vehicle crops from the real
sample footage (downloaded by download_sample_video.py) to use as photo-
realistic sprites in synthetic_video.py. See that module's docstring for why:
a COCO-pretrained YOLOv8 does not detect flat-shaded procedural renders, so
the synthetic demo clip composites real vehicle photos instead.

Produces:
  data/samples/car_template.png      -- one clean, mostly-white, side-view
                                         car (used as the ambulance base so
                                         the drawn light bar reads clearly).
  data/samples/car_template_0..N.png -- a handful of other vehicles, picked
                                         to be visually distinct from each
                                         other (greedy max mean-color distance).

Run:  python -m src.detection.extract_car_templates
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.detection.download_sample_video import download as download_sample_video


def _collect_candidates(video_path, detector, max_frames=900, stride=3,
                         min_w=50, max_w=220, min_h=40, max_h=180, min_conf=0.55):
    cap = cv2.VideoCapture(video_path)
    candidates = []
    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        for d in detector.detect(frame):
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            if not (min_w < w < max_w and min_h < h < max_h and d["conf"] > min_conf):
                continue
            pad = 4
            xi1, yi1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
            xi2, yi2 = int(x2 + pad), int(y2 + pad)
            crop = frame[yi1:yi2, xi1:xi2]
            if crop.size == 0:
                continue
            mean_color = crop.reshape(-1, 3).mean(axis=0)
            candidates.append({"area": w * h, "conf": d["conf"], "color": mean_color, "crop": crop})
    cap.release()
    return candidates


def _pick_diverse(candidates, n):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: -(c["area"] * c["conf"]))
    picked = [candidates[0]]
    for _ in range(n - 1):
        best, best_dist = None, -1.0
        for c in candidates:
            dist = min(np.linalg.norm(c["color"] - p["color"]) for p in picked)
            if dist > best_dist:
                best_dist, best = dist, c
        if best is None:
            break
        picked.append(best)
    return picked


def _pick_cleanest_white(candidates):
    """Prefers a mostly-white/light-colored, high-confidence crop for the
    ambulance base, since it's what the drawn light-bar overlay is drawn onto."""
    if not candidates:
        return None
    def whiteness(c):
        b, g, r = c["color"]
        return -(abs(b - g) + abs(g - r) + abs(b - r)) + (b + g + r) / 3
    return max(candidates, key=lambda c: whiteness(c) * c["conf"])


def run(n_traffic_templates=3, force=False):
    ambulance_path = os.path.join(config.SAMPLES_DATA_DIR, "car_template.png")
    traffic_paths = [os.path.join(config.SAMPLES_DATA_DIR, f"car_template_{i}.png")
                      for i in range(n_traffic_templates)]
    if not force and os.path.exists(ambulance_path) and all(os.path.exists(p) for p in traffic_paths):
        print("Templates already extracted.")
        return

    if not os.path.exists(config.SAMPLE_VIDEO_PATH):
        download_sample_video()

    from src.detection.detector import VehicleDetector  # heavy import, deferred
    detector = VehicleDetector()

    print("Scanning real footage for clean vehicle crops...")
    candidates = _collect_candidates(config.SAMPLE_VIDEO_PATH, detector)
    print(f"  {len(candidates)} candidate crops found.")

    ambulance = _pick_cleanest_white(candidates)
    if ambulance is not None:
        cv2.imwrite(ambulance_path, ambulance["crop"])
        print(f"  Saved ambulance base -> {ambulance_path}")

    remaining = [c for c in candidates if ambulance is None or not np.array_equal(c["crop"], ambulance["crop"])]
    traffic = _pick_diverse(remaining, n_traffic_templates)
    for path, cand in zip(traffic_paths, traffic):
        cv2.imwrite(path, cand["crop"])
        print(f"  Saved traffic template -> {path}")


if __name__ == "__main__":
    run()
