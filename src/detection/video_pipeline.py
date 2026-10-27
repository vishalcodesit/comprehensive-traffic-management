"""
End-to-end per-frame pipeline: YOLOv8 detection -> DeepSORT tracking ->
emergency-vehicle scoring -> annotated frame + structured summary.

The summary's per-zone vehicle counts and emergency flags are what feed
src/integration/decision_engine.py, which maps a single camera's lane split
onto the 4 approaches of the signal-control simulation.

Run standalone for a smoke test:  python -m src.detection.video_pipeline
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.detection.detector import VehicleDetector
from src.detection.tracker import VehicleTracker
from src.detection.emergency_classifier import EmergencyVehicleDetector

BOX_COLOR_NORMAL = (60, 200, 60)
BOX_COLOR_EMERGENCY = (0, 0, 255)
LANE_LABEL_BG = (40, 40, 40)
LANE_DIVIDER_COLOR = (230, 230, 230)

DEFAULT_ZONES = [
    ("approach_1", 0.0, 0.25), ("approach_2", 0.25, 0.5),
    ("approach_3", 0.5, 0.75), ("approach_4", 0.75, 1.0),
]


def draw_lane_overlay(frame, zones, zone_counts, zone_emergency, top=34):
    """Draws lane dividers plus, at the top of each lane, its name and live
    vehicle count -- on a red background with a PRIORITY line when a
    priority (emergency) vehicle is in that lane."""
    h, w = frame.shape[:2]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    for name, lo, hi in zones:
        if lo > 0:
            x = int(lo * w)
            for y in range(top, h, 24):  # dashed divider
                cv2.line(frame, (x, y), (x, min(h, y + 12)), LANE_DIVIDER_COLOR, 1, cv2.LINE_AA)

        lines = [f"{name}: {zone_counts[name]}"]
        if zone_emergency[name]:
            lines.append("PRIORITY")
        sizes = [cv2.getTextSize(s, font, scale, 1)[0] for s in lines]
        box_w = max(sw for sw, _ in sizes) + 12
        box_h = sum(sh for _, sh in sizes) + 8 * len(lines) + 4
        cx = int((lo + hi) / 2 * w)
        x1 = max(0, min(w - box_w, cx - box_w // 2))
        bg = BOX_COLOR_EMERGENCY if zone_emergency[name] else LANE_LABEL_BG
        cv2.rectangle(frame, (x1, top), (x1 + box_w, top + box_h), bg, -1)
        y = top + 4
        for s, (sw, sh) in zip(lines, sizes):
            y += sh + 6
            cv2.putText(frame, s, (x1 + (box_w - sw) // 2, y), font, scale,
                        (255, 255, 255), 1, cv2.LINE_AA)


class VideoPipeline:
    def __init__(self, zones=None, draw_lanes=True):
        """zones: list of (name, x_min_frac, x_max_frac) splitting the frame
        width into named regions (e.g. lanes / camera-visible approaches)."""
        self.detector = VehicleDetector()
        self.tracker = VehicleTracker()
        self.emergency = EmergencyVehicleDetector()
        self.zones = zones or DEFAULT_ZONES
        self.draw_lanes = draw_lanes

    def _zone_for(self, bbox, frame_w):
        cx = (bbox[0] + bbox[2]) / 2.0
        frac = cx / frame_w
        for name, lo, hi in self.zones:
            if lo <= frac < hi:
                return name
        return self.zones[-1][0]

    def process_frame(self, frame):
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        scored = [self.emergency.evaluate(frame, t) for t in tracks]

        h, w = frame.shape[:2]
        zone_counts = {name: 0 for name, _, _ in self.zones}
        zone_emergency = {name: False for name, _, _ in self.zones}

        for t in scored:
            t["zone"] = self._zone_for(t["bbox"], w)
            zone_counts[t["zone"]] += 1
            zone_emergency[t["zone"]] = zone_emergency[t["zone"]] or t["is_emergency"]

        annotated = frame.copy()
        if self.draw_lanes:
            # Drawn first so vehicle boxes/labels stay readable on top of it.
            draw_lane_overlay(annotated, self.zones, zone_counts, zone_emergency)
        for t in scored:
            is_emg = t["is_emergency"]
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            color = BOX_COLOR_EMERGENCY if is_emg else BOX_COLOR_NORMAL
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f'#{t["track_id"]} {t["class_name"]}'
            if is_emg:
                label = f'EMERGENCY {label} ({t["emergency_score"]:.2f})'
            cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        summary = {
            "n_vehicles": len(scored),
            "zone_counts": zone_counts,
            "zone_emergency": zone_emergency,
            "any_emergency": any(zone_emergency.values()),
        }
        return annotated, summary

    def frames(self, source):
        """Generator over (annotated_frame, summary) for a video file / camera index."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.process_frame(frame)
        finally:
            cap.release()

    def process_video(self, source, output_path=None):
        writer = None
        n = 0
        t0 = time.time()
        for annotated, summary in self.frames(source):
            if output_path and writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, 20, (w, h))
            if writer:
                writer.write(annotated)
            n += 1
            if summary["any_emergency"]:
                print(f"[frame {n}] EMERGENCY VEHICLE detected -> zones: "
                      f"{[z for z, v in summary['zone_emergency'].items() if v]}")
        if writer:
            writer.release()
        elapsed = time.time() - t0
        print(f"Processed {n} frames in {elapsed:.1f}s ({n / max(elapsed, 1e-6):.1f} fps).")
        if output_path:
            print(f"Annotated video saved to {output_path}")


if __name__ == "__main__":
    from src.detection.synthetic_video import generate
    video_path = generate(seconds=10)
    pipeline = VideoPipeline()
    out_path = os.path.join(config.RESULTS_DIR, "demo_annotated.mp4")
    pipeline.process_video(video_path, output_path=out_path)
