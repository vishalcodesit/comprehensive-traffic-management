"""
Generates a deterministic synthetic traffic video so the emergency-vehicle
detection pipeline can be demoed with a guaranteed, on-cue "ambulance" --
something a real video clip can't offer (you can't script when a real
ambulance drives through frame).

Earlier version of this module drew flat-colored rectangles for cars, which
a COCO-pretrained YOLOv8 never detects (verified empirically: 0 detections
across 200 frames) -- it has only ever seen real photographs, and a solid-
color rectangle carries none of the texture/shading cues it learned from.
So instead this composites real vehicle photo crops (extracted from the real
intersection footage in data/samples/, see download_sample_video.py) onto a
synthetic road, which YOLOv8 *does* detect, while keeping full scripted
control over when the "ambulance" (a real car crop with a drawn strobing
red/blue light bar on its roof) appears.

Run:  python -m src.detection.synthetic_video
"""
import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

WIDTH, HEIGHT = 960, 540
FPS = 20
LANES = (280, 400, 560, 680)


def lane_zones(lanes=LANES):
    """VideoPipeline zones with one zone per lane (boundaries at lane
    midpoints), so each lane maps to its own approach. The pipeline's default
    equal-quarter split would put all four lanes into only the middle two
    zones, since the road occupies the centre of the frame."""
    fracs = [x / WIDTH for x in lanes]
    edges = [0.0] + [(a + b) / 2 for a, b in zip(fracs, fracs[1:])] + [1.0]
    return [(f"approach_{i + 1}", lo, hi) for i, (lo, hi) in enumerate(zip(edges, edges[1:]))]


def _load_templates():
    """Real vehicle photo crops extracted from data/samples/intersection_sample.webm
    (see extract_car_templates.py / download_sample_video.py), auto-extracted
    on first use if missing. Falls back to a plain rectangle only if that
    extraction can't run -- keeps this module runnable standalone, just
    without the "real pixels" detectability boost."""
    ambulance_path = os.path.join(config.SAMPLES_DATA_DIR, "car_template.png")
    traffic_paths = sorted(glob.glob(os.path.join(config.SAMPLES_DATA_DIR, "car_template_*.png")))

    if not (traffic_paths and os.path.exists(ambulance_path)):
        try:
            from src.detection.extract_car_templates import run as extract_templates
            extract_templates()
            traffic_paths = sorted(glob.glob(os.path.join(config.SAMPLES_DATA_DIR, "car_template_*.png")))
        except Exception as e:
            print(f"[synthetic_video] Could not auto-extract real vehicle templates ({e}); "
                  f"falling back to plain rectangles (won't be YOLO-detectable).")

    traffic_imgs = [cv2.imread(p) for p in traffic_paths]
    traffic_imgs = [img for img in traffic_imgs if img is not None]
    ambulance_img = cv2.imread(ambulance_path) if os.path.exists(ambulance_path) else None

    if not traffic_imgs:
        traffic_imgs = [np.full((90, 60, 3), (150, 150, 150), dtype=np.uint8)]
    if ambulance_img is None:
        ambulance_img = np.full((90, 60, 3), (245, 245, 245), dtype=np.uint8)

    return traffic_imgs, ambulance_img


TRAFFIC_TEMPLATES, AMBULANCE_TEMPLATE = _load_templates()


def _paste(frame, img, cx, cy):
    """Pastes img centered at (cx, cy) into frame, clipped to frame bounds."""
    h, w = img.shape[:2]
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = x1 + w, y1 + h

    src_x1, src_y1 = max(0, -x1), max(0, -y1)
    dst_x1, dst_y1 = max(0, x1), max(0, y1)
    dst_x2, dst_y2 = min(WIDTH, x2), min(HEIGHT, y2)
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return None  # fully off-screen
    frame[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    return (x1, y1, x2, y2)  # full (possibly off-screen) bbox for the light bar overlay


class _Car:
    def __init__(self, lane_x, is_emergency=False):
        self.x = lane_x
        self.y = -random.randint(20, 300)
        self.speed = random.uniform(3.5, 6.5)
        self.is_emergency = is_emergency
        self.frame_count = 0

        template = AMBULANCE_TEMPLATE if is_emergency else random.choice(TRAFFIC_TEMPLATES)
        # Ambulances are bigger than cars -- and the ambulance base crop is
        # only reliably YOLO-detectable from ~70px wide, so drawing it at
        # car size made it vanish from detection (and thus from tracking).
        target_w = random.randint(80, 92) if is_emergency else random.randint(56, 74)
        scale = target_w / template.shape[1]
        target_h = max(1, int(template.shape[0] * scale))
        self.img = cv2.resize(template, (target_w, target_h))
        self.w, self.h = target_w, target_h

    def step(self):
        self.y += self.speed
        self.frame_count += 1

    def offscreen(self):
        return self.y > HEIGHT + 20

    def draw(self, frame):
        bbox = _paste(frame, self.img, self.x, self.y)
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        if self.is_emergency:
            bar_h = max(6, int(self.h * 0.22))
            # Strobe red/blue at ~3-4 Hz so emergency_classifier's flicker
            # heuristic (which watches red/blue ratio oscillation) fires.
            on_red = (self.frame_count // 3) % 2 == 0
            color = (0, 0, 255) if on_red else (255, 60, 0)
            # Mounted on top of the vehicle like a real light bar. Painting
            # it over the car photo hid the windows/roofline YOLO relies on,
            # so it lost the ambulance after a few frames.
            bx1, bx2 = max(0, x1 + 3), min(WIDTH, x2 - 3)
            by1, by2 = max(0, y1 - bar_h), min(HEIGHT, y1)
            if bx2 > bx1 and by2 > by1:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)


def generate(path=None, seconds=20, lanes=LANES, seed=7):
    path = path or os.path.join(config.SYNTHETIC_DATA_DIR, "demo_traffic.mp4")
    random.seed(seed)
    np.random.seed(seed)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))

    n_frames = seconds * FPS
    cars = []
    spawn_lo, spawn_hi = 30, n_frames - 60
    if spawn_hi <= spawn_lo:
        # Too short a clip for the usual [30, n_frames-60) window -- still
        # guarantee at least one emergency spawn if there's room at all.
        emergency_spawn_frames = {n_frames // 2} if n_frames > 10 else set()
    else:
        k = min(max(1, seconds // 6), spawn_hi - spawn_lo)
        emergency_spawn_frames = set(random.sample(range(spawn_lo, spawn_hi), k=k))

    for f in range(n_frames):
        frame = np.full((HEIGHT, WIDTH, 3), (60, 130, 60), dtype=np.uint8)  # grass
        road_x1, road_x2 = 220, 740
        cv2.rectangle(frame, (road_x1, 0), (road_x2, HEIGHT), (50, 50, 50), -1)
        for lx in lanes[1:-1]:
            for dash_y in range(0, HEIGHT, 40):
                cv2.line(frame, (lx, dash_y), (lx, dash_y + 20), (220, 220, 220), 2)

        if random.random() < 0.06:
            lane_x = random.choice(lanes)
            cars.append(_Car(lane_x))
        if f in emergency_spawn_frames:
            cars.append(_Car(random.choice(lanes), is_emergency=True))

        for car in cars:
            car.step()
        cars = [c for c in cars if not c.offscreen()]
        cars.sort(key=lambda c: c.y)
        for car in cars:
            car.draw(frame)

        cv2.putText(frame, f"SYNTHETIC DEMO FEED  frame {f}/{n_frames}", (16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(f"Wrote {n_frames} frames ({seconds}s @ {FPS}fps) -> {path}")
    return path


if __name__ == "__main__":
    generate()
