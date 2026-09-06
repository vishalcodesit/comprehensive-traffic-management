"""DeepSORT multi-object tracking on top of per-frame YOLOv8 detections.

Tracking (rather than raw per-frame detection) matters here because it gives
every vehicle a stable ID across frames, which is what lets the emergency
classifier accumulate evidence (siren-light flicker) over time instead of
guessing from a single frame, and lets the signal-control layer count
distinct vehicles per approach instead of re-counting the same car each frame.
"""
from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleTracker:
    def __init__(self, max_age=20, n_init=2):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, embedder="mobilenet", half=True)

    def update(self, detections, frame):
        """
        detections: list of {"bbox": [x1,y1,x2,y2], "conf": float, "class_name": str}
        Returns list of {"track_id": int, "bbox": [x1,y1,x2,y2], "class_name": str}
        """
        ds_input = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            ds_input.append(([x1, y1, x2 - x1, y2 - y1], d["conf"], d["class_name"]))

        tracks = self.tracker.update_tracks(ds_input, frame=frame)

        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, top, r, b = t.to_ltrb()
            results.append({
                "track_id": t.track_id,
                "bbox": [float(l), float(top), float(r), float(b)],
                "class_name": t.get_det_class() or "vehicle",
            })
        return results
