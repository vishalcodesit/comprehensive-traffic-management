"""Thin wrapper around Ultralytics YOLOv8 restricted to vehicle classes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class VehicleDetector:
    def __init__(self, weights=config.YOLO_WEIGHTS, conf=config.DETECTION_CONF_THRESHOLD, device=None):
        from ultralytics import YOLO
        import torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)
        self.conf = conf
        self.vehicle_classes = config.EMERGENCY_VEHICLE_COCO_CLASSES  # {2: car, 3: motorcycle, 5: bus, 7: truck}

    def detect(self, frame):
        """
        frame: BGR numpy image (as read by OpenCV).
        Returns list of dicts: {bbox: [x1,y1,x2,y2], conf, class_name}
        """
        results = self.model.predict(
            frame, conf=self.conf, classes=list(self.vehicle_classes.keys()),
            device=self.device, verbose=False,
        )
        detections = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            detections.append({
                "bbox": xyxy,
                "conf": conf,
                "class_id": cls_id,
                "class_name": self.vehicle_classes.get(cls_id, "vehicle"),
            })
        return detections
