# Optional: labeled emergency-vehicle dataset

Drop cropped vehicle images here to fine-tune the optional CNN head
(`src/detection/train_emergency_cnn.py`) that ensembles into the emergency
classifier. Not required — the project runs on heuristics alone without this.

Expected layout (ImageFolder-style):

```
train/emergency/*.jpg
train/normal/*.jpg
val/emergency/*.jpg
val/normal/*.jpg
```

Any public labeled set of ambulance/fire-truck/police vs. ordinary vehicle
crops works (e.g. a Kaggle "emergency vehicle identification" dataset).
