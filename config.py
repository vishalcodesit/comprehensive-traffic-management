"""Central configuration: paths and shared hyperparameters."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
SAMPLES_DATA_DIR = os.path.join(DATA_DIR, "samples")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

METRO_CSV = os.path.join(RAW_DATA_DIR, "Metro_Interstate_Traffic_Volume.csv")
PROCESSED_NPZ = os.path.join(PROCESSED_DATA_DIR, "congestion_windows.npz")
SCALERS_PKL = os.path.join(PROCESSED_DATA_DIR, "congestion_scalers.pkl")

# ---- Congestion prediction ----
LOOKBACK_HOURS = 24
HORIZONS = [1, 3, 6]  # predict traffic volume this many hours ahead
CONGESTION_LEVELS = ["Low", "Medium", "High", "Severe"]
CONGESTION_QUANTILES = [0.33, 0.67, 0.90]

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3

CONGESTION_MODEL_PATH = os.path.join(MODELS_DIR, "congestion_lstm.pt")

# ---- Vehicle / emergency detection ----
YOLO_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n.pt")
EMERGENCY_VEHICLE_COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DETECTION_CONF_THRESHOLD = 0.35

# ---- Signal control (RL) ----
NUM_APPROACHES = 4  # N, E, S, W
MAX_QUEUE = 30
RL_MODEL_PATH = os.path.join(MODELS_DIR, "dqn_signal_control.pt")
RL_EPISODES = 400
RL_MAX_STEPS = 300
RL_GAMMA = 0.95
RL_LR = 5e-4
RL_BATCH_SIZE = 64
RL_REPLAY_CAPACITY = 20000
RL_TARGET_UPDATE_EVERY = 10
MIN_GREEN_STEPS = 3

# Real-footage demo clip (CC-BY-SA 4.0, Thomas1313, Wikimedia Commons) --
# used because a COCO-pretrained YOLOv8 does not reliably fire on the
# flat-shaded synthetic_video.py renders (verified empirically: 0 detections
# across 200 synthetic frames vs. ~5/frame on this real clip). See
# src/detection/download_sample_video.py.
SAMPLE_VIDEO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/1/18/"
    "Intersection_Pie-IX-Sherbrooke.webm/Intersection_Pie-IX-Sherbrooke.webm.480p.vp9.webm"
)
SAMPLE_VIDEO_ATTRIBUTION = (
    "\"Intersection Pie-IX-Sherbrooke\" by Thomas1313, Wikimedia Commons, "
    "CC BY-SA 4.0 -- https://commons.wikimedia.org/wiki/File:Intersection_Pie-IX-Sherbrooke.webm"
)
SAMPLE_VIDEO_PATH = os.path.join(SAMPLES_DATA_DIR, "intersection_sample.webm")

for _d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, SAMPLES_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)
