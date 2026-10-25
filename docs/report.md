# Comprehensive Traffic Management Using Machine Learning Techniques

**BCSE 497J — Project-I** · Tanmay Agarwal (23BAI0172) · Vishal Tiwari (23BCB0144)
Faculty guide: Ushus Elizabeth Zachariah

This document extends the Project Review presentation (aim, abstract,
literature review, research gap, objectives, SDG alignment — reproduced
below for completeness) with the system design, implementation, and
experimental results of the working prototype in this repository.

---

## 1. Aim

To develop an intelligent, AI-driven traffic management system that
integrates Traffic Congestion Prediction and Priority Vehicle Detection to
optimize urban traffic flow, reduce congestion, and provide seamless
right-of-way for emergency vehicles through adaptive traffic signal control.

## 2. Abstract

Urban traffic congestion and delayed emergency vehicle movement are major
challenges, increasing travel time, fuel consumption, emissions, and
reducing emergency response efficiency. Existing systems rely on fixed
signal timings with limited adaptability. This project implements a
Comprehensive Traffic Management System integrating Traffic Congestion
Prediction and Priority Vehicle Detection into one framework: an LSTM
forecasts congestion from historical and real-time traffic data; YOLOv8 +
DeepSORT detect and track vehicles from video, with siren-flicker and
livery heuristics flagging emergency vehicles; a Deep Q-Network learns
adaptive signal control on a simulated intersection. A `DecisionEngine`
fuses all three: emergency detections hard-override the signal (safety
interlock), and congestion forecasts soft-bias the DQN's action selection,
so the controller both reacts to what a camera sees right now and
anticipates congestion before it fully forms.

## 3. Literature review & research gap

See the original review deck (`Project Review 1 .pdf`) for the 10-paper
literature review and the identified research gap: most existing work
treats congestion prediction, signal optimization, and emergency-vehicle
detection as **independent** problems, with limited unified frameworks and
limited real-time, automatic emergency prioritization. This project's
`DecisionEngine` (§5.4) is the direct response to that gap — it is the
single component whose entire job is fusing the other three.

## 4. Objectives — status

| Objective | Status |
|---|---|
| Predict traffic congestion from historical + real-time data | Implemented — LSTM on real UCI Metro Interstate Traffic Volume data (`src/congestion/`) |
| Detect priority vehicles from live video | Implemented — YOLOv8 + DeepSORT + heuristic/optional-CNN emergency classifier (`src/detection/`) |
| Adaptive signal control (congestion + emergency aware) | Implemented — DQN on custom intersection simulator (`src/signal_control/`) |
| Integrated decision-making framework | Implemented — `DecisionEngine` (`src/integration/`) |
| Smart-city / ITS scalability | Architected for it (stateless prediction/detection services + a pluggable environment interface), not deployed at city scale — see §7 |

## 5. System design

### 5.1 Congestion prediction

- **Data**: UCI Metro Interstate Traffic Volume — ~48,200 hourly records of
  I-94 westbound traffic (Minneapolis-St Paul), Oct 2012–Sep 2018, with
  weather (temperature, rain, snow, cloud cover, conditions) and US holidays.
- **Cleaning**: collapse duplicate per-hour weather-event rows, clip a known
  rain-sensor outlier (>1000mm/hr readings), reindex to a continuous hourly
  grid and interpolate short gaps.
- **Features**: raw volume, weather numerics, cyclical hour/day-of-week/month
  encodings (sin/cos), weekend/holiday/rush-hour flags, one-hot weather
  buckets.
- **Labels**: congestion level {Low, Medium, High, Severe} via quantile
  thresholds (33rd/67th/90th percentile) fit on the training split only.
- **Model**: 2-layer LSTM (hidden=128) with an attention-pooling layer over
  the 24-hour lookback window, feeding two heads: volume regression and
  congestion-level classification, jointly trained (MSE + 0.3×CrossEntropy)
  at 1h/3h/6h horizons.
- **Split**: chronological 70/15/15 train/val/test (no shuffling across the
  boundary, to avoid leaking future information backward).

### 5.2 Priority vehicle detection

- **Detection**: YOLOv8n restricted to COCO vehicle classes (car, motorcycle,
  bus, truck).
- **Tracking**: DeepSORT (MobileNet embedder) assigns each vehicle a stable
  ID across frames — necessary because the emergency signal below is
  inherently temporal.
- **Emergency classification**: no licensed labeled dataset of emergency
  vehicles ships with this project (see §7), so classification runs on two
  physically-motivated heuristics computed per tracked vehicle over a
  rolling window: (a) red/blue pixel-ratio oscillation in the upper
  "light-bar" band of the bounding box (a strobing beacon flickers; a
  static-colored roof does not), and (b) edge density + red/white livery
  ratio across the full crop (ambulances/fire trucks are disproportionately
  white with high-contrast markings). An optional MobileNetV3 CNN head
  (`train_emergency_cnn.py`) ensembles in automatically once a labeled
  dataset is added.

### 5.3 Adaptive signal control

- **Environment**: a custom Gymnasium `IntersectionEnv` — 4 approaches (N/E/S/
  W), Poisson vehicle arrivals with per-episode-randomized rates, a
  saturation discharge rate for the green approach, a minimum-green-time
  safety interlock, and stochastic emergency-vehicle spawns that persist
  (accruing wait-time penalty) until the flagged approach is served.
- **Agent**: DQN (2 hidden layers, 128 units), replay buffer, target
  network, ε-greedy exploration decayed over training.
- **Reward**: `-(0.1 × total queue) - 4×(emergency vehicles waiting) -
  5×(approaches at gridlock) - 0.5×(invalid phase-switch attempts)`.
- **Baseline**: a conventional fixed-time round-robin controller, evaluated
  on identical random seeds (same arrival/emergency patterns) for a fair
  comparison.

### 5.4 Decision engine (the integration layer)

Given a live observation (queues, arrival-rate estimate, emergency flags,
current phase), `DecisionEngine.decide()`:

1. If any approach has an emergency vehicle waiting → **hard override**:
   green goes to that approach immediately, regardless of the learned
   policy. This is a safety interlock, not a suggestion the DQN can
   overrule — a false negative here has real consequences, so it isn't left
   to a black box.
2. Otherwise, take the DQN's Q-values and **add a bias** proportional to
   each approach's forecast congestion level (from the LSTM, broadcast
   across approaches weighted by their current share of detected traffic),
   then pick the arg-max. This lets the controller start clearing an
   approach that's about to get busy, not only one that already is.

## 6. Results

From the actual training/evaluation run used to build this repository
(RTX 4050 laptop GPU; full plots in `results/`).

**Congestion LSTM**, test set (7,878 held-out hours, chronological split so
no future data leaks into training):

| Horizon | MAE (veh/hr) | RMSE | MAPE | Congestion-level accuracy | Macro F1 |
|---|---|---|---|---|---|
| +1h | 225 | 321 | 14.0% | 89.7% | 0.878 |
| +3h | 261 | 372 | 14.9% | 88.2% | 0.864 |
| +6h | 260 | 405 | 14.2% | 87.0% | 0.850 |

Per-class precision/recall at +1h (`results/congestion_confusion_matrix.png`)
shows the errors are concentrated at adjacent-band boundaries (e.g. Medium
predicted as Low) rather than gross misses (e.g. Low predicted as Severe),
which is the failure mode that matters least for signal-control decisions.

**DQN adaptive signal control vs. fixed-time baseline**, 30 held-out
evaluation episodes on identical random seeds (same arrival patterns and
emergency-vehicle spawns, so the comparison isn't confounded by luckier
traffic):

| Metric | DQN adaptive | Fixed-time | Improvement |
|---|---|---|---|
| Avg. queue length/step | 19.9 vehicles | 33.8 vehicles | **-41.2%** |
| Throughput | 563.6 veh/episode | 522.3 veh/episode | **+7.9%** |
| Avg. emergency-vehicle wait/step | 0.025 | 0.078 | **-68.6%** |

This reproduces the literature's headline finding (RL-based adaptive control
beats fixed-time on queue length and emissions-correlated metrics) on this
project's own simulator, and additionally shows the emergency-preemption
reward term translating into a large, specific reduction in emergency wait
time — not just an average-case improvement.

**Integration**: `results/end_to_end_demo_trace.png` traces the real trained
LSTM and DQN running together through `DecisionEngine` for 200 steps,
including two live emergency-override events (queue for the flagged approach
is marked with a star) and a mid-run shift in the LSTM's forecast level from
Low to High visibly changing the Q-value bias in the printed decision trace.

**Detection**: YOLOv8 was verified to produce ~5 vehicle detections/frame on
real footage (`data/samples/`) at 30+ fps on the RTX 4050, and the
siren-flicker heuristic fires on 134/500 frames (~27%) of the synthetic
emergency-demo clip -- i.e., reliably while the "ambulance" is actually in
frame, not spuriously on ordinary traffic.

## 7. Honest limitations & future work

- **The dashboard runs two video feeds on purpose.** A COCO-pretrained
  YOLOv8 does not fire on flat-shaded procedural renders (verified
  empirically: 0/200 frames), so `/video_feed` uses a real ~1-minute
  intersection clip (CC BY-SA 4.0, Thomas1313, Wikimedia Commons) to prove
  detection/tracking work on genuine footage, while `/video_feed_emergency`
  uses a deterministic synthetic clip -- built by compositing real vehicle
  photo crops extracted from that same real clip
  (`src/detection/extract_car_templates.py`) onto a scripted road scene, with
  a drawn strobing light bar on one "ambulance" -- specifically to guarantee
  an emergency vehicle appears on cue, which no real clip can promise.
  Point `VideoPipeline` at any other real video file or webcam index and
  nothing else in the pipeline changes.
- **Emergency classification** is heuristic-only until a labeled dataset is
  supplied (§5.2); the CNN ensemble path is implemented and ready.
- **Intersection dynamics** use a lightweight custom simulator rather than
  SUMO (the standard tool in the reviewed literature), trading simulation
  fidelity for a dependency-free, fast-training setup. A SUMO-backed
  `IntersectionEnv` implementing the same `reset()`/`step()` contract is a
  natural next step, especially for city-scale, multi-intersection
  coordination (currently single-intersection only).
- **Multi-intersection coordination**, IoT sensor fusion, and live
  deployment integration (the "Smart City Readiness" objective) are
  architected for — stateless, swappable model/environment interfaces — but
  not demonstrated at that scale here.

## 8. SDG alignment

SDG 3 (Good Health & Well-being — faster emergency response via automatic
signal preemption), SDG 9 (Industry, Innovation & Infrastructure — AI-driven
ITS), SDG 11 (Sustainable Cities — adaptive urban mobility), SDG 13 (Climate
Action — reduced idling/congestion and emissions from optimized signal
timing).

## 9. References

See `Project Review 1 .pdf` for the full IEEE-formatted reference list
(10 papers) this project's literature review is drawn from, spanning
DQN/SUMO-based signal control, CNN/HMM/genetic-algorithm congestion
detection, generative-AI traffic forecasting, YOLOv8+DeepSORT+LSTM pipelines,
and IoT-based traffic systems.
