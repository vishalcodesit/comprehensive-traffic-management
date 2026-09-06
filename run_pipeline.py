"""
Runs the whole project end to end, in order, from a clean checkout:
preprocessing -> congestion model training/eval -> RL training/eval ->
integrated demo. Vehicle-detection is exercised separately (it needs a
video/webcam) -- see src/detection/video_pipeline.py and the dashboard.

Run:  python run_pipeline.py [--skip-slow]

--skip-slow trims epoch/episode counts for a quick correctness check rather
than a full training run (useful for CI or a fast sanity pass).
"""
import argparse
import subprocess
import sys
import time


STEPS = [
    ("Preprocessing traffic dataset", [sys.executable, "-m", "src.congestion.preprocess"]),
    ("Training congestion LSTM", [sys.executable, "-m", "src.congestion.train"]),
    ("Evaluating congestion LSTM", [sys.executable, "-m", "src.congestion.evaluate"]),
    ("Training DQN signal controller", [sys.executable, "-m", "src.signal_control.train_rl"]),
    ("Evaluating DQN vs fixed-time baseline", [sys.executable, "-m", "src.signal_control.evaluate_rl"]),
    ("Running integrated end-to-end demo", [sys.executable, "-m", "src.integration.run_end_to_end_demo"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-slow", action="store_true",
                         help="Not implemented as a flag pass-through yet; edit config.py's "
                              "EPOCHS/RL_EPISODES directly for a quick pass.")
    parser.parse_args()

    t0 = time.time()
    for i, (label, cmd) in enumerate(STEPS, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {label}\n{'=' * 70}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nStep failed: {label} (exit {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print(f"\nAll steps complete in {time.time() - t0:.1f}s. "
          f"See results/ for plots and metrics, models/ for checkpoints.\n"
          f"Run `python dashboard/app.py` for the live demo.")


if __name__ == "__main__":
    main()
