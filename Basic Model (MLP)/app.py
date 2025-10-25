#!/usr/bin/env python3
"""
Real-time ISL -> Text Translator (Keypoint-based)

- Loads your trained MLP model (expects 126-dim keypoints: Left then Right, 21x3 each).
- Extracts MediaPipe hands keypoints in real-time (matching training format).
- Temporal smoothing (EMA) + hold-to-commit with cooldown.
- Resizable window with uniform scaling (no webcam aspect ratio distortion).
- 'blank' class commits a single space; prevents spamming.

Keys:
  ESC: quit
  SPACE: insert space
  B / Backspace: delete last char
  C: clear
  +/-: zoom display (uniform scale)
  R: reset smoothing buffer
  S: save transcript to file
"""

import os
import sys
import time
import json
import argparse
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

import keras

# ---------------- CONFIG (defaults; override via CLI) ----------------
DEFAULT_MODEL_PATH = "models/isl_wcs_raw_aug_light/best.keras"
DEFAULT_LABELS_JSON = "models/isl_wcs_raw_aug_light/labels.json"

# Commit logic
CONF_THRESH = 0.60         # min probability to consider a class a 'candidate'
HOLD_SECONDS = 3.0         # how long a class must remain stable to commit
COOLDOWN_SECONDS = 0.8     # minimum gap between commits

# Smoothing
SMOOTH_EMA_ALPHA = 0.20    # EMA factor for probability smoothing (0..1)
SMOOTH_MIN_FRAMES = 3      # require at least N frames before trusting EMA

# GUI scaling (no effect on capture; keeps aspect ratio)
DISPLAY_SCALE = 0.80
DISPLAY_MIN, DISPLAY_MAX = 0.40, 1.20

# Webcam
CAM_INDEX = 0
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# ROI display size
ROI_PREVIEW_SIZE = 220

# Model input dims
NUM_LM_PER_HAND = 21
DIMS_PER_LM = 3
FEAT_DIM = NUM_LM_PER_HAND * DIMS_PER_LM * 2  # 126

# --------------------------------------------------------------------

# --- Register the custom Lambda function used in the trained model ---


@tf.keras.utils.register_keras_serializable(package="Custom", name="wcs_fn")
def wcs_fn(t):
    """Wrist-center + scale invariance + presence flags.
    Input t: (B, 2, 21, 3) -> returns (centered, present_flags)."""
    EPS = 1e-6
    wrist = t[:, :, 0:1, :]                           # (B,2,1,3)
    centered = t - wrist                              # (B,2,21,3)
    dist = tf.norm(centered, axis=-1)                 # (B,2,21)
    span = tf.reduce_max(dist, axis=-1, keepdims=True)  # (B,2,1)
    span = tf.maximum(span, EPS)
    centered = centered / span[..., None]             # (B,2,21,3)
    present = tf.reduce_sum(tf.abs(t), axis=[2, 3])   # (B,2)
    present = tf.cast(tf.not_equal(present, 0.0), tf.float32)
    return centered, present


@tf.keras.utils.register_keras_serializable(package="Custom", name="pres_fn")
def pres_fn(t):
    """Presence flags only: (B, 2, 21, 3) -> (B, 2) float32 in {0,1}."""
    present = tf.reduce_sum(tf.abs(t), axis=[2, 3])   # (B,2)
    present = tf.cast(tf.not_equal(present, 0.0), tf.float32)
    return present

# Some graphs save explicit slicing as Lambda. Register them too to be safe.


@tf.keras.utils.register_keras_serializable(package="Custom", name="lhand_fn")
def lhand_fn(z):
    """(B,2,21,3) -> (B,21,3) - left hand slice."""
    return z[:, 0, :, :]


@tf.keras.utils.register_keras_serializable(package="Custom", name="rhand_fn")
def rhand_fn(z):
    """(B,2,21,3) -> (B,21,3) - right hand slice."""
    return z[:, 1, :, :]


def load_labels(labels_path):
    with open(labels_path, "r") as f:
        obj = json.load(f)
    if "classes" in obj:
        return obj["classes"]
    if isinstance(obj, dict) and "label2idx" in obj:
        return [c for c, _ in sorted(obj["label2idx"].items(), key=lambda kv: kv[1])]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unrecognized labels file format: {labels_path}")


def load_model_auto(model_path: str):
    """
    Load a Keras model containing Lambda layers & custom functions.
    We pass safe_mode=False and supply custom_objects.
    """
    custom = {
        "wcs_fn": wcs_fn,
        "pres_fn": pres_fn,
        "lhand_fn": lhand_fn,
        "rhand_fn": rhand_fn,
    }

    e1 = None
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom,
            safe_mode=False,   # allow Lambda deserialization
        )
        print(f"[INFO] Loaded model (safe_mode=False): {model_path}")
        return model
    except Exception as _e1:
        e1 = _e1
        print(f"[WARN] Keras load (safe_mode=False) failed: {e1}")

    # Second attempt with normalized path (sometimes helps on Windows)
    e2 = None
    try:
        model = tf.keras.models.load_model(
            os.path.normpath(model_path),
            compile=False,
            custom_objects=custom,
            safe_mode=False,
        )
        print(f"[INFO] Loaded model on second attempt: {model_path}")
        return model
    except Exception as _e2:
        e2 = _e2
        msg_first = f"{e1}" if e1 is not None else "<no first error>"
        raise RuntimeError(
            f"Failed to load model '{model_path}'.\n"
            f"First error: {msg_first}\nSecond error: {e2}"
        )


# --------------- MediaPipe Hands (2-hand) ---------------
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    USE_MP = True
except Exception:
    print("[ERROR] mediapipe not available. Run: pip install mediapipe")
    USE_MP = False


def extract_2hand_keypoints(frame_bgr, hands_ctx):
    """
    Returns:
      vec (126,) float32  -> Left(63) then Right(63); zeros if missing
      present (L_present, R_present) -> booleans
      results -> mediapipe detection (for drawing)
      bbox -> bounding box that encloses detected landmarks (x1,y1,x2,y2) or full frame if none
    """
    h, w = frame_bgr.shape[:2]
    vec_left = np.zeros((NUM_LM_PER_HAND, DIMS_PER_LM), dtype=np.float32)
    vec_right = np.zeros((NUM_LM_PER_HAND, DIMS_PER_LM), dtype=np.float32)
    left_present = False
    right_present = False
    bbox = (0, 0, w, h)

    if not USE_MP or hands_ctx is None:
        # No MP -> cannot extract; return zeros & full box
        out = np.concatenate(
            [vec_left.reshape(-1), vec_right.reshape(-1)], axis=0)
        return out, (left_present, right_present), None, bbox

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands_ctx.process(rgb)
    rgb.flags.writeable = True

    if not getattr(res, "multi_hand_landmarks", None):
        out = np.concatenate(
            [vec_left.reshape(-1), vec_right.reshape(-1)], axis=0)
        return out, (left_present, right_present), res, bbox

    # Collect handedness
    hand_map = {}
    if getattr(res, "multi_handedness", None):
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            label = handed.classification[0].label  # 'Left' or 'Right'
            hand_map[label] = lm
    else:
        # fallback: assign order
        lm_list = res.multi_hand_landmarks
        if len(lm_list) >= 1:
            hand_map["Right"] = lm_list[0]
        if len(lm_list) >= 2:
            hand_map["Left"] = lm_list[1]

    # Fill arrays with normalized coords
    if "Left" in hand_map:
        lm = hand_map["Left"]
        for i, p in enumerate(lm.landmark):
            vec_left[i, 0] = p.x
            vec_left[i, 1] = p.y
            vec_left[i, 2] = p.z
        left_present = True

    if "Right" in hand_map:
        lm = hand_map["Right"]
        for i, p in enumerate(lm.landmark):
            vec_right[i, 0] = p.x
            vec_right[i, 1] = p.y
            vec_right[i, 2] = p.z
        right_present = True

    # Combined bbox from all detected points (in pixel space)
    xs, ys = [], []
    for label in hand_map:
        lm = hand_map[label]
        for p in lm.landmark:
            xs.append(int(p.x * w))
            ys.append(int(p.y * h))
    if xs and ys:
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))
        # margin
        margin = int(0.10 * max(1, max(x2 - x1, y2 - y1)))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        bbox = (x1, y1, x2, y2)

    out = np.concatenate([vec_left.reshape(-1), vec_right.reshape(-1)], axis=0)
    return out.astype(np.float32), (left_present, right_present), res, bbox


# --------------- Smoothing (EMA) ---------------------
class ProbEMASmoother:
    def __init__(self, num_classes, alpha=0.2, min_frames=3):
        self.alpha = float(alpha)
        self.buf = None
        self.count = 0
        self.num_classes = num_classes
        self.min_frames = int(min_frames)

    def reset(self):
        self.buf = None
        self.count = 0

    def update(self, probs):
        probs = np.asarray(probs, dtype=np.float32)
        if self.buf is None:
            self.buf = probs.copy()
            self.count = 1
        else:
            self.buf = self.alpha * probs + (1.0 - self.alpha) * self.buf
            self.count += 1
        return self.buf, (self.count >= self.min_frames)


# --------------- GUI Helpers ------------------------
def draw_top3(panel, base_x, base_y, top3):
    for k, (lbl, c) in enumerate(top3):
        y = base_y + 24 * k
        bar_w = int(280 * c)
        cv2.rectangle(panel, (base_x, y),
                      (base_x + 280, y + 18), (45, 45, 45), 1)
        cv2.rectangle(panel, (base_x, y), (base_x +
                      bar_w, y + 18), (0, 180, 0), -1)
        cv2.putText(panel, f"{lbl}: {c*100:.1f}%", (base_x + 290, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 240, 220), 1, cv2.LINE_AA)


def draw_ring_progress(panel, cx, cy, radius, fraction, color=(0, 255, 0)):
    end_angle = int(360 * np.clip(fraction, 0.0, 1.0))
    cv2.circle(panel, (cx, cy), radius, (180, 180, 180), 2)
    cv2.ellipse(panel, (cx, cy), (radius, radius), -90, 0, end_angle, color, 4)


def save_transcript(text):
    os.makedirs("transcripts", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("transcripts", f"isl_transcript_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ----------------------------- Main Loop -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="ISL -> Text real-time translator (keypoint MLP).")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                    help="Path to .keras or .h5 model")
    ap.add_argument("--labels", type=str,
                    default=DEFAULT_LABELS_JSON, help="Path to labels.json")
    ap.add_argument("--conf", type=float, default=CONF_THRESH,
                    help="Confidence threshold to consider candidate")
    ap.add_argument("--hold", type=float, default=HOLD_SECONDS,
                    help="Hold seconds before commit")
    ap.add_argument("--cooldown", type=float, default=COOLDOWN_SECONDS,
                    help="Cooldown seconds after commit")
    ap.add_argument("--alpha", type=float,
                    default=SMOOTH_EMA_ALPHA, help="EMA alpha for smoothing")
    ap.add_argument("--min_frames", type=int, default=SMOOTH_MIN_FRAMES,
                    help="Min frames before trusting EMA")
    ap.add_argument("--scale", type=float, default=DISPLAY_SCALE,
                    help="Initial display scale (0.4..1.2)")
    args = ap.parse_args()

    # Load
    classes = load_labels(args.labels)
    num_classes = len(classes)
    print("[INFO] Classes:", classes)
    model = load_model_auto(args.model)

    if not USE_MP:
        print("[ERROR] mediapipe is required for this app. Install and retry.")
        return

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

    # Smoothing + commit state
    smoother = ProbEMASmoother(
        num_classes=num_classes, alpha=args.alpha, min_frames=args.min_frames)
    last_candidate = None
    candidate_since = None
    last_commit_time = 0.0
    typed_text = ""

    # Camera
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    cv2.namedWindow("ISL Real-time Translator", cv2.WINDOW_NORMAL)
    display_scale = float(np.clip(args.scale, DISPLAY_MIN, DISPLAY_MAX))

    # FPS
    fps_hist = deque(maxlen=30)
    last_t = time.time()

    print("[INFO] Controls: ESC=quit | Space=space | B/Backspace=delete | C=clear | +/- zoom | R=reset smooth | S=save")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Extract keypoints (126)
        vec, (L_present, R_present), mp_res, bbox = extract_2hand_keypoints(
            frame, hands)

        # Predict (only if at least one hand OR class 'blank' can still be predicted with zeros)
        # Note: During training, 'blank' is a valid class. So we always predict, but smoothing/threshold controls commit.
        inp = vec.reshape(1, -1)
        probs = model.predict(inp, verbose=0)[0]

        # smooth
        smoothed, ready = smoother.update(probs)
        use_probs = smoothed if ready else probs

        # Top-3
        order = np.argsort(-use_probs)
        top_idx = int(order[0])
        top_conf = float(use_probs[top_idx])
        top3 = [(classes[i], float(use_probs[i])) for i in order[:3]]
        top_label = classes[top_idx]

        # Draw overlay
        panel = frame.copy()
        base_x, base_y = 20, 36
        cv2.putText(panel, "ISL -> Text  (hold to commit)", (base_x, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        draw_top3(panel, base_x, base_y + 24, top3)

        # Candidate/commit logic (with threshold + hold + cooldown)
        now = time.time()
        committed = False

        # Only consider as candidate if >= conf threshold AND smoothing is ready (to reduce flicker)
        if ready and top_conf >= args.conf:
            if last_candidate == top_label:
                # same as previous candidate; accumulate hold time
                if candidate_since is None:
                    candidate_since = now
                elapsed = now - candidate_since
                if elapsed >= args.hold and (now - last_commit_time) >= args.cooldown:
                    # Commit
                    if top_label.lower() == "blank":
                        if not typed_text.endswith(" "):
                            typed_text += " "
                    else:
                        typed_text += top_label
                    last_commit_time = now
                    candidate_since = None
                    committed = True
                    smoother.reset()  # reset smoothing after commit
            else:
                # new candidate
                last_candidate = top_label
                candidate_since = now
        else:
            candidate_since = None
            last_candidate = None

        # Show candidate text + progress ring
        cand_y = base_y + 24 * 4 + 10
        cv2.putText(panel, f"Candidate: {top_label}  ({top_conf*100:.1f}%)",
                    (base_x, cand_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cx, cy, r = base_x + 30, cand_y + 36, 18
        draw_ring_progress(panel, cx, cy, r,
                           0.0 if not candidate_since else min(
                               1.0, (now - candidate_since)/args.hold),
                           color=(0, 255, 0))
        # Helper text
        if candidate_since:
            rem = max(0.0, args.hold - (now - candidate_since))
            cv2.putText(panel, f"Hold: {rem:.1f}s", (base_x + 60, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2, cv2.LINE_AA)
        else:
            cv2.putText(panel, f"Hold {args.hold:.0f}s to type",
                        (base_x + 60, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (160, 160, 160), 1, cv2.LINE_AA)

        # Draw bbox if any
        x1, y1, x2, y2 = bbox
        cv2.rectangle(panel, (x1, y1), (x2, y2), (240, 240, 240), 1)
        # Draw landmarks
        if mp_res and getattr(mp_res, "multi_hand_landmarks", None):
            for hand_lms in mp_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    panel, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        # ROI preview
        if (y2 - y1) > 0 and (x2 - x1) > 0:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                preview = cv2.resize(roi, (ROI_PREVIEW_SIZE, ROI_PREVIEW_SIZE))
                y0, y1p = 20, 20 + ROI_PREVIEW_SIZE
                x0, x1p = w - ROI_PREVIEW_SIZE - 20, w - 20
                panel[y0:y1p, x0:x1p] = preview
                cv2.putText(panel, "ROI", (x0 + 4, y0 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Typed text bar
        cv2.rectangle(panel, (20, h - 90), (w - 20, h - 30), (30, 30, 30), -1)
        cv2.putText(panel, f"Typed: {typed_text}", (30, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # FPS
        t = time.time()
        fps = 1.0 / max(1e-6, (t - last_t))
        last_t = t
        fps_hist.append(fps)
        fps_avg = sum(fps_hist) / len(fps_hist)
        cv2.putText(panel, f"FPS: {fps_avg:.1f}", (w - 160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        # Uniform display scaling (keeps aspect ratio)
        if abs(display_scale - 1.0) > 1e-3:
            disp = cv2.resize(panel, None, fx=display_scale,
                              fy=display_scale, interpolation=cv2.INTER_AREA)
        else:
            disp = panel
        cv2.imshow("ISL Real-time Translator", disp)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                      # ESC
            break
        elif key == 32:                    # SPACE
            if not typed_text.endswith(" "):
                typed_text += " "
        elif key in (8, ord('b'), ord('B')):  # Backspace or 'B'
            typed_text = typed_text[:-1] if typed_text else typed_text
        elif key in (ord('c'), ord('C')):
            typed_text = ""
        elif key in (ord('+'), ord('=')):
            display_scale = min(DISPLAY_MAX, display_scale + 0.05)
        elif key in (ord('-'), ord('_')):
            display_scale = max(DISPLAY_MIN, display_scale - 0.05)
        elif key in (ord('r'), ord('R')):
            smoother.reset()
            last_candidate = None
            candidate_since = None
        elif key in (ord('s'), ord('S')):
            path = save_transcript(typed_text)
            print(f"[INFO] Transcript saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    # Informative prints
    print(
        f"[INFO] TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    main()
