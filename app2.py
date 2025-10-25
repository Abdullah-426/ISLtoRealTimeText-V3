#!/usr/bin/env python3
"""
ISL v5 Tester (Segmented + Start Flag + TTA): LSTM / TCN / ENSEMBLE

Pipeline (tuned for live accuracy):
  1) WAIT_START: require BOTH hands visible for a short hold to arm the system
  2) CAPTURE: collect exactly T frames (default 48) matching your dataset layout
     - optional motion-onset gate (default ON): start the 48 when hand motion spikes
  3) PREDICT: run models (+ optional test-time augmentation) and commit if conf >= threshold
  4) COOLDOWN: 1.5s visual timer, then loop back to CAPTURE (no more start-flag needed)

Landmarks layout (matches your collector script):
  Pose 33 * [x,y,z,visibility]      -> 132
  Face 468 * [x,y,z]                -> 1404
  Left hand 21 * [x,y,z]            -> 63
  Right hand 21 * [x,y,z]           -> 63
  Total per frame                   -> 1662

Keys:
  ESC: quit
  SPACE: insert space
  B / Backspace: delete last char
  C: clear text
  +/-: zoom window
  S: save transcript
  P: pause/resume
"""

import os
import time
import json
import argparse
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

# ---------------- Feature layout (MATCH COLLECTOR) ----------------
POSE_LM = 33     # (x,y,z,visibility)
FACE_LM = 468    # (x,y,z)
HAND_LM = 21     # (x,y,z)

POSE_DIM = POSE_LM * 4        # 132
FACE_DIM = FACE_LM * 3        # 1404
L_HAND_DIM = HAND_LM * 3      # 63
R_HAND_DIM = HAND_LM * 3      # 63
FRAME_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662

# ---------------- UI / Camera ----------------
DISPLAY_SCALE = 0.95
DISPLAY_MIN, DISPLAY_MAX = 0.50, 1.30
CAM_INDEX = 0
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# ---------------- MediaPipe Holistic ----------------
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    USE_MP = True
except Exception:
    print("[ERROR] mediapipe not installed. Run: pip install mediapipe")
    USE_MP = False

# ---------------- Labels loader ----------------


def load_labels(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "classes" in obj:
        return obj["classes"]
    if isinstance(obj, dict) and "label2idx" in obj:
        return [c for c, _ in sorted(obj["label2idx"].items(), key=lambda kv: kv[1])]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unrecognized labels format: {labels_path}")

# ---------------- Helpers ----------------


def save_transcript(text):
    os.makedirs("transcripts", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("transcripts", f"isl_transcript_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def draw_bar(panel, x, y, w, h, frac, color=(0, 200, 0)):
    cv2.rectangle(panel, (x, y), (x + w, y + h), (70, 70, 70), 2)
    ww = int(w * max(0.0, min(1.0, frac)))
    cv2.rectangle(panel, (x, y), (x + ww, y + h), color, -1)


def draw_top3(panel, x, y, top3):
    for k, (lbl, c) in enumerate(top3):
        yy = y + 24 * k
        bar_w = int(280 * float(c))
        cv2.rectangle(panel, (x, yy), (x + 280, yy + 18), (45, 45, 45), 1)
        cv2.rectangle(panel, (x, yy), (x + bar_w, yy + 18), (0, 180, 0), -1)
        cv2.putText(panel, f"{lbl}: {float(c)*100:.1f}%", (x + 290, yy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 240, 220), 1, cv2.LINE_AA)


def draw_status(panel, txt, x=20, y=32, color=(0, 255, 255)):
    cv2.putText(panel, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.82, color, 2, cv2.LINE_AA)

# ---------------- Feature extraction (MATCH COLLECTOR) ----------------


def extract_keypoints(results) -> np.ndarray:
    # Pose (33*4)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        pose = np.zeros((POSE_DIM,), dtype=np.float32)
    # Face (468*3)
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        face = np.zeros((FACE_DIM,), dtype=np.float32)
    # Left hand (21*3)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        lh = np.zeros((L_HAND_DIM,), dtype=np.float32)
    # Right hand (21*3)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        rh = np.zeros((R_HAND_DIM,), dtype=np.float32)
    return np.concatenate([pose, face, lh, rh], axis=0)  # (1662,)

# ---------------- Deltas (same as training) ----------------


def add_deltas_seq(x: np.ndarray) -> np.ndarray:
    # x: (T, D) -> concat([x, dx]) where dx = [x0; x1-x0; x2-x1; ...]
    dx = np.concatenate([x[:1], x[1:] - x[:-1]], axis=0)
    return np.concatenate([x, dx], axis=-1)

# ---------------- Attention layer used in training ----------------


class TemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = tf.keras.layers.Dense(units, activation="tanh")
        self.score = tf.keras.layers.Dense(1, activation=None)
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, x, mask=None):
        s = self.proj(x)
        s = self.score(s)
        if mask is not None:
            mask_f = tf.cast(mask[:, :, None], dtype=s.dtype)
            s = s + (1.0 - mask_f) * tf.constant(-1e9, dtype=s.dtype)
        w = self.softmax(s)
        return tf.reduce_sum(x * w, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# ---------------- v5 models ----------------


def build_lstm_model(num_classes, seq_len, feat_dim,
                     lstm_w1=224, lstm_w2=128, dropout=0.45, l2_reg=1e-4):
    reg = regularizers.l2(l2_reg)
    inp = tf.keras.Input(shape=(seq_len, feat_dim), name="seq")
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_w1, return_sequences=True,
                             kernel_regularizer=reg, recurrent_regularizer=reg))(inp)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    y = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_w2, return_sequences=True,
                             kernel_regularizer=reg, recurrent_regularizer=reg))(x)
    y = tf.keras.layers.LayerNormalization()(y)
    if int(x.shape[-1]) != int(y.shape[-1]):
        x = tf.keras.layers.Dense(
            int(y.shape[-1]), activation=None, kernel_regularizer=reg)(x)
    x = tf.keras.layers.Add()([x, y])
    x = TemporalAttentionLayer(units=128, name="temporal_attention")(x)
    x = tf.keras.layers.Dense(256, activation=None, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation=None, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    out = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inp, out, name="ISL_BiLSTM_Attn_v5")


def TCNBlock(filters, kernel_size=5, dilation_base=2, n_stacks=2, dropout=0.25, l2_reg=1e-4):
    reg = regularizers.l2(l2_reg)

    def f(x):
        for s in range(n_stacks):
            dil = dilation_base ** s
            y = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                       dilation_rate=dil, kernel_regularizer=reg)(x)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.Activation("relu")(y)
            y = tf.keras.layers.Dropout(dropout)(y)
            if int(x.shape[-1]) != int(y.shape[-1]):
                x = tf.keras.layers.Conv1D(
                    filters, 1, padding="same", kernel_regularizer=reg)(x)
            x = tf.keras.layers.Add()([x, y])
        return x
    return f


def build_tcn_model(num_classes, seq_len, feat_dim, dropout=0.45, l2_reg=1e-4):
    inp = tf.keras.Input(shape=(seq_len, feat_dim), name="seq")
    x = tf.keras.layers.Dense(256, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = TCNBlock(256, kernel_size=5, dilation_base=2,
                 n_stacks=3, dropout=0.25, l2_reg=l2_reg)(x)
    x = TCNBlock(256, kernel_size=3, dilation_base=2,
                 n_stacks=2, dropout=0.25, l2_reg=l2_reg)(x)
    x = TemporalAttentionLayer(units=128, name="temporal_attention")(x)
    x = tf.keras.layers.Dense(256, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    out = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inp, out, name="ISL_TCN_v5")

# ---------------- Models loader ----------------


def load_models(mode, labels_path, add_deltas, lstm_weights, tcn_weights,
                lstm_w1=224, lstm_w2=128, lstm_dropout=0.45, lstm_l2=1e-4,
                tcn_dropout=0.45, tcn_l2=1e-4, seq_len=48):
    classes = load_labels(labels_path)
    num_classes = len(classes)
    feat_dim = FRAME_DIM * (2 if add_deltas else 1)

    lstm_model = None
    tcn_model = None

    if mode in ("lstm", "ensemble"):
        if not lstm_weights:
            raise SystemExit(
                "[ERROR] --lstm_weights is required for mode lstm/ensemble")
        lstm_model = build_lstm_model(num_classes, seq_len, feat_dim,
                                      lstm_w1=lstm_w1, lstm_w2=lstm_w2,
                                      dropout=lstm_dropout, l2_reg=lstm_l2)
        lstm_model.load_weights(lstm_weights)
        print(f"[OK] Loaded LSTM weights: {lstm_weights}")

    if mode in ("tcn", "ensemble"):
        if not tcn_weights:
            raise SystemExit(
                "[ERROR] --tcn_weights is required for mode tcn/ensemble")
        tcn_model = build_tcn_model(num_classes, seq_len, feat_dim,
                                    dropout=tcn_dropout, l2_reg=tcn_l2)
        tcn_model.load_weights(tcn_weights)
        print(f"[OK] Loaded TCN weights: {tcn_weights}")

    return classes, lstm_model, tcn_model

# ---------------- TTA (temporal jitter) ----------------


def tta_variants(x, kind="shift3"):
    """
    x: (T,D) base clip
    Returns list of (T,D) variants for test-time augmentation.
    - 'none'   : [x]
    - 'shift3' : shifts {-2, 0, +2} with edge padding
    - 'shift5' : shifts {-4,-2,0,+2,+4}
    """
    T, D = x.shape

    def shift_clip(delta):
        if delta == 0:
            return x
        if delta > 0:
            # pad first frames, drop last
            pad = np.repeat(x[:1], delta, axis=0)
            return np.concatenate([pad, x[:-delta]], axis=0)
        else:
            delta = -delta
            pad = np.repeat(x[-1:], delta, axis=0)
            return np.concatenate([x[delta:], pad], axis=0)

    if kind == "none":
        return [x]
    if kind == "shift3":
        return [shift_clip(-2), x, shift_clip(+2)]
    if kind == "shift5":
        return [shift_clip(-4), shift_clip(-2), x, shift_clip(+2), shift_clip(+4)]
    return [x]

# ---------------- Simple hand-motion energy (for onset gate) ----------------


def hand_motion(prev_vec, curr_vec):
    """Mean |delta| over hand dims only (last 126 features)."""
    if prev_vec is None or curr_vec is None:
        return 0.0
    lh_start = POSE_DIM + FACE_DIM
    rh_start = lh_start + L_HAND_DIM
    hands_prev = np.concatenate([prev_vec[lh_start:lh_start+L_HAND_DIM],
                                 prev_vec[rh_start:rh_start+R_HAND_DIM]])
    hands_curr = np.concatenate([curr_vec[lh_start:lh_start+L_HAND_DIM],
                                 curr_vec[rh_start:rh_start+R_HAND_DIM]])
    return float(np.mean(np.abs(hands_curr - hands_prev)))

# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser(
        description="ISL v5 segmented tester (start-flag, motion-gate, TTA)")
    ap.add_argument("--mode", type=str,
                    choices=["lstm", "tcn", "ensemble"], default="tcn")
    ap.add_argument("--labels", type=str, required=True,
                    help="labels.json path")
    ap.add_argument("--lstm_weights", type=str, default=None)
    ap.add_argument("--tcn_weights", type=str, default=None)
    ap.add_argument("--add_deltas", action="store_true",
                    help="Use if trained with --add_deltas")
    ap.add_argument("--frames", type=int, default=48,
                    help="Frames per segment (seq_len)")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="Sampling fps during CAPTURE; 0=every frame")
    ap.add_argument("--segment_cooldown", type=float, default=1.5,
                    help="Cooldown seconds between segments")
    ap.add_argument("--conf", type=float, default=0.60,
                    help="Min prob to accept top1; else skip committing")
    ap.add_argument("--auto_space", action="store_true",
                    help="Append a space after each committed label")

    # LSTM/TCN hyperparams (keep default as in training)
    ap.add_argument("--lstm_w1", type=int, default=224)
    ap.add_argument("--lstm_w2", type=int, default=128)
    ap.add_argument("--lstm_dropout", type=float, default=0.45)
    ap.add_argument("--lstm_l2", type=float, default=1e-4)
    ap.add_argument("--tcn_dropout", type=float, default=0.45)
    ap.add_argument("--tcn_l2", type=float, default=1e-4)

    # TTA
    ap.add_argument("--tta", type=str,
                    choices=["none", "shift3", "shift5"], default="shift3")

    # Motion gate
    ap.add_argument("--gate_motion", dest="gate_motion",
                    action="store_true", help="Start capture on motion onset")
    ap.add_argument("--no_gate_motion", dest="gate_motion",
                    action="store_false")
    ap.set_defaults(gate_motion=True)
    ap.add_argument("--motion_thresh", type=float, default=0.0035,
                    help="Mean |Δ| over hand dims to trigger onset")
    ap.add_argument("--motion_warmup", type=int, default=2,
                    help="Require motion threshold for N out of last M frames")
    ap.add_argument("--motion_window", type=int, default=4,
                    help="Window M for N-out-of-M motion onset")

    # Start flag (both-hands) config
    ap.add_argument("--start_hold_frames", type=int, default=8,
                    help="Consecutive frames with both hands to arm start")

    # Camera/display
    ap.add_argument("--cam", type=int, default=CAM_INDEX)
    ap.add_argument("--width", type=int, default=CAP_WIDTH)
    ap.add_argument("--height", type=int, default=CAP_HEIGHT)
    ap.add_argument("--scale", type=float, default=DISPLAY_SCALE)

    # Holistic thresholds
    ap.add_argument("--det_conf", type=float, default=0.50)
    ap.add_argument("--trk_conf", type=float, default=0.50)
    ap.add_argument("--model_complexity", type=int, default=1)
    args = ap.parse_args()

    print(
        f"[INFO] TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    if not USE_MP:
        return

    # Load models
    classes, lstm_model, tcn_model = load_models(
        mode=args.mode,
        labels_path=args.labels,
        add_deltas=args.add_deltas,
        lstm_weights=args.lstm_weights,
        tcn_weights=args.tcn_weights,
        lstm_w1=args.lstm_w1, lstm_w2=args.lstm_w2,
        lstm_dropout=args.lstm_dropout, lstm_l2=args.lstm_l2,
        tcn_dropout=args.tcn_dropout, tcn_l2=args.tcn_l2,
        seq_len=args.frames
    )
    num_classes = len(classes)
    print(f"[INFO] #classes={num_classes}")

    # Camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cv2.namedWindow("ISL v5 (Segmented+TTA)", cv2.WINDOW_NORMAL)
    display_scale = float(np.clip(args.scale, DISPLAY_MIN, DISPLAY_MAX))

    # States
    STATE_WAIT_START = 0
    STATE_CAPTURE = 1
    STATE_PREDICT = 2
    STATE_COOLDOWN = 3
    state = STATE_WAIT_START
    paused = False

    seq = []
    target_gap = 0.0 if args.fps <= 0 else 1.0 / args.fps
    last_sample_t = time.time()

    typed_text = ""
    last_probs = None
    last_top3 = None
    last_label = None

    cooldown_t0 = None

    # Start flag counters
    both_hands_count = 0
    armed_once = False

    # Motion gate buffers
    prev_vec = None
    motion_hist = deque(maxlen=max(1, args.motion_window))

    # FPS tracker
    fps_hist = deque(maxlen=30)
    last_t = time.time()

    with mp_holistic.Holistic(
        model_complexity=args.model_complexity,
        refine_face_landmarks=False,  # ensures 468 face pts
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.trk_conf
    ) as holistic:

        print("[INFO] Controls: ESC quit | Space space | B backspace | C clear | +/- zoom | S save | P pause/resume")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Process landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            rgb.flags.writeable = True

            # Draw landmarks (safe)
            panel = frame.copy()
            try:
                if res.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        panel, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_styles.get_default_pose_landmarks_style()
                    )
                if res.face_landmarks:
                    mp_drawing.draw_landmarks(
                        panel, res.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        panel, res.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                    )
                if res.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        panel, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                if res.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        panel, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
            except Exception:
                pass

            # HUD constants
            header_x, header_y = 20, 32
            bar_x, bar_y, bar_w, bar_h = 20, 60, 440, 18

            # Keypoints for this frame (for gating/capture)
            curr_vec = extract_keypoints(res)

            if paused:
                draw_status(panel, "PAUSED - press P to resume",
                            header_x, header_y, (0, 200, 255))
            else:
                if state == STATE_WAIT_START:
                    # Require both hands visible for N consecutive frames
                    lh_on = res.left_hand_landmarks is not None
                    rh_on = res.right_hand_landmarks is not None
                    if lh_on and rh_on:
                        both_hands_count += 1
                    else:
                        both_hands_count = 0

                    draw_status(panel, "SHOW BOTH HANDS to start",
                                header_x, header_y, (0, 255, 255))
                    draw_bar(panel, bar_x, bar_y, bar_w, bar_h,
                             frac=min(1.0, both_hands_count /
                                      max(1, args.start_hold_frames)),
                             color=(0, 200, 0))

                    if both_hands_count >= args.start_hold_frames:
                        both_hands_count = 0
                        armed_once = True
                        seq = []
                        prev_vec = None
                        motion_hist.clear()
                        state = STATE_CAPTURE
                        last_sample_t = time.time()

                elif state == STATE_CAPTURE:
                    # Display capture info
                    label_txt = "CAPTURE"
                    if args.gate_motion:
                        label_txt += " (waiting motion)" if len(
                            seq) == 0 else " (recording)"
                    draw_status(panel, f"{label_txt}: frames {len(seq)}/{args.frames}",
                                header_x, header_y, (0, 255, 255))
                    draw_bar(panel, bar_x, bar_y, bar_w, bar_h,
                             frac=len(seq) / float(args.frames), color=(0, 200, 0))

                    # Motion onset gate (hands only)
                    ready_to_add = True
                    if args.gate_motion:
                        m = hand_motion(prev_vec, curr_vec)
                        motion_hist.append(1 if m >= args.motion_thresh else 0)
                        enough_motion = (sum(motion_hist) >=
                                         args.motion_warmup)
                        # Only start filling seq when we see enough motion at least once
                        if len(seq) == 0 and not enough_motion:
                            ready_to_add = False
                    # FPS sampling gate
                    now = time.time()
                    if ready_to_add and (args.fps <= 0 or (now - last_sample_t) >= target_gap):
                        last_sample_t = now
                        seq.append(curr_vec)

                    prev_vec = curr_vec

                    if len(seq) >= args.frames:
                        state = STATE_PREDICT

                elif state == STATE_PREDICT:
                    draw_status(panel, "PREDICTING...",
                                header_x, header_y, (0, 255, 0))
                    # Build base clip
                    x = np.stack(seq[:args.frames], axis=0).astype(
                        np.float32)  # (T,D)
                    if args.add_deltas:
                        x = add_deltas_seq(x)  # (T,2D)

                    # TTA variants
                    variants = tta_variants(
                        x, kind=args.tta)  # list of (T,Dor2D)
                    X = np.stack(variants, axis=0)             # (N,T,D)

                    # Predict across models & TTA
                    probs_acc = None
                    n_eval = 0

                    def model_predict(mdl):
                        nonlocal probs_acc, n_eval
                        if mdl is None:
                            return
                        p = mdl.predict(X, verbose=0)        # (N, C)
                        # (C,) average across TTA
                        p = np.mean(p, axis=0)
                        probs_acc = p if probs_acc is None else (probs_acc + p)
                        n_eval += 1

                    model_predict(lstm_model)
                    model_predict(tcn_model)
                    probs = probs_acc / float(max(1, n_eval))
                    last_probs = probs

                    order = np.argsort(-probs)
                    top1, top2, top3 = int(order[0]), int(
                        order[1]), int(order[2])
                    last_top3 = [(classes[top1], float(probs[top1])),
                                 (classes[top2], float(probs[top2])),
                                 (classes[top3], float(probs[top3]))]
                    last_label = classes[top1]

                    # Commit if confident
                    if float(probs[top1]) >= args.conf:
                        typed_text += last_label
                        if args.auto_space and not typed_text.endswith(" "):
                            typed_text += " "

                    # Move to cooldown
                    cooldown_t0 = time.time()
                    state = STATE_COOLDOWN
                    seq = []
                    motion_hist.clear()
                    prev_vec = None

                elif state == STATE_COOLDOWN:
                    elapsed = time.time() - cooldown_t0 if cooldown_t0 else 0.0
                    remain = max(0.0, args.segment_cooldown - elapsed)
                    draw_status(panel, f"COOLDOWN: {remain:.1f}s  (get ready for next sign)",
                                header_x, header_y, (180, 255, 180))
                    frac = (args.segment_cooldown - remain) / \
                        max(1e-6, args.segment_cooldown)
                    draw_bar(panel, bar_x, bar_y, bar_w,
                             bar_h, frac, color=(0, 180, 255))
                    if last_top3 is not None:
                        draw_top3(panel, bar_x + bar_w +
                                  20, bar_y - 2, last_top3)
                    if remain <= 0.0:
                        state = STATE_CAPTURE
                        cooldown_t0 = None
                        last_sample_t = time.time()
                        # After first arming, we do NOT require both hands again
                        if not armed_once:
                            state = STATE_WAIT_START

            # Typed text bar
            cv2.rectangle(panel, (20, h - 90),
                          (w - 20, h - 30), (30, 30, 30), -1)
            cv2.putText(panel, f"Typed: {typed_text}", (30, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS
            t = time.time()
            fps = 1.0 / max(1e-6, (t - last_t))
            last_t = t
            fps_hist.append(fps)
            fps_avg = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
            cv2.putText(panel, f"FPS: {fps_avg:.1f}", (w - 160, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Show
            disp = cv2.resize(panel, None, fx=display_scale, fy=display_scale,
                              interpolation=cv2.INTER_AREA) if abs(display_scale - 1.0) > 1e-3 else panel
            cv2.imshow("ISL v5 (Segmented+TTA)", disp)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if not typed_text.endswith(" "):
                    typed_text += " "
            elif key in (8, ord('b'), ord('B')):
                typed_text = typed_text[:-1] if typed_text else typed_text
            elif key in (ord('c'), ord('C')):
                typed_text = ""
            elif key in (ord('+'), ord('=')):
                display_scale = min(DISPLAY_MAX, display_scale + 0.05)
            elif key in (ord('-'), ord('_')):
                display_scale = max(DISPLAY_MIN, display_scale - 0.05)
            elif key in (ord('s'), ord('S')):
                path = save_transcript(typed_text)
                print(f"[INFO] Transcript saved: {path}")
            elif key in (ord('p'), ord('P')):
                paused = not paused
                if not paused:
                    last_sample_t = time.time()
                    if state == STATE_COOLDOWN:
                        cooldown_t0 = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
