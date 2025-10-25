#!/usr/bin/env python3
"""
app4.py — ISL v5 Real-time Tester (High-Accuracy Segmented Inference)

Why this works better:
- EXACT same feature layout as your collector (pose→face→left hand→right hand)
- Deterministic 48-frame segments @ ~20 FPS, with PRE-ROLL so the motion apex is inside the clip
- Optional ADAPTIVE motion onset gate (or disable to verify)
- Segment QUALITY gate (min frames with hands + min motion) to avoid bad commits
- Test-Time Aug (shift3/shift5) with max/mean pooling
- Weighted ENSEMBLE (TCN favored by default)
- Optional debug dump of captured segments (pre/post-deltas) for offline eval

Keys:
  ESC: quit
  SPACE: append space
  B/Backspace: delete last char
  C: clear text
  S: save transcript
  P: pause/resume
  +/-: zoom window
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

# ---------------- Feature layout (MATCHES COLLECTOR) ----------------
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


# ---------------- Utility ----------------
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
        bar_w = int(300 * float(c))
        cv2.rectangle(panel, (x, yy), (x + 300, yy + 18), (45, 45, 45), 1)
        cv2.rectangle(panel, (x, yy), (x + bar_w, yy + 18), (0, 180, 0), -1)
        cv2.putText(panel, f"{lbl}: {float(c)*100:.1f}%", (x + 310, yy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 240, 225), 1, cv2.LINE_AA)


def draw_status(panel, txt, x=20, y=32, color=(0, 255, 255)):
    cv2.putText(panel, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.82, color, 2, cv2.LINE_AA)


# ---------------- Feature extraction (MATCH COLLECTOR) ----------------
def extract_keypoints(results):
    """Return (vector1662, hand_present_bool)."""
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

    v = np.concatenate([pose, face, lh, rh], axis=0)
    hands_present = (results.left_hand_landmarks is not None) or (
        results.right_hand_landmarks is not None)
    return v, hands_present


# ---------------- Deltas ----------------
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


# ---------------- v5 models (same shapes as training) ----------------
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


# ---------------- TTA ----------------
def tta_variants(x, kind="shift3"):
    """
    x: (T,D) base clip
    Returns list of (T,D) variants for test-time augmentation.
    - 'none'   : [x]
    - 'shift3' : {-2, 0, +2}
    - 'shift5' : {-4, -2, 0, +2, +4}
    """
    T, D = x.shape

    def shift_clip(delta):
        if delta == 0:
            return x
        if delta > 0:
            pad = np.repeat(x[:1], delta, axis=0)
            return np.concatenate([pad, x[:-delta]], axis=0)
        else:
            d = -delta
            pad = np.repeat(x[-1:], d, axis=0)
            return np.concatenate([x[d:], pad], axis=0)

    if kind == "none":
        return [x]
    if kind == "shift3":
        return [shift_clip(-2), x, shift_clip(+2)]
    if kind == "shift5":
        return [shift_clip(-4), shift_clip(-2), x, shift_clip(+2), shift_clip(+4)]
    return [x]


def pool_probs(P, pool="max"):
    """P: (N, C) -> (C,) pooled across TTA variants."""
    if pool == "max":
        return np.max(P, axis=0)
    return np.mean(P, axis=0)


# ---------------- Motion / Quality ----------------
def hands_motion_energy(prev_vec, curr_vec):
    """Mean |Δ| over hand dims only (last 126 features)."""
    if prev_vec is None or curr_vec is None:
        return 0.0
    lh_start = POSE_DIM + FACE_DIM
    rh_start = lh_start + L_HAND_DIM
    hands_prev = np.concatenate([prev_vec[lh_start:lh_start+L_HAND_DIM],
                                 prev_vec[rh_start:rh_start+R_HAND_DIM]])
    hands_curr = np.concatenate([curr_vec[lh_start:lh_start+L_HAND_DIM],
                                 curr_vec[rh_start:rh_start+R_HAND_DIM]])
    return float(np.mean(np.abs(hands_curr - hands_prev)))


def segment_quality(hand_present_flags, motion_values, min_hand_frames, min_motion_mean):
    """Simple quality gate: enough frames with hands + enough average motion."""
    hp = int(np.sum(hand_present_flags))
    mv = float(np.mean(motion_values)) if len(motion_values) else 0.0
    ok_hands = hp >= int(min_hand_frames)
    ok_motion = mv >= float(min_motion_mean)
    return ok_hands and ok_motion, hp, mv


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


# ---------------- Commit rule ----------------
def should_commit(probs, conf_hi, conf_lo, margin):
    """Commit if p1 >= conf_hi, or (p1 >= conf_lo and p1 - p2 >= margin)."""
    order = np.argsort(-probs)
    p1, p2 = float(probs[order[0]]), float(
        probs[order[1]] if len(order) > 1 else 0.0)
    if p1 >= conf_hi:
        return True
    if p1 >= conf_lo and (p1 - p2) >= margin:
        return True
    return False


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="ISL v5 segmented tester (high-accuracy)")
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

    # Confidence rules
    ap.add_argument("--conf", type=float, default=0.70,
                    help="High threshold for commit")
    ap.add_argument("--conf_lo", type=float, default=0.55,
                    help="Low threshold if margin is large")
    ap.add_argument("--margin", type=float, default=0.20,
                    help="p1 - p2 margin for low-threshold commit")
    ap.add_argument("--auto_space", action="store_true",
                    help="Append a space after each committed label")

    # LSTM/TCN hyperparams (must match training)
    ap.add_argument("--lstm_w1", type=int, default=224)
    ap.add_argument("--lstm_w2", type=int, default=128)
    ap.add_argument("--lstm_dropout", type=float, default=0.45)
    ap.add_argument("--lstm_l2", type=float, default=1e-4)
    ap.add_argument("--tcn_dropout", type=float, default=0.45)
    ap.add_argument("--tcn_l2", type=float, default=1e-4)

    # Ensemble weights
    ap.add_argument("--ens_w_tcn", type=float, default=0.8,
                    help="Weight for TCN in ensemble")
    ap.add_argument("--ens_w_lstm", type=float, default=0.2,
                    help="Weight for LSTM in ensemble")

    # TTA
    ap.add_argument("--tta", type=str,
                    choices=["none", "shift3", "shift5"], default="shift3")
    ap.add_argument("--tta_pool", type=str,
                    choices=["mean", "max"], default="max")

    # Motion gate
    ap.add_argument("--gate_motion", dest="gate_motion", action="store_true",
                    help="Start capture on motion onset")
    ap.add_argument("--no_gate_motion", dest="gate_motion",
                    action="store_false")
    ap.set_defaults(gate_motion=True)
    ap.add_argument("--motion_thresh", type=float, default=0.0020,
                    help="Mean |Δ| (hands) to trigger onset")
    ap.add_argument("--motion_window", type=int, default=4,
                    help="N-frame window for onset")
    ap.add_argument("--pre_roll", type=int, default=6,
                    help="Frames to include before onset")

    # Quality gate
    ap.add_argument("--min_hand_frames", type=int, default=12,
                    help="Min frames with any hand visible")
    ap.add_argument("--min_motion_mean", type=float,
                    default=0.0009, help="Min avg motion over segment")

    # Start flag (both hands once)
    ap.add_argument("--start_hold_frames", type=int, default=8,
                    help="Frames with both hands to arm")

    # Camera/display
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=CAP_WIDTH)
    ap.add_argument("--height", type=int, default=CAP_HEIGHT)
    ap.add_argument("--scale", type=float, default=DISPLAY_SCALE)

    # Holistic thresholds
    ap.add_argument("--det_conf", type=float, default=0.50)
    ap.add_argument("--trk_conf", type=float, default=0.50)
    ap.add_argument("--model_complexity", type=int, default=1)

    # Debug dump
    ap.add_argument("--dump_dir", type=str, default=None,
                    help="If set, save each captured segment here")

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

    cv2.namedWindow("ISL v5 (Segmented)", cv2.WINDOW_NORMAL)
    display_scale = float(np.clip(args.scale, DISPLAY_MIN, DISPLAY_MAX))

    # States
    STATE_WAIT_START = 0
    STATE_CAPTURE = 1
    STATE_PREDICT = 2
    STATE_COOLDOWN = 3

    state = STATE_WAIT_START
    paused = False

    # Timing (stable ~20 fps sampler)
    target_gap = 0.0 if args.fps <= 0 else 1.0 / args.fps
    last_sample_t = time.time()

    # Segment buffers
    seq = []                     # vectors
    hand_flags = []              # bool per frame
    motion_vals = []             # per-frame |Δ| over hands
    prebuf = deque(maxlen=max(0, args.pre_roll))

    prev_vec = None
    last_probs = None
    last_top3 = None

    # Start gating
    both_hands_count = 0
    armed_once = False

    # COOLDOWN
    cooldown_t0 = None

    # FPS
    fps_hist = deque(maxlen=30)
    last_t = time.time()

    # Dump dir
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    with mp_holistic.Holistic(
        model_complexity=args.model_complexity,
        refine_face_landmarks=False,  # EXACTLY 468 points like collector
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.trk_conf
    ) as holistic:

        print("[INFO] Controls: ESC quit | Space space | B backspace | C clear | +/- zoom | S save | P pause/resume")

        typed_text = ""

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

            # Draw landmarks (robust face drawing using face_mesh)
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

            # HUD anchors
            header_x, header_y = 20, 32
            bar_x, bar_y, bar_w, bar_h = 20, 60, 520, 18

            # Current vector & hand presence
            curr_vec, hands_present = extract_keypoints(res)

            if paused:
                draw_status(panel, "PAUSED - press P to resume",
                            header_x, header_y, (0, 200, 255))
            else:
                if state == STATE_WAIT_START:
                    # Require both hands visible for N consecutive frames
                    lh_on = res.left_hand_landmarks is not None
                    rh_on = res.right_hand_landmarks is not None
                    both_hands_count = both_hands_count + \
                        1 if (lh_on and rh_on) else 0

                    draw_status(panel, "SHOW BOTH HANDS to start",
                                header_x, header_y, (0, 255, 255))
                    draw_bar(panel, bar_x, bar_y, bar_w, bar_h,
                             frac=min(1.0, both_hands_count /
                                      max(1, args.start_hold_frames)),
                             color=(0, 200, 0))

                    # keep a short baseline of prebuf motion noise
                    if args.pre_roll > 0:
                        prebuf.append(curr_vec)

                    if both_hands_count >= args.start_hold_frames:
                        both_hands_count = 0
                        armed_once = True
                        seq.clear()
                        hand_flags.clear()
                        motion_vals.clear()
                        prebuf.clear()
                        prev_vec = None
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

                    # Motion (hands only)
                    m = hands_motion_energy(prev_vec, curr_vec)
                    prev_vec = curr_vec

                    # Sampler at target FPS
                    now = time.time()
                    if args.fps <= 0 or (now - last_sample_t) >= target_gap:
                        last_sample_t = now

                        # If not yet started filling seq, honor motion gate with pre-roll
                        if args.gate_motion and len(seq) == 0:
                            prebuf.append(curr_vec)
                            # Onset condition: average motion over recent window
                            if m >= args.motion_thresh:
                                # prepend pre-roll frames
                                # (we already have 'curr_vec' in prebuf)
                                while len(prebuf) > args.pre_roll:
                                    prebuf.popleft()
                                for v in list(prebuf):
                                    seq.append(v)
                                    hand_flags.append(hands_present)
                                    # neutral for pre-roll
                                    motion_vals.append(0.0)
                                prebuf.clear()
                        else:
                            # Already started or gate disabled: just push
                            seq.append(curr_vec)
                            hand_flags.append(hands_present)
                            motion_vals.append(m)

                    # When enough frames, go predict
                    if len(seq) >= args.frames:
                        state = STATE_PREDICT

                elif state == STATE_PREDICT:
                    draw_status(panel, "PREDICTING...",
                                header_x, header_y, (0, 255, 0))

                    # Build 48-frame clip
                    x = np.stack(seq[:args.frames], axis=0).astype(
                        np.float32)  # (T,D)

                    # Quality gate — avoid committing junk
                    ok, hand_cnt, mv = segment_quality(
                        hand_present_flags=np.array(
                            hand_flags[:args.frames], dtype=bool),
                        motion_values=np.array(
                            motion_vals[:args.frames], dtype=np.float32),
                        min_hand_frames=args.min_hand_frames,
                        min_motion_mean=args.min_motion_mean
                    )

                    # Add deltas if requested
                    x_in = add_deltas_seq(x) if args.add_deltas else x

                    # TTA variants
                    variants = tta_variants(
                        x_in, kind=args.tta)  # list (T, D_or_2D)
                    X = np.stack(variants, axis=0)                # (N, T, D)

                    # Predict across models & TTA
                    probs_acc = None
                    wsum = 0.0

                    def predict_with(mdl, w):
                        nonlocal probs_acc, wsum
                        if mdl is None or w <= 0:
                            return
                        p = mdl.predict(X, verbose=0)  # (N, C)
                        p = pool_probs(p, pool=args.tta_pool)  # (C,)
                        if probs_acc is None:
                            probs_acc = w * p
                        else:
                            probs_acc = probs_acc + w * p
                        wsum += w

                    if args.mode == "ensemble":
                        predict_with(tcn_model, args.ens_w_tcn)
                        predict_with(lstm_model, args.ens_w_lstm)
                    elif args.mode == "tcn":
                        predict_with(tcn_model, 1.0)
                    else:
                        predict_with(lstm_model, 1.0)

                    if wsum <= 0:
                        probs = np.zeros((len(classes),), dtype=np.float32)
                    else:
                        probs = probs_acc / wsum

                    order = np.argsort(-probs)
                    top_idx = int(order[0])
                    last_probs = probs
                    last_top3 = [(classes[i], float(probs[i]))
                                 for i in order[:3]]
                    top_label = classes[top_idx]
                    commit_ok = ok and should_commit(
                        probs, conf_hi=args.conf, conf_lo=args.conf_lo, margin=args.margin)

                    # Dump for offline debug
                    if args.dump_dir:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        base = os.path.join(args.dump_dir, f"{ts}_{top_label}")
                        np.save(base + "_x.npy", x)      # pre-delta (T,1662)
                        if args.add_deltas:
                            # post-delta (T,3324)
                            np.save(base + "_xd.npy", x_in)
                        meta = {
                            "top3": [(lbl, float(p)) for lbl, p in last_top3] if last_top3 else None,
                            "hands_frames": int(np.sum(hand_flags[:args.frames])),
                            "motion_mean": float(np.mean(motion_vals[:args.frames])) if motion_vals else 0.0,
                            "ok_quality": bool(ok)
                        }
                        with open(base + "_meta.json", "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)

                    # Commit
                    if commit_ok:
                        typed_text += top_label
                        if args.auto_space and not typed_text.endswith(" "):
                            typed_text += " "
                    else:
                        # Show reason
                        reason = []
                        if not ok:
                            reason.append("low-quality segment")
                        if not should_commit(probs, args.conf, args.conf_lo, args.margin):
                            reason.append("low confidence")
                        draw_status(panel, "SKIPPED: " + ", ".join(reason) if reason else "SKIPPED",
                                    header_x, header_y + 26, (40, 220, 255))

                    # Reset for next loop
                    cooldown_t0 = time.time()
                    state = STATE_COOLDOWN
                    seq.clear()
                    hand_flags.clear()
                    motion_vals.clear()
                    prebuf.clear()
                    prev_vec = None

                elif state == STATE_COOLDOWN:
                    elapsed = time.time() - cooldown_t0 if cooldown_t0 else 0.0
                    remain = max(0.0, args.segment_cooldown - elapsed)
                    draw_status(
                        panel, f"COOLDOWN: {remain:.1f}s  (get ready)", header_x, header_y, (180, 255, 180))
                    frac = (args.segment_cooldown - remain) / \
                        max(1e-6, args.segment_cooldown)
                    draw_bar(panel, bar_x, bar_y, bar_w,
                             bar_h, frac, color=(0, 180, 255))

                    if last_top3 is not None:
                        draw_top3(panel, bar_x + bar_w +
                                  24, bar_y - 2, last_top3)

                    if remain <= 0.0:
                        state = STATE_CAPTURE
                        cooldown_t0 = None
                        last_sample_t = time.time()
                        # After first arming, we don't require both hands again
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
            cv2.imshow("ISL v5 (Segmented)", disp)

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
    # (Optional) make TF CPU thread usage reasonable on Windows
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    main()


""" 
1) TCN (recommended baseline)
python app4.py --mode tcn `
  --tcn_weights "models/isl_v5_tcn_deltas/best.weights.h5" `
  --labels "models/isl_v5_tcn_deltas/labels.json" `
  --add_deltas `
  --fps 20 --frames 48 `
  --tta shift3 --tta_pool max `
  --conf 0.70 --conf_lo 0.55 --margin 0.20 `
  --gate_motion --motion_thresh 0.0020 --pre_roll 6 `
  --min_hand_frames 12 --min_motion_mean 0.0009 `
  --segment_cooldown 1.5 `
  --auto_space `
  --dump_dir "debug_captures_tcn"

2) LSTM (BETTER)
python app4.py --mode lstm `
  --lstm_weights "models/isl_v5_lstm_mild_aw_deltas/best.weights.h5" `
  --labels "models/isl_v5_tcn_deltas/labels.json" `
  --add_deltas `
  --fps 20 --frames 48 `
  --tta shift3 --tta_pool max `
  --conf 0.70 --conf_lo 0.55 --margin 0.20 `
  --gate_motion --motion_thresh 0.0020 --pre_roll 6 `
  --min_hand_frames 12 --min_motion_mean 0.0009 `
  --segment_cooldown 1.5 `
  --auto_space `
  --dump_dir "debug_captures_lstm"

3) ENSEMBLE (TCN-weighted)
python app4.py --mode ensemble `
  --lstm_weights "models/isl_v5_lstm_mild_aw_deltas/best.weights.h5" `
  --tcn_weights  "models/isl_v5_tcn_deltas/best.weights.h5" `
  --labels "models/isl_v5_tcn_deltas/labels.json" `
  --add_deltas `
  --fps 20 --frames 48 `
  --tta shift3 --tta_pool max `
  --ens_w_tcn 0.8 --ens_w_lstm 0.2 `
  --conf 0.70 --conf_lo 0.55 --margin 0.20 `
  --gate_motion --motion_thresh 0.0020 --pre_roll 6 `
  --min_hand_frames 12 --min_motion_mean 0.0009 `
  --segment_cooldown 1.5 `
  --auto_space `
  --dump_dir "debug_captures_ens"
  """
