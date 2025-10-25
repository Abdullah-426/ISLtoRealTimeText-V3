#!/usr/bin/env python3
"""
ISL v5 Live (Highest-Accuracy) — LSTM / TCN / ENSEMBLE

Design for accuracy:
 - Matches collector features exactly (pose 33*4, face 468*3, LH 21*3, RH 21*3 = 1662)
 - 20 FPS * 48 frames window, two-hands start flag, motion-onset gate (adaptive default)
 - Strong TTA (shift ±2; optional speed ±10%) with max pooling
 - Weighted ENSEMBLE (default TCN 0.6, LSTM 0.4)
 - Hand presence ratio check; low-confidence quick retry before "?"

Keys:
  ESC quit | Space insert space | B/Backspace delete | C clear | +/- zoom | S save | P pause
"""

import os
import time
import json
import argparse
from collections import deque
import numpy as np
import cv2
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

# ---------------- Labels ----------------


def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "classes" in obj:
        return obj["classes"]
    if isinstance(obj, dict) and "label2idx" in obj:
        return [c for c, _ in sorted(obj["label2idx"].items(), key=lambda kv: kv[1])]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unrecognized labels format: {path}")

# ---------------- File helpers ----------------


def save_transcript(text):
    os.makedirs("transcripts", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    p = os.path.join("transcripts", f"isl_transcript_{stamp}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p

# ---------------- Drawing helpers ----------------


def draw_status(panel, txt, x=18, y=30, color=(0, 255, 255)):
    cv2.putText(panel, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.78, color, 2, cv2.LINE_AA)


def draw_bar(panel, x, y, w, h, frac, color=(0, 200, 0)):
    cv2.rectangle(panel, (x, y), (x+w, y+h), (70, 70, 70), 2)
    ww = int(np.clip(frac, 0.0, 1.0)*w)
    cv2.rectangle(panel, (x, y), (x+ww, y+h), color, -1)


def draw_top3(panel, x, y, top3):
    for k, (lbl, c) in enumerate(top3):
        yy = y + 24*k
        bar_w = int(280*float(c))
        cv2.rectangle(panel, (x, yy), (x+280, yy+18), (45, 45, 45), 1)
        cv2.rectangle(panel, (x, yy), (x+bar_w, yy+18), (0, 180, 0), -1)
        cv2.putText(panel, f"{lbl}: {float(c)*100:.1f}%", (x+290, yy+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 240, 220), 1, cv2.LINE_AA)

# ---------------- Feature extraction (MATCH COLLECTOR) ----------------


def extract_keypoints(res) -> np.ndarray:
    if res.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        pose = np.zeros((POSE_DIM,), dtype=np.float32)
    if res.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in res.face_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        face = np.zeros((FACE_DIM,), dtype=np.float32)
    if res.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in res.left_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        lh = np.zeros((L_HAND_DIM,), dtype=np.float32)
    if res.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in res.right_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        rh = np.zeros((R_HAND_DIM,), dtype=np.float32)
    return np.concatenate([pose, face, lh, rh], axis=0)


def hands_present(res) -> int:
    return int(res.left_hand_landmarks is not None or res.right_hand_landmarks is not None)


def two_hands_present(res) -> bool:
    return (res.left_hand_landmarks is not None) and (res.right_hand_landmarks is not None)

# ---------------- Deltas ----------------


def add_deltas_seq(x: np.ndarray) -> np.ndarray:
    dx = np.concatenate([x[:1], x[1:] - x[:-1]], axis=0)
    return np.concatenate([x, dx], axis=-1)

# ---------------- Attention + Models (v5) ----------------


class TemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, **kw):
        super().__init__(**kw)
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
        return tf.reduce_sum(x*w, axis=1)

    def get_config(self):
        c = super().get_config()
        c.update({"units": self.units})
        return c


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

# ---------------- Load models ----------------


def load_models(mode, labels_path, add_deltas, lstm_weights, tcn_weights,
                lstm_w1=224, lstm_w2=128, lstm_dropout=0.45, lstm_l2=1e-4,
                tcn_dropout=0.45, tcn_l2=1e-4, seq_len=48):
    classes = load_labels(labels_path)
    feat_dim = FRAME_DIM * (2 if add_deltas else 1)
    lstm_model = tcn_model = None
    if mode in ("lstm", "ensemble"):
        if not lstm_weights:
            raise SystemExit(
                "[ERROR] --lstm_weights required for lstm/ensemble")
        lstm_model = build_lstm_model(
            len(classes), seq_len, feat_dim, lstm_w1, lstm_w2, lstm_dropout, lstm_l2)
        lstm_model.load_weights(lstm_weights)
        print(f"[OK] Loaded LSTM: {lstm_weights}")
    if mode in ("tcn", "ensemble"):
        if not tcn_weights:
            raise SystemExit("[ERROR] --tcn_weights required for tcn/ensemble")
        tcn_model = build_tcn_model(
            len(classes), seq_len, feat_dim, tcn_dropout, tcn_l2)
        tcn_model.load_weights(tcn_weights)
        print(f"[OK] Loaded TCN: {tcn_weights}")
    return classes, lstm_model, tcn_model

# ---------------- TTA: shifts + optional speed resample ----------------


def resample_time(x, target_len, speed=1.0):
    """Linear resample in time to target_len after speed scaling."""
    if speed == 1.0:  # just ensure exact length
        if x.shape[0] == target_len:
            return x
        # fallback simple linear stretch
    t_orig = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float32)
    t_new = np.linspace(0.0, 1.0, int(
        round(x.shape[0]/speed)), dtype=np.float32)
    xs = np.stack([np.interp(t_new, t_orig, x[:, d])
                  for d in range(x.shape[1])], axis=1)
    # now re-sample to target_len exactly
    t2 = np.linspace(0.0, 1.0, xs.shape[0], dtype=np.float32)
    t_final = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    xf = np.stack([np.interp(t_final, t2, xs[:, d])
                  for d in range(xs.shape[1])], axis=1)
    return xf.astype(np.float32)


def tta_variants(x, kind="pro"):
    """
    Returns list of (T,D) variants.
     - 'none'  : [x]
     - 'shift3': shifts {-2, 0, +2}
     - 'speed3': speeds {0.9, 1.0, 1.1}
     - 'pro'   : speeds {0.9, 1.0, 1.1} × shifts {-2,0,+2}
    Shifts use edge padding.
    """
    T, D = x.shape

    def shift_clip(z, delta):
        if delta == 0:
            return z
        if delta > 0:
            pad = np.repeat(z[:1], delta, axis=0)
            return np.concatenate([pad, z[:-delta]], axis=0)
        else:
            d = -delta
            pad = np.repeat(z[-1:], d, axis=0)
            return np.concatenate([z[d:], pad], axis=0)

    if kind == "none":
        return [x]

    if kind == "shift3":
        return [shift_clip(x, -2), x, shift_clip(x, +2)]

    if kind == "speed3":
        outs = []
        for sp in (0.9, 1.0, 1.1):
            outs.append(resample_time(x, T, sp))
        return outs

    # pro = speed3 × shift3
    outs = []
    for sp in (0.9, 1.0, 1.1):
        xsp = resample_time(x, T, sp) if sp != 1.0 else x
        outs.extend([shift_clip(xsp, -2), xsp, shift_clip(xsp, +2)])
    return outs

# ---------------- Motion energy (hands only) ----------------


def hand_motion(prev_vec, curr_vec):
    if prev_vec is None or curr_vec is None:
        return 0.0
    lh = POSE_DIM + FACE_DIM
    rh = lh + L_HAND_DIM
    prev = np.concatenate(
        [prev_vec[lh:lh+L_HAND_DIM], prev_vec[rh:rh+R_HAND_DIM]])
    curr = np.concatenate(
        [curr_vec[lh:lh+L_HAND_DIM], curr_vec[rh:rh+R_HAND_DIM]])
    return float(np.mean(np.abs(curr - prev)))


def robust_thresh(values, k=3.0):
    """median + k * MAD (robust)."""
    arr = np.asarray(values, dtype=np.float32)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + 1e-9
    return float(med + k*mad)

# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser("ISL v5 Live (accuracy-first)")
    # Mode / weights / labels
    ap.add_argument(
        "--mode", choices=["lstm", "tcn", "ensemble"], default="tcn")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--lstm_weights", default=None)
    ap.add_argument("--tcn_weights",  default=None)
    # Sequence / sampling
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=float, default=20.0)  # match collector
    ap.add_argument("--add_deltas", action="store_true")
    # TTA
    ap.add_argument(
        "--tta", choices=["none", "shift3", "speed3", "pro"], default="pro")
    ap.add_argument("--tta_pool", choices=["mean", "max"], default="max")
    # Ensemble weight
    ap.add_argument("--ens_w_tcn", type=float, default=0.6,
                    help="Weight for TCN in ensemble (rest for LSTM)")
    # Confidence / commit rules
    ap.add_argument("--conf", type=float, default=0.60,
                    help="Commit if top-1 >= conf")
    ap.add_argument("--conf_low", type=float, default=0.40,
                    help="Below this, force quick retry/unknown")
    ap.add_argument("--retry_on_low", action="store_true", default=True)
    ap.add_argument("--allow_unknown", action="store_true", default=True)
    ap.add_argument("--auto_space", action="store_true", default=True)
    # Hand presence
    ap.add_argument("--min_hand_ratio", type=float, default=0.50,
                    help="Min fraction of frames with any hand visible")
    # Start flag & cooldowns
    # consecutive frames with both hands
    ap.add_argument("--start_hold_frames", type=int, default=8)
    ap.add_argument("--cooldown", type=float, default=1.5)
    ap.add_argument("--quick_retry_cooldown", type=float, default=0.6)
    # Motion gating
    ap.add_argument("--gate_motion", action="store_true", default=True)
    ap.add_argument("--motion_auto",  action="store_true",
                    default=True, help="Adaptive motion threshold")
    ap.add_argument("--motion_thresh", type=float,
                    default=0.0035, help="Used if motion_auto=False")
    ap.add_argument("--motion_window", type=int, default=5)
    ap.add_argument("--motion_need", type=int, default=2,
                    help="Require N positives in window M to start filling")
    # Camera / display
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--scale", type=float, default=DISPLAY_SCALE)
    # Holistic
    ap.add_argument("--det_conf", type=float, default=0.50)
    ap.add_argument("--trk_conf", type=float, default=0.50)
    ap.add_argument("--model_complexity", type=int, default=1)

    args = ap.parse_args()
    print(
        f"[INFO] TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    if not USE_MP:
        return

    # Build/load
    classes, lstm_model, tcn_model = load_models(
        mode=args.mode,
        labels_path=args.labels,
        add_deltas=args.add_deltas,
        lstm_weights=args.lstm_weights, tcn_weights=args.tcn_weights,
        seq_len=args.frames
    )
    num_classes = len(classes)
    feat_dim = FRAME_DIM * (2 if args.add_deltas else 1)
    print(f"[INFO] #classes={num_classes}")

    def model_predict(X):
        """Return (C,) probs pooled over TTA and ensemble with weights."""
        outs = []
        if args.mode in ("lstm", "ensemble"):
            p = lstm_model.predict(X, verbose=0)  # (N,C)
            outs.append(("lstm", p))
        if args.mode in ("tcn", "ensemble"):
            p = tcn_model.predict(X, verbose=0)
            outs.append(("tcn", p))
        # Pool across TTA
        pooled = {}
        for name, P in outs:
            if args.tta_pool == "mean":
                pooled[name] = P.mean(axis=0)
            else:  # max
                pooled[name] = P.max(axis=0)
        # Ensemble
        if args.mode == "ensemble":
            w_t = float(np.clip(args.ens_w_tcn, 0.0, 1.0))
            w_l = 1.0 - w_t
            return w_l*pooled.get("lstm", 0.0) + w_t*pooled.get("tcn", 0.0)
        return pooled[outs[0][0]]

    # Camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cv2.namedWindow("ISL v5 (Precision Live)", cv2.WINDOW_NORMAL)
    display_scale = float(np.clip(args.scale, DISPLAY_MIN, DISPLAY_MAX))

    # FSM
    WAIT_START, CAPTURE, PREDICT, COOLDOWN = 0, 1, 2, 3
    state = WAIT_START
    started_once = False
    paused = False

    # Buffers/state
    seq = []
    hand_vis = []
    target_gap = 1.0 / max(1e-6, float(args.fps))
    last_sample_t = time.time()
    cooldown_until = 0.0
    retry_pending = False

    prev_vec = None
    motion_hist = deque(maxlen=max(3, args.motion_window))
    motion_bg = deque(maxlen=40)  # for adaptive threshold

    typed_text = ""
    last_top3 = None

    fps_hist = deque(maxlen=30)
    last_t = time.time()

    with mp_holistic.Holistic(
        model_complexity=args.model_complexity,
        refine_face_landmarks=False,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.trk_conf
    ) as holistic:

        print("[INFO] Controls: ESC quit | Space space | B backspace | C clear | +/- zoom | S save | P pause")

        both_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            rgb.flags.writeable = True
            panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Draw landmarks (safe)
            try:
                if res.pose_landmarks:
                    mp_drawing.draw_landmarks(panel, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                              mp_styles.get_default_pose_landmarks_style())
                if res.face_landmarks:
                    mp_drawing.draw_landmarks(panel, res.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                              landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(panel, res.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
                if res.left_hand_landmarks:
                    mp_drawing.draw_landmarks(panel, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_styles.get_default_hand_landmarks_style(),
                                              mp_styles.get_default_hand_connections_style())
                if res.right_hand_landmarks:
                    mp_drawing.draw_landmarks(panel, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                              mp_styles.get_default_hand_landmarks_style(),
                                              mp_styles.get_default_hand_connections_style())
            except Exception:
                pass

            h, w = panel.shape[:2]
            header_x, header_y = 18, 30
            bar_x, bar_y, bar_w, bar_h = 18, 56, 440, 16

            curr_vec = extract_keypoints(res)
            now = time.time()

            # Update motion baseline (when NOT currently recording)
            if state in (WAIT_START, COOLDOWN) or paused:
                mtmp = hand_motion(prev_vec, curr_vec)
                motion_bg.append(mtmp)

            if paused:
                draw_status(panel, "PAUSED - press P to resume",
                            header_x, header_y, (0, 200, 255))
            else:
                if state == WAIT_START:
                    if two_hands_present(res):
                        both_count += 1
                    else:
                        both_count = 0
                    draw_status(panel, "SHOW BOTH HANDS to start",
                                header_x, header_y, (0, 255, 255))
                    draw_bar(panel, bar_x, bar_y, bar_w, bar_h,
                             both_count / max(1, args.start_hold_frames), color=(0, 200, 0))
                    if both_count >= args.start_hold_frames:
                        both_count = 0
                        state = CAPTURE
                        seq.clear()
                        hand_vis.clear()
                        motion_hist.clear()
                        prev_vec = None
                        last_sample_t = now
                        started_once = True
                        retry_pending = False

                elif state == CAPTURE:
                    # Motion gate
                    ready_to_add = True
                    if args.gate_motion:
                        m = hand_motion(prev_vec, curr_vec)
                        if args.motion_auto and len(motion_bg) >= 12:
                            thr = robust_thresh(motion_bg, k=3.0)
                        else:
                            thr = float(args.motion_thresh)
                        motion_hist.append(1 if m >= thr else 0)
                        if len(seq) == 0 and (sum(motion_hist) < min(args.motion_need, len(motion_hist))):
                            ready_to_add = False
                    # Sample at target fps
                    if ready_to_add and ((now - last_sample_t) >= target_gap):
                        last_sample_t = now
                        seq.append(curr_vec)
                        hand_vis.append(hands_present(res))

                    draw_status(
                        panel, f"CAPTURE: {len(seq)}/{args.frames}", header_x, header_y, (0, 255, 255))
                    draw_bar(panel, bar_x, bar_y, bar_w, bar_h, len(
                        seq)/float(args.frames), color=(0, 200, 0))

                    if len(seq) >= args.frames:
                        state = PREDICT

                elif state == PREDICT:
                    # Hand presence guard
                    hand_ratio = (
                        sum(hand_vis)/max(1, len(hand_vis))) if hand_vis else 0.0
                    x = np.stack(seq[:args.frames], axis=0).astype(np.float32)
                    if args.add_deltas:
                        x = add_deltas_seq(x)
                    variants = tta_variants(x, args.tta)
                    X = np.stack(variants, axis=0)

                    draw_status(panel, "PREDICTING...",
                                header_x, header_y, (0, 255, 0))
                    probs = model_predict(X)  # (C,)
                    order = np.argsort(-probs)
                    top1, top2, top3 = int(order[0]), int(
                        order[1]), int(order[2])
                    top3_list = [(classes[top1], float(probs[top1])),
                                 (classes[top2], float(probs[top2])),
                                 (classes[top3], float(probs[top3]))]
                    last_top3 = top3_list

                    # Decide commit
                    commit = False
                    unknown = False
                    if hand_ratio < args.min_hand_ratio:
                        unknown = True
                    elif float(probs[top1]) >= args.conf:
                        commit = True
                    elif float(probs[top1]) < args.conf_low and args.retry_on_low and not retry_pending:
                        # quick retry: don't commit, shorter cooldown, immediately capture again
                        retry_pending = True
                        seq.clear()
                        hand_vis.clear()
                        motion_hist.clear()
                        prev_vec = None
                        last_sample_t = now
                        state = COOLDOWN
                        cooldown_until = now + float(args.quick_retry_cooldown)
                    else:
                        unknown = True

                    if commit:
                        # type and space if needed
                        if typed_text and not typed_text.endswith(" "):
                            typed_text += " "
                        typed_text += classes[top1]
                        if args.auto_space and not typed_text.endswith(" "):
                            typed_text += " "
                        retry_pending = False
                        state = COOLDOWN
                        cooldown_until = now + float(args.cooldown)
                    elif unknown:
                        if args.allow_unknown:
                            if typed_text and not typed_text.endswith(" "):
                                typed_text += " "
                            typed_text += "?"
                            if args.auto_space and not typed_text.endswith(" "):
                                typed_text += " "
                        retry_pending = False
                        state = COOLDOWN
                        cooldown_until = now + float(args.cooldown)

                    # reset capture buffers
                    seq.clear()
                    hand_vis.clear()
                    motion_hist.clear()
                    prev_vec = None

                elif state == COOLDOWN:
                    remain = max(0.0, cooldown_until - now)
                    label = "RETRY" if retry_pending else "COOLDOWN"
                    draw_status(panel, f"{label}: {remain:.1f}s",
                                header_x, header_y, (180, 255, 180))
                    frac = (args.cooldown - remain)/max(1e-6, args.cooldown) if not retry_pending \
                        else (args.quick_retry_cooldown - remain)/max(1e-6, args.quick_retry_cooldown)
                    draw_bar(panel, bar_x, bar_y, bar_w, bar_h,
                             np.clip(frac, 0, 1), color=(0, 180, 255))
                    if last_top3:
                        draw_top3(panel, bar_x+bar_w+20, bar_y-2, last_top3)
                    if now >= cooldown_until:
                        state = CAPTURE
                        last_sample_t = now

            # Typed text bar
            cv2.rectangle(panel, (18, h-80), (w-18, h-20), (30, 30, 30), -1)
            cv2.putText(panel, f"Typed: {typed_text}", (26, h-38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS
            t = time.time()
            fps = 1.0/max(1e-6, t-last_t)
            last_t = t
            fps_hist.append(fps)
            fps_avg = sum(fps_hist)/len(fps_hist) if fps_hist else 0.0
            cv2.putText(panel, f"FPS:{fps_avg:.1f}", (w-140, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Show
            disp = cv2.resize(panel, None, fx=display_scale, fy=display_scale,
                              interpolation=cv2.INTER_AREA) if abs(display_scale-1.0) > 1e-3 else panel
            cv2.imshow("ISL v5 (Precision Live)", disp)

            # Keys
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == 32:
                if not typed_text.endswith(" "):
                    typed_text += " "
            elif k in (8, ord('b'), ord('B')):
                typed_text = typed_text[:-1] if typed_text else ""
            elif k in (ord('c'), ord('C')):
                typed_text = ""
            elif k in (ord('+'), ord('=')):
                display_scale = min(DISPLAY_MAX, display_scale + 0.05)
            elif k in (ord('-'), ord('_')):
                display_scale = max(DISPLAY_MIN, display_scale - 0.05)
            elif k in (ord('s'), ord('S')):
                pth = save_transcript(typed_text)
                print(f"[INFO] Transcript saved: {pth}")
            elif k in (ord('p'), ord('P')):
                paused = not paused
                if not paused:
                    last_sample_t = time.time()
                    if state == COOLDOWN:
                        cooldown_until = time.time() + 0.4

            prev_vec = curr_vec

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
