#!/usr/bin/env python3
"""
Evaluate LSTM / TCN / ENSEMBLE on a FIXED SUBSET of classes.

- Scans Dataset_Split/<split>/<class_dir>/**/sequence.npy
- Only includes samples from the target classes below (104 items)
- Randomly samples up to --k_per_class clips per class per split
- seq_len handling: center-crop or edge-pad to --frames (default 48)
- Optional deltas to match training (--add_deltas)
- TTA: none | shift3 | shift5 | pro (speed {0.9,1.0,1.1} x shift {-2,0,+2})
- TTA pool: mean or max
- Reports: Top-1 / Top-3 for each model + ensemble, per-class accuracy,
           and a short list of most frequent confusions (for the subset).

Usage examples are at the bottom.
"""

import os
import json
import glob
import math
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

# ---- The 104 target classes (exact labels) ----
TARGET_CLASSES = [
    # Greetings & social (Video 3)
    "Hello", "Indian", "Namaste", "Bye-bye", "Thank you", "Please", "Sorry", "Welcome",
    "How are you?", "I'm fine", "My name is", "Again",

    # Yes/No & daily basics (Video 4)
    "Yes", "No", "Good", "Bad", "Correct", "Wrong", "Child", "Boy", "Girl", "Food", "Morning",
    "Good morning", "Good afternoon", "Good evening", "Good night", "Peace", "No fear",
    "Understand", "I don't understand", "Remember",

    # Questions / deictics / time (Video 5)
    "What", "Why", "How", "Where", "Who", "When", "Which", "This", "Time", "Place",

    # People & pronouns (Video 3)
    "I", "You", "He", "She", "Man", "Woman", "Deaf", "Hearing", "Teacher",

    # Family & relations (Video 7)
    "Family", "Mother", "Father", "Wife", "Husband", "Daughter", "Son", "Sister", "Brother",
    "Grandmother", "Grandfather", "Aunt", "Uncle",

    # Calendar (Video 8 & 9)
    "Day", "Week", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "Month", "Year",

    # Home / objects / states (Video 10)
    "House", "Apartment", "Car", "Chair", "Table", "Happy", "Beautiful", "Ugly", "Tall", "Short",
    "Clever", "Sweet", "Bright", "Dark", "Camera", "Photo", "Work",

    # Colours (Video 6)
    "Colours", "Black", "Green", "Brown", "Red", "Pink", "Blue", "Yellow", "Orange",
    "Golden", "Silver", "Grey",
]

# ---------------- Feature layout (MATCHES v5 collector) ----------------
POSE_LM = 33     # (x,y,z,visibility)
FACE_LM = 468    # (x,y,z)
HAND_LM = 21     # (x,y,z)

POSE_DIM = POSE_LM * 4        # 132
FACE_DIM = FACE_LM * 3        # 1404
L_HAND_DIM = HAND_LM * 3      # 63
R_HAND_DIM = HAND_LM * 3      # 63
FRAME_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662

# ---------------- Folder-name sanitizer (same as your collector) ----------------
INVALID_FS_CHARS = set('<>:"/\\|?*')


def sanitize_dirname(label: str) -> str:
    s = "".join('_' if ch in INVALID_FS_CHARS else ch for ch in label)
    s = s.replace("  ", " ").strip()
    s = s.replace("?", "")  # they were removed in your collector
    return s

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

# ---------------- IO helpers ----------------


def find_class_clip_paths(split_root, split, label):
    """
    Returns list of sequence.npy paths for one class in a split, trying both
    the raw label folder and the sanitized folder name.
    """
    split_dir = os.path.join(split_root, split)
    raw_dir = os.path.join(split_dir, label)
    san_dir = os.path.join(split_dir, sanitize_dirname(label))

    paths = []
    for d in (raw_dir, san_dir):
        if os.path.isdir(d):
            # look for .../<class>/clip_xxx/sequence.npy
            paths.extend(glob.glob(os.path.join(
                d, "**", "sequence.npy"), recursive=True))
    return sorted(set(paths))


def load_sequence_npy(path):
    arr = np.load(path)  # expected (T, 1662)
    if arr.ndim != 2:
        raise RuntimeError(f"Bad shape at {path}: {arr.shape}")
    return arr.astype(np.float32)


def center_crop_or_pad(x, T):
    """Ensure temporal length T: center-crop if longer, edge-pad if shorter."""
    t = x.shape[0]
    if t == T:
        return x
    if t > T:
        start = (t - T) // 2
        return x[start:start+T]
    # pad by repeating last frame
    pad = np.repeat(x[-1:], T - t, axis=0)
    return np.concatenate([x, pad], axis=0)


def add_deltas_seq(x):
    """x: (T, D) -> concat([x, dx]) where dx = [x0; x1-x0; ...]."""
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

# ---------------- Build/Load models ----------------


def load_models(mode, labels_path, add_deltas, lstm_weights, tcn_weights,
                seq_len, lstm_w1=224, lstm_w2=128, lstm_dropout=0.45, lstm_l2=1e-4,
                tcn_dropout=0.45, tcn_l2=1e-4):
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
        print(f"[OK] Loaded TCN : {tcn_weights}")
    return classes, lstm_model, tcn_model

# ---------------- TTA ----------------


def resample_time(x, target_len, speed=1.0):
    if speed == 1.0:
        # Still ensure exact length
        if x.shape[0] == target_len:
            return x
    t0 = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float32)
    t1 = np.linspace(0.0, 1.0, int(round(x.shape[0]/speed)), dtype=np.float32)
    xs = np.stack([np.interp(t1, t0, x[:, d])
                  for d in range(x.shape[1])], axis=1)
    t2 = np.linspace(0.0, 1.0, xs.shape[0], dtype=np.float32)
    tf = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    xf = np.stack([np.interp(tf, t2, xs[:, d])
                  for d in range(xs.shape[1])], axis=1)
    return xf.astype(np.float32)


def tta_variants(x, kind="none"):
    T, D = x.shape

    def shift_clip(z, delta):
        if delta == 0:
            return z
        if delta > 0:
            pad = np.repeat(z[:1], delta, axis=0)
            return np.concatenate([pad, z[:-delta]], axis=0)
        d = -delta
        pad = np.repeat(z[-1:], d, axis=0)
        return np.concatenate([z[d:], pad], axis=0)
    if kind == "none":
        return [x]
    if kind == "shift3":
        return [shift_clip(x, -2), x, shift_clip(x, +2)]
    if kind == "shift5":
        return [shift_clip(x, -4), shift_clip(x, -2), x, shift_clip(x, +2), shift_clip(x, +4)]
    if kind == "pro":
        outs = []
        for sp in (0.9, 1.0, 1.1):
            xsp = resample_time(x, T, sp) if sp != 1.0 else x
            outs.extend([shift_clip(xsp, -2), xsp, shift_clip(xsp, +2)])
        return outs
    return [x]

# ---------------- Prediction helper ----------------


def predict_with_models(X, mode, lstm_model, tcn_model, tta_pool="max", ens_w_tcn=0.6):
    """
    X: (N, T, D) after TTA
    Returns dict {"lstm": (C,), "tcn": (C,), "ens": (C,)} depending on mode.
    """
    outs = {}
    if mode in ("lstm", "ensemble"):
        P = lstm_model.predict(X, verbose=0)  # (N,C)
        outs["lstm"] = P.mean(axis=0) if tta_pool == "mean" else P.max(axis=0)
    if mode in ("tcn", "ensemble"):
        P = tcn_model.predict(X, verbose=0)
        outs["tcn"] = P.mean(axis=0) if tta_pool == "mean" else P.max(axis=0)
    if mode == "ensemble":
        wl = 1.0 - float(ens_w_tcn)
        wt = float(ens_w_tcn)
        outs["ens"] = wl*outs.get("lstm", 0.0) + wt*outs.get("tcn", 0.0)
    return outs

# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser(
        "Evaluate v5 models on a fixed subset of classes")
    ap.add_argument("--split_root", required=True, help="e.g., Dataset_Split")
    ap.add_argument("--splits", default="val,test",
                    help="comma-separated: train,val,test")
    ap.add_argument("--labels", required=True, help="Path to labels.json")
    ap.add_argument("--k_per_class", type=int, default=5,
                    help="Max clips per class per split")
    ap.add_argument("--total_cap", type=int, default=0,
                    help="Optional overall cap after sampling (0=off)")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--add_deltas", action="store_true")
    ap.add_argument(
        "--tta", choices=["none", "shift3", "shift5", "pro"], default="shift3")
    ap.add_argument("--tta_pool", choices=["mean", "max"], default="max")
    ap.add_argument(
        "--mode", choices=["tcn", "lstm", "ensemble"], default="ensemble")
    ap.add_argument(
        "--lstm_weights", default="models/isl_v5_lstm_mild_aw_deltas/best.weights.h5")
    ap.add_argument("--tcn_weights",
                    default="models/isl_v5_tcn_deltas/best.weights.h5")
    ap.add_argument("--ens_w_tcn", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    print(
        f"[INFO] TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    # Load label list (for class index mapping)
    all_labels = load_labels(args.labels)
    label2idx = {c: i for i, c in enumerate(all_labels)}

    # Filter to only TARGET_CLASSES that actually exist in labels.json
    target = [c for c in TARGET_CLASSES if c in label2idx]
    missing = [c for c in TARGET_CLASSES if c not in label2idx]
    if missing:
        print(
            f"[WARN] {len(missing)} target classes not found in labels.json; they will be skipped.")
        # print("\n".join(missing))

    if not target:
        raise SystemExit(
            "[ERROR] None of the TARGET_CLASSES exist in labels.json.")

    # Gather samples
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = []  # list of (path, gt_idx, label)
    for sp in splits:
        for lab in target:
            paths = find_class_clip_paths(args.split_root, sp, lab)
            if not paths:
                continue
            kk = min(args.k_per_class, len(paths))
            pick = random.sample(paths, kk)
            gt_idx = label2idx[lab]
            samples.extend([(p, gt_idx, lab) for p in pick])

    if args.total_cap and len(samples) > args.total_cap:
        samples = random.sample(samples, args.total_cap)

    if not samples:
        raise SystemExit(
            "[ERROR] No samples found for the selected classes/splits.")

    print(f"[INFO] Samples collected: {len(samples)} from splits {splits}")
    # Build/load models
    classes, lstm_model, tcn_model = load_models(
        mode=args.mode,
        labels_path=args.labels,
        add_deltas=args.add_deltas,
        lstm_weights=args.lstm_weights,
        tcn_weights=args.tcn_weights,
        seq_len=args.frames
    )
    num_classes = len(classes)

    # Stats
    top1 = {"lstm": 0, "tcn": 0, "ens": 0}
    top3 = {"lstm": 0, "tcn": 0, "ens": 0}
    per_class_counts = Counter()
    per_class_corr = {"lstm": Counter(), "tcn": Counter(), "ens": Counter()}
    confusions = {"lstm": Counter(), "tcn": Counter(),
                  "ens": Counter()}  # (gt, pred) -> count

    def eval_one(path, gt_idx):
        x = load_sequence_npy(path)   # (T0, 1662)
        x = center_crop_or_pad(x, args.frames)
        if args.add_deltas:
            x = add_deltas_seq(x)     # (T, 2*1662)
        variants = tta_variants(x, args.tta)
        X = np.stack(variants, axis=0)   # (N,T,D)
        outs = predict_with_models(X, args.mode, lstm_model, tcn_model,
                                   tta_pool=args.tta_pool, ens_w_tcn=args.ens_w_tcn)
        return outs

    # Evaluate
    for i, (p, gt, lab) in enumerate(samples, 1):
        outs = eval_one(p, gt)
        per_class_counts[lab] += 1

        for key in list(outs.keys()):  # "lstm","tcn","ens"
            probs = outs[key]
            order = np.argsort(-probs)
            pred1 = int(order[0])
            if pred1 == gt:
                top1[key] += 1
                per_class_corr[key][lab] += 1
            if gt in order[:3]:
                top3[key] += 1
            confusions[key][(classes[gt], classes[pred1])] += 1

        if i % 50 == 0 or i == len(samples):
            def r(v): return v/float(i)
            msg = [f"[{i}/{len(samples)}]"]
            if "lstm" in outs:
                msg.append(f"LSTM@1={r(top1['lstm']):.3f}")
            if "tcn" in outs:
                msg.append(f"TCN@1={r(top1['tcn']):.3f}")
            if "ens" in outs:
                msg.append(f"ENS@1={r(top1['ens']):.3f}")
            print(" ".join(msg))

    N = len(samples)

    def report_one(name):
        if name not in top1:
            return
        print(f"\n=== {name.upper()} Results ===")
        print(f"Samples: {N}")
        print(f"Top-1: {top1[name]/N:.4f}")
        print(f"Top-3: {top3[name]/N:.4f}")
        print("Per-class Top-1 (subset):")
        # show in alphabetical order for stability
        for lab in sorted(per_class_counts.keys()):
            n = per_class_counts[lab]
            c = per_class_corr[name][lab]
            acc = (c/n) if n > 0 else 0.0
            print(f"  {lab:30s} acc={acc:.2f}  n={n}")
        # top confusions
        print("Top confusions:")
        for (gt, pd), cnt in confusions[name].most_common(15):
            if gt == pd:
                continue
            print(f"  {gt} -> {pd}: {cnt}")

    report_one("lstm")
    report_one("tcn")
    report_one("ens")


if __name__ == "__main__":
    main()
