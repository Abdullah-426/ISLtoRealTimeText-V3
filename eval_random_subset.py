#!/usr/bin/env python3
"""
Random-subset evaluator for ISL v5 models (LSTM / TCN / ENSEMBLE).

- Samples random clips from Dataset_Split/{train,val,test}/<class>/clip_xxx/sequence.npy
- Matches collector layout per frame: 1662 dims (pose 132 + face 1404 + LH 63 + RH 63)
- Optional --add_deltas doubles feature size as in training
- Reports Top-1 and Top-3 accuracy per model and ensemble (avg of available models)

Usage examples at the bottom of this file or see the README in the chat message.
"""

import os
import sys
import json
import math
import argparse
import random
from collections import defaultdict, Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

# ------------- Feature layout (MATCHES YOUR COLLECTOR) -------------
POSE_LM = 33      # (x,y,z,visibility)
FACE_LM = 468     # (x,y,z)
HAND_LM = 21      # (x,y,z)

POSE_DIM = POSE_LM * 4        # 132
FACE_DIM = FACE_LM * 3        # 1404
L_HAND_DIM = HAND_LM * 3      # 63
R_HAND_DIM = HAND_LM * 3      # 63
FRAME_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662
SEQ_LEN_DEFAULT = 48

# --------------------- Label / split helpers ----------------------
INVALID_FS_CHARS = set('<>:"/\\|?*')


def sanitize_dirname(label: str) -> str:
    s = "".join('_' if ch in INVALID_FS_CHARS else ch for ch in label)
    s = s.replace("  ", " ").strip()
    s = s.replace("?", "")  # collector removed '?'
    return s


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


def discover_sequences(split_root, splits, labels):
    """
    Return list of (seq_path, class_idx) across given splits.
    Expects structure: split_root/split/<sanitized_label>/clip_xxx/sequence.npy
    """
    items = []
    for label_idx, label in enumerate(labels):
        sname = sanitize_dirname(label)
        for split in splits:
            class_dir = os.path.join(split_root, split, sname)
            if not os.path.isdir(class_dir):
                continue
            for clip_name in os.listdir(class_dir):
                clip_dir = os.path.join(class_dir, clip_name)
                if not os.path.isdir(clip_dir):
                    continue
                seq_path = os.path.join(clip_dir, "sequence.npy")
                if os.path.isfile(seq_path):
                    items.append((seq_path, label_idx))
    return items

# -------------------- Sequence shaping helpers --------------------


def add_deltas_seq(x: np.ndarray) -> np.ndarray:
    # x: (T, D) -> concat([x, dx]) where dx[0]=x0, dx[t]=x[t]-x[t-1]
    dx = np.concatenate([x[:1], x[1:] - x[:-1]], axis=0)
    return np.concatenate([x, dx], axis=-1)


def trim_or_pad_to_len(x: np.ndarray, T: int) -> np.ndarray:
    # x: (t, D) -> (T, D): center-crop or edge-pad
    t, D = x.shape
    if t == T:
        return x
    if t > T:
        start = (t - T) // 2
        return x[start:start+T]
    # t < T: pad at both ends (edge repeat)
    pad_total = T - t
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    left = np.repeat(x[:1], pad_left, axis=0)
    right = np.repeat(x[-1:], pad_right, axis=0)
    return np.concatenate([left, x, right], axis=0)

# ----------------------- Small TTA utility ------------------------


def tta_variants(x, kind="none"):
    """
    x: (T,D)
    - 'none'   : [x]
    - 'shift3' : [-2, 0, +2]
    - 'shift5' : [-4, -2, 0, +2, +4]
    """
    if kind == "none":
        return [x]

    def shift(delta):
        if delta == 0:
            return x
        T = x.shape[0]
        if delta > 0:
            pad = np.repeat(x[:1], delta, axis=0)
            return np.concatenate([pad, x[:-delta]], axis=0)
        else:
            delta = -delta
            pad = np.repeat(x[-1:], delta, axis=0)
            return np.concatenate([x[delta:], pad], axis=0)
    if kind == "shift3":
        return [shift(-2), x, shift(+2)]
    if kind == "shift5":
        return [shift(-4), shift(-2), x, shift(+2), shift(+4)]
    return [x]

# ---------------------- Attention layer ---------------------------


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

# ----------------------- Model definitions ------------------------


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

# -------------------------- Main eval -----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Random-subset evaluation for ISL v5 models")
    ap.add_argument("--split_root", type=str, required=True,
                    help="Dataset_Split root")
    ap.add_argument("--splits", type=str, default="test",
                    help="Comma list of splits to sample from (e.g. test or val,test)")
    ap.add_argument("--labels", type=str, required=True,
                    help="labels.json (for class order)")

    # Sampling
    g = ap.add_argument_group("Sampling")
    g.add_argument("--k_per_class", type=int, default=0,
                   help="Pick K random clips per class (balanced). If 0, use --total_samples.")
    g.add_argument("--total_samples", type=int, default=300,
                   help="Total random clips across classes (ignored if --k_per_class>0).")
    g.add_argument("--seed", type=int, default=17)

    # Models
    g2 = ap.add_argument_group("Models")
    g2.add_argument("--lstm_weights", type=str, default=None)
    g2.add_argument("--tcn_weights", type=str, default=None)
    g2.add_argument("--frames", type=int, default=SEQ_LEN_DEFAULT)
    g2.add_argument("--add_deltas", action="store_true")
    g2.add_argument("--tta", type=str,
                    choices=["none", "shift3", "shift5"], default="none")

    # LSTM/TCN hyperparams (keep defaults as training)
    g2.add_argument("--lstm_w1", type=int, default=224)
    g2.add_argument("--lstm_w2", type=int, default=128)
    g2.add_argument("--lstm_dropout", type=float, default=0.45)
    g2.add_argument("--lstm_l2", type=float, default=1e-4)
    g2.add_argument("--tcn_dropout", type=float, default=0.45)
    g2.add_argument("--tcn_l2", type=float, default=1e-4)

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"[INFO] TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    labels = load_labels(args.labels)
    num_classes = len(labels)
    print(f"[INFO] #classes={num_classes}")

    # Build models (only those provided)
    feat_dim = FRAME_DIM * (2 if args.add_deltas else 1)
    lstm_model = None
    tcn_model = None
    if args.lstm_weights:
        lstm_model = build_lstm_model(num_classes, args.frames, feat_dim,
                                      lstm_w1=args.lstm_w1, lstm_w2=args.lstm_w2,
                                      dropout=args.lstm_dropout, l2_reg=args.lstm_l2)
        lstm_model.load_weights(args.lstm_weights)
        print(f"[OK] Loaded LSTM weights: {args.lstm_weights}")
    if args.tcn_weights:
        tcn_model = build_tcn_model(num_classes, args.frames, feat_dim,
                                    dropout=args.tcn_dropout, l2_reg=args.tcn_l2)
        tcn_model.load_weights(args.tcn_weights)
        print(f"[OK] Loaded TCN weights: {args.tcn_weights}")

    if lstm_model is None and tcn_model is None:
        sys.exit("[ERROR] Provide at least one of --lstm_weights or --tcn_weights")

    # Gather list of sequences
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_items = discover_sequences(args.split_root, split_list, labels)
    if not all_items:
        sys.exit(
            f"[ERROR] No sequences found under {args.split_root} for splits={split_list}")

    # Index by class
    by_class = defaultdict(list)
    for p, idx in all_items:
        by_class[idx].append(p)

    # Build sample list
    samples = []
    if args.k_per_class > 0:
        for ci in range(num_classes):
            pool = by_class.get(ci, [])
            if not pool:
                continue
            k = min(args.k_per_class, len(pool))
            picks = random.sample(pool, k)
            for sp in picks:
                samples.append((sp, ci))
    else:
        flat = list(all_items)
        if len(flat) <= args.total_samples:
            picks = flat
        else:
            picks = random.sample(flat, args.total_samples)
        samples = picks

    if not samples:
        sys.exit("[ERROR] Sampling produced 0 items. Check your split/classes.")

    print(f"[INFO] Sampling {len(samples)} clips from splits {split_list}")

    # Eval accumulators
    def init_stats():
        return {
            "n": 0,
            "top1": 0,
            "top3": 0,
            "per_class_n": Counter(),
            "per_class_top1": Counter(),
        }

    stats_lstm = init_stats() if lstm_model else None
    stats_tcn = init_stats() if tcn_model else None
    stats_ens = init_stats() if (lstm_model and tcn_model) else None

    # Inference fn
    def run_model(mdl, X):
        """X: (N,T,D) -> (C,) averaged over TTA"""
        p = mdl.predict(X, verbose=0)  # (N, C)
        return np.mean(p, axis=0)

    # Iterate samples
    for i, (seq_path, true_idx) in enumerate(samples, 1):
        try:
            x = np.load(seq_path).astype(np.float32)  # expect (T,D)
        except Exception as e:
            print(f"[WARN] failed to load {seq_path}: {e}")
            continue

        x = trim_or_pad_to_len(x, args.frames)
        if args.add_deltas:
            x = add_deltas_seq(x)

        # TTA pack
        variants = tta_variants(x, kind=args.tta)
        X = np.stack(variants, axis=0)  # (N,T,D or 2D)

        preds = {}

        if lstm_model is not None:
            p = run_model(lstm_model, X)
            preds["lstm"] = p

        if tcn_model is not None:
            p = run_model(tcn_model, X)
            preds["tcn"] = p

        if "lstm" in preds and "tcn" in preds:
            preds["ens"] = (preds["lstm"] + preds["tcn"]) / 2.0

        # Update stats
        for key, p in preds.items():
            order = np.argsort(-p)
            top1 = int(order[0])
            top3 = set(order[:3].tolist())
            st = stats_lstm if key == "lstm" else (
                stats_tcn if key == "tcn" else stats_ens)

            st["n"] += 1
            st["per_class_n"][true_idx] += 1
            if top1 == true_idx:
                st["top1"] += 1
                st["per_class_top1"][true_idx] += 1
            if true_idx in top3:
                st["top3"] += 1

        if i % 50 == 0 or i == len(samples):
            msg = f"[{i}/{len(samples)}]"
            if stats_lstm:
                msg += f" LSTM@1={stats_lstm['top1']/max(1,stats_lstm['n']):.3f}"
            if stats_tcn:
                msg += f" TCN@1={stats_tcn['top1']/max(1,stats_tcn['n']):.3f}"
            if stats_ens:
                msg += f" ENS@1={stats_ens['top1']/max(1,stats_ens['n']):.3f}"
            print(msg)

    def report(name, st):
        if not st or st["n"] == 0:
            return
        n = st["n"]
        acc1 = st["top1"] / n
        acc3 = st["top3"] / n
        print(f"\n=== {name} Results ===")
        print(f"Samples: {n}")
        print(f"Top-1: {acc1:.4f}")
        print(f"Top-3: {acc3:.4f}")
        # quick per-class (only those sampled)
        print("Per-class Top-1 (for sampled classes):")
        rows = []
        for ci, cnt in st["per_class_n"].items():
            hit = st["per_class_top1"][ci]
            rows.append((acc1 if cnt == 0 else hit / cnt, cnt, labels[ci]))
        rows.sort(key=lambda r: r[0])  # worst first
        for a, c, lab in rows[:10]:
            print(f"  {lab:30s}  acc={a:.2f}  n={c}")
        if len(rows) > 10:
            print("  ...")

    report("LSTM", stats_lstm)
    report("TCN",  stats_tcn)
    report("ENSEMBLE", stats_ens)


if __name__ == "__main__":
    main()
