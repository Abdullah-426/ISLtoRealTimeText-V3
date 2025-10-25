#!/usr/bin/env python3
"""
Analyze ISL v5 diagnostics on Dataset_Split.

What it does:
- Loads clips from Dataset_Split/<split>/<class>/clip_xxx/(sequence.npy or f_*.npy)
- Preprocesses to MATCH training (1662 dims per frame, optional deltas)
- Optional resample to T=48 frames (if a clip isn't exactly T)
- Strong TTA (shift ±2, warp 0.9x/1.0x/1.1x) and hand-focus (downweight face+pose)
- Evaluates LSTM, TCN, and Ensemble, and logs per-class & overall metrics
- Measures motion energy (hands, face, pose), hand presence %, confidence margin, confusions
- Creates CSVs: diagnostics_per_class.csv, confusions_[model].csv, per_clip_errors_[model].csv

Run this first. Share the printed “WORST CLASSES (by Ensemble)” section.
"""

import os
import re
import json
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

# ---------- Layout (MATCHES YOUR COLLECTOR) ----------
POSE_LM = 33     # (x,y,z,visibility)
FACE_LM = 468    # (x,y,z)
HAND_LM = 21     # (x,y,z)

POSE_DIM = POSE_LM * 4        # 132
FACE_DIM = FACE_LM * 3        # 1404
L_HAND_DIM = HAND_LM * 3      # 63
R_HAND_DIM = HAND_LM * 3      # 63
FRAME_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662

POSE_SL = slice(0, POSE_DIM)
FACE_SL = slice(POSE_DIM, POSE_DIM + FACE_DIM)
LH_SL = slice(POSE_DIM + FACE_DIM, POSE_DIM + FACE_DIM + L_HAND_DIM)
RH_SL = slice(POSE_DIM + FACE_DIM + L_HAND_DIM, POSE_DIM +
              FACE_DIM + L_HAND_DIM + R_HAND_DIM)

# ---------- Helpers ----------
INVALID_FS_CHARS = set('<>:"/\\|?*')


def sanitize_dirname(label: str) -> str:
    s = "".join('_' if ch in INVALID_FS_CHARS else ch for ch in label)
    s = s.replace("  ", " ").strip()
    s = s.replace("?", "")
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


def add_deltas_seq(x: np.ndarray) -> np.ndarray:
    dx = np.concatenate([x[:1], x[1:] - x[:-1]], axis=0)
    return np.concatenate([x, dx], axis=-1)


def resample_to_T_uniform(x: np.ndarray, T_out: int) -> np.ndarray:
    """x: (T_in, D) -> (T_out, D) linear interpolation on index grid."""
    T_in, D = x.shape
    if T_in == T_out:
        return x.astype(np.float32, copy=False)
    if T_in <= 1:
        return np.repeat(x, repeats=T_out, axis=0).astype(np.float32)
    s = np.linspace(0, T_in - 1, T_out, dtype=np.float64)
    i0 = np.floor(s).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, T_in - 1)
    w = (s - i0).astype(np.float32)[:, None]
    y = (1.0 - w) * x[i0, :] + w * x[i1, :]
    return y.astype(np.float32)

# ---------- TTA ----------


def shift_clip(x, delta):
    if delta == 0:
        return x
    T = x.shape[0]
    if delta > 0:
        pad = np.repeat(x[:1], delta, axis=0)
        return np.concatenate([pad, x[:-delta]], axis=0)
    d = -delta
    pad = np.repeat(x[-1:], d, axis=0)
    return np.concatenate([x[d:], pad], axis=0)


def time_warp(x, speed=1.0):
    T, D = x.shape
    if T <= 1 or abs(speed - 1.0) < 1e-6:
        return x
    pos = np.linspace(0, T-1, T, dtype=np.float64) / max(1e-6, speed)
    pos = np.clip(pos, 0.0, T-1.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, T-1)
    w = (pos - i0).astype(np.float32)[:, None]
    return ((1.0 - w) * x[i0, :] + w * x[i1, :]).astype(np.float32)


def hand_focus_variant(x, face_pose_scale=0.75):
    y = x.copy()
    y[:, POSE_SL] *= face_pose_scale
    y[:, FACE_SL] *= face_pose_scale
    return y


def build_tta_set(x_in, do_shift=True, do_warp=True, do_hand_focus=True,
                  shift_vals=(-2, 0, +2), warp_speeds=(0.9, 1.0, 1.1), face_pose_scale=0.75):
    variants = []
    bases = [x_in]
    if do_hand_focus:
        bases.append(hand_focus_variant(x_in, face_pose_scale=face_pose_scale))
    for b in bases:
        warped = [b]
        if do_warp:
            warped = [time_warp(b, s) for s in warp_speeds]
        if do_shift:
            for w in warped:
                for d in shift_vals:
                    variants.append(shift_clip(w, d))
        else:
            variants.extend(warped)
    return variants

# ---------- Motion / Presence ----------


def l1_energy(x, sl: slice):
    if x.shape[0] <= 1:
        return 0.0
    diff = np.abs(np.diff(x[:, sl], axis=0))
    return float(np.mean(diff))


def hand_presence_ratio(x: np.ndarray):
    """Frame-level 'any hand visible' heuristic: sum(abs(hand)) > 0."""
    lh = x[:, LH_SL]
    rh = x[:, RH_SL]
    present = (np.sum(np.abs(lh), axis=1) > 0) | (
        np.sum(np.abs(rh), axis=1) > 0)
    return float(np.mean(present)), present

# ---------- Models ----------


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

# ---------- Data loading ----------


def find_clip_arrays(clip_dir: Path):
    seq = clip_dir / "sequence.npy"
    if seq.exists():
        x = np.load(str(seq))
        return x.astype(np.float32)
    # else stack f_*.npy
    Fs = sorted(clip_dir.glob("f_*.npy"))
    if not Fs:
        return None
    arrs = [np.load(str(f)).astype(np.float32) for f in Fs]
    return np.stack(arrs, axis=0)  # (T,D)


def collect_dataset(root: Path, splits, labels, T_target=48, k_per_class=None, total_samples=None):
    """Return list of (x:(T_target,1662), y:int, meta). Samples balanced if k_per_class given."""
    label_to_idx = {c: i for i, c in enumerate(labels)}
    per_class_files = defaultdict(list)

    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for label in labels:
            cdir = split_dir / sanitize_dirname(label)
            if not cdir.is_dir():
                continue
            for clip in cdir.iterdir():
                if not clip.is_dir():
                    continue
                per_class_files[label].append(clip)

    # sampling
    rng = np.random.default_rng(1234)
    picked = []
    if k_per_class is not None:
        for label, clips in per_class_files.items():
            if not clips:
                continue
            sel = clips if len(clips) <= k_per_class else list(
                rng.choice(clips, size=k_per_class, replace=False))
            picked.extend([(label, p) for p in sel])
    elif total_samples is not None:
        all_pairs = [(label, p)
                     for label, L in per_class_files.items() for p in L]
        if total_samples < len(all_pairs):
            sel_idx = rng.choice(
                len(all_pairs), size=total_samples, replace=False)
            picked = [all_pairs[i] for i in sel_idx]
        else:
            picked = all_pairs
    else:
        picked = [(label, p) for label, L in per_class_files.items()
                  for p in L]

    samples = []
    for label, clip_dir in picked:
        x = find_clip_arrays(clip_dir)
        if x is None:
            continue
        # Resample to T_target if needed
        if x.shape[0] != T_target:
            x = resample_to_T_uniform(x, T_target)
        if x.shape[-1] != FRAME_DIM:
            # skip malformed
            continue
        y = label_to_idx[label]
        samples.append((x.astype(np.float32), y, {"split": str(clip_dir.parents[1].name),
                                                  "class": label, "clip": str(clip_dir)}))
    return samples

# ---------- Diagnostics core ----------


def eval_model_on_variants(model, X, tta_pool="max"):
    if model is None:
        return None
    P = model.predict(X, verbose=0)  # (N,C)
    if tta_pool == "max":
        return np.max(P, axis=0)
    return np.mean(P, axis=0)


def run_diagnostics(
    split_root: str,
    splits: list,
    labels_path: str,
    frames: int,
    add_deltas: bool,
    lstm_weights: str = None,
    tcn_weights: str = None,
    k_per_class: int = None,
    total_samples: int = None,
    tta_pool: str = "max",
    do_shift: bool = True,
    do_warp: bool = True,
    do_hand_focus: bool = True,
    ens_w_tcn: float = 0.8,
    ens_w_lstm: float = 0.2,
    out_dir: str = "diag_out"
):
    os.makedirs(out_dir, exist_ok=True)
    labels = load_labels(labels_path)
    num_classes = len(labels)

    feat_dim = FRAME_DIM * (2 if add_deltas else 1)

    # Build models if weights provided
    lstm = None
    tcn = None
    if lstm_weights:
        lstm = build_lstm_model(num_classes, frames, feat_dim)
        lstm.load_weights(lstm_weights)
    if tcn_weights:
        tcn = build_tcn_model(num_classes, frames, feat_dim)
        tcn.load_weights(tcn_weights)

    # Load dataset
    samples = collect_dataset(Path(split_root), splits, labels, T_target=frames,
                              k_per_class=k_per_class, total_samples=total_samples)
    if not samples:
        print("[ERROR] No samples found.")
        return

    # Aggregates
    per_class = {
        i: {
            "label": labels[i],
            "n": 0,
            "acc_lstm": 0, "acc_tcn": 0, "acc_ens": 0,
            "top3_lstm": 0, "top3_tcn": 0, "top3_ens": 0,
            "p1_mean_lstm": 0.0, "p1_mean_tcn": 0.0, "p1_mean_ens": 0.0,
            "margin_mean_lstm": 0.0, "margin_mean_tcn": 0.0, "margin_mean_ens": 0.0,
            "E_pose": 0.0, "E_face": 0.0, "E_hands": 0.0,
            "hand_presence": 0.0,
            "confusions_lstm": Counter(), "confusions_tcn": Counter(), "confusions_ens": Counter()
        } for i in range(num_classes)
    }

    per_clip_err_rows = {"lstm": [], "tcn": [], "ens": []}
    confusions_mat = {
        "lstm": np.zeros((num_classes, num_classes), dtype=np.int64),
        "tcn": np.zeros((num_classes, num_classes), dtype=np.int64),
        "ens": np.zeros((num_classes, num_classes), dtype=np.int64)
    }

    # Iterate
    for idx, (x, y, meta) in enumerate(samples, 1):
        # Energy + presence
        hand_ratio, hand_present = hand_presence_ratio(x)
        E_pose = l1_energy(x, POSE_SL)
        E_face = l1_energy(x, FACE_SL)
        E_hands = 0.5 * (l1_energy(x, LH_SL) + l1_energy(x, RH_SL))

        # Input (with deltas if requested)
        xin = add_deltas_seq(x) if add_deltas else x

        # TTA variants
        variants = build_tta_set(
            xin,
            do_shift=do_shift, do_warp=do_warp, do_hand_focus=do_hand_focus,
            shift_vals=(-2, 0, +2), warp_speeds=(0.9, 1.0, 1.1), face_pose_scale=0.75
        )
        X = np.stack(variants, axis=0)  # (N,T,D)

        # Predict
        probs_lstm = eval_model_on_variants(
            lstm, X, tta_pool) if lstm is not None else None
        probs_tcn = eval_model_on_variants(
            tcn,  X, tta_pool) if tcn is not None else None

        # Ensemble (TCN-weighted if both exist)
        probs_ens = None
        if probs_lstm is not None and probs_tcn is not None:
            probs_ens = ens_w_tcn * probs_tcn + ens_w_lstm * probs_lstm
        elif probs_tcn is not None:
            probs_ens = probs_tcn
        elif probs_lstm is not None:
            probs_ens = probs_lstm

        # Update metrics func
        def upd(tag, probs):
            if probs is None:
                return
            order = np.argsort(-probs)
            p1 = float(probs[order[0]])
            p2 = float(probs[order[1]] if len(order) > 1 else 0.0)
            acc1 = 1 if order[0] == y else 0
            acc3 = 1 if (y in order[:3]) else 0

            pc = per_class[y]
            pc[f"acc_{tag}"] += acc1
            pc[f"top3_{tag}"] += acc3
            pc[f"p1_mean_{tag}"] += p1
            pc[f"margin_mean_{tag}"] += (p1 - p2)
            pc["E_pose"] += E_pose
            pc["E_face"] += E_face
            pc["E_hands"] += E_hands
            pc["hand_presence"] += hand_ratio
            pc["n"] += 1

            confusions_mat[tag][y, order[0]] += 1
            if order[0] != y:
                pc[f"confusions_{tag}"][order[0]] += 1
                per_clip_err_rows[tag].append({
                    "true": labels[y], "pred": labels[order[0]], "p1": p1, "margin": p1-p2,
                    "clip": meta["clip"], "split": meta["split"]
                })

        upd("lstm", probs_lstm)
        upd("tcn",  probs_tcn)
        upd("ens",  probs_ens)

        if idx % 50 == 0:
            acc_t = sum(per_class[i]["acc_tcn"] for i in per_class) / \
                max(1, sum(per_class[i]["n"] for i in per_class))
            acc_l = sum(per_class[i]["acc_lstm"] for i in per_class) / \
                max(1, sum(per_class[i]["n"] for i in per_class))
            acc_e = sum(per_class[i]["acc_ens"] for i in per_class) / \
                max(1, sum(per_class[i]["n"] for i in per_class))
            print(
                f"[{idx}/{len(samples)}] LSTM@1={acc_l:.3f}  TCN@1={acc_t:.3f}  ENS@1={acc_e:.3f}")

    # Finalize per-class
    rows = []
    N_total = sum(pc["n"] for pc in per_class.values())
    acc_overall = {"lstm": 0.0, "tcn": 0.0, "ens": 0.0}
    top3_overall = {"lstm": 0.0, "tcn": 0.0, "ens": 0.0}

    for i, pc in per_class.items():
        n = max(1, pc["n"])
        for key in ["acc_lstm", "acc_tcn", "acc_ens", "top3_lstm", "top3_tcn", "top3_ens",
                    "p1_mean_lstm", "p1_mean_tcn", "p1_mean_ens", "margin_mean_lstm",
                    "margin_mean_tcn", "margin_mean_ens", "E_pose", "E_face", "E_hands", "hand_presence"]:
            pc[key] = pc[key] / n

        acc_overall["lstm"] += pc["acc_lstm"] * (n / max(1, N_total))
        acc_overall["tcn"] += pc["acc_tcn"] * (n / max(1, N_total))
        acc_overall["ens"] += pc["acc_ens"] * (n / max(1, N_total))

        top3_overall["lstm"] += pc["top3_lstm"] * (n / max(1, N_total))
        top3_overall["tcn"] += pc["top3_tcn"] * (n / max(1, N_total))
        top3_overall["ens"] += pc["top3_ens"] * (n / max(1, N_total))

        # Top confusion for quick read
        def top_conf_str(cnt: Counter):
            if not cnt:
                return ""
            k, v = cnt.most_common(1)[0]
            return f"{labels[k]} ({v})"
        rows.append({
            "class": pc["label"],
            "n": pc["n"],
            "acc_lstm": pc["acc_lstm"], "acc_tcn": pc["acc_tcn"], "acc_ens": pc["acc_ens"],
            "top3_lstm": pc["top3_lstm"], "top3_tcn": pc["top3_tcn"], "top3_ens": pc["top3_ens"],
            "p1_lstm": pc["p1_mean_lstm"], "p1_tcn": pc["p1_mean_tcn"], "p1_ens": pc["p1_mean_ens"],
            "margin_lstm": pc["margin_mean_lstm"], "margin_tcn": pc["margin_mean_tcn"], "margin_ens": pc["margin_mean_ens"],
            "E_pose": pc["E_pose"], "E_face": pc["E_face"], "E_hands": pc["E_hands"],
            "hand_presence": pc["hand_presence"],
            "top_conf_lstm": top_conf_str(pc["confusions_lstm"]),
            "top_conf_tcn": top_conf_str(pc["confusions_tcn"]),
            "top_conf_ens": top_conf_str(pc["confusions_ens"]),
        })

    # Save per-class CSV
    import csv
    per_class_csv = os.path.join(out_dir, "diagnostics_per_class.csv")
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Save confusions
    def save_conf_csv(tag):
        path = os.path.join(out_dir, f"confusions_{tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)
            M = confusions_mat[tag]
            for i, lab in enumerate(labels):
                writer.writerow([lab] + list(M[i]))
    if lstm is not None:
        save_conf_csv("lstm")
    if tcn is not None:
        save_conf_csv("tcn")
    save_conf_csv("ens")

    # Save per-clip errors
    for tag, rows_err in per_clip_err_rows.items():
        if not rows_err:
            continue
        path = os.path.join(out_dir, f"per_clip_errors_{tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_err[0].keys()))
            writer.writeheader()
            writer.writerows(rows_err)

    # Print summary
    print("\n=== OVERALL ===")
    print(
        f"LSTM  Top-1: {acc_overall['lstm']:.4f}  Top-3: {top3_overall['lstm']:.4f}")
    print(
        f"TCN   Top-1: {acc_overall['tcn'] :.4f}  Top-3: {top3_overall['tcn'] :.4f}")
    print(
        f"ENS   Top-1: {acc_overall['ens'] :.4f}  Top-3: {top3_overall['ens'] :.4f}")
    print(f"[Wrote] {per_class_csv}")
    if lstm is not None:
        print("[Wrote] confusions_lstm.csv  per_clip_errors_lstm.csv")
    if tcn is not None:
        print("[Wrote] confusions_tcn.csv   per_clip_errors_tcn.csv")
    print("[Wrote] confusions_ens.csv    per_clip_errors_ens.csv")

    # Worst classes by Ensemble (Top-1)
    rows_sorted = sorted(rows, key=lambda r: (r["acc_ens"], r["p1_ens"]))
    print("\n=== WORST CLASSES (by Ensemble) ===")
    for r in rows_sorted[:20]:
        print(f"{r['class']:<24} acc_ens={r['acc_ens']:.2f}  p1={r['p1_ens']:.2f}  "
              f"margin={r['margin_ens']:.2f}  Ehands={r['E_hands']:.4f}  "
              f"hand%={r['hand_presence']:.2f}  top_conf={r['top_conf_ens']}")


def main():
    ap = argparse.ArgumentParser(
        description="Diagnostics for ISL v5 dataset (LSTM/TCN/Ensemble)")
    ap.add_argument("--split_root", type=str, required=True)
    ap.add_argument("--splits", type=str, default="val,test",
                    help="comma sep: train,val,test")
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--add_deltas", action="store_true")

    ap.add_argument("--lstm_weights", type=str, default=None)
    ap.add_argument("--tcn_weights", type=str, default=None)

    ap.add_argument("--k_per_class", type=int, default=None)
    ap.add_argument("--total_samples", type=int, default=None)

    ap.add_argument("--tta_pool", type=str,
                    choices=["mean", "max"], default="max")
    ap.add_argument("--no_shift", action="store_true")
    ap.add_argument("--no_warp", action="store_true")
    ap.add_argument("--no_hand_focus", action="store_true")

    ap.add_argument("--ens_w_tcn", type=float, default=0.8)
    ap.add_argument("--ens_w_lstm", type=float, default=0.2)

    ap.add_argument("--out_dir", type=str, default="diag_out")
    args = ap.parse_args()

    run_diagnostics(
        split_root=args.split_root,
        splits=[s.strip() for s in args.splits.split(",") if s.strip()],
        labels_path=args.labels,
        frames=args.frames,
        add_deltas=args.add_deltas,
        lstm_weights=args.lstm_weights,
        tcn_weights=args.tcn_weights,
        k_per_class=args.k_per_class,
        total_samples=args.total_samples,
        tta_pool=args.tta_pool,
        do_shift=(not args.no_shift),
        do_warp=(not args.no_warp),
        do_hand_focus=(not args.no_hand_focus),
        ens_w_tcn=args.ens_w_tcn,
        ens_w_lstm=args.ens_w_lstm,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    # Keep CPU threads modest (Windows-friendly)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    main()
