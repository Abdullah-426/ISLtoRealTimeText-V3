#!/usr/bin/env python3
from tensorflow.keras import regularizers
import os
import json
import argparse
from pathlib import Path
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# -------------------- Constants --------------------
POSE_LM = 33
FACE_LM = 468
HAND_LM = 21

POSE_DIM = POSE_LM * 4
FACE_DIM = FACE_LM * 3
L_HAND_DIM = HAND_LM * 3
R_HAND_DIM = HAND_LM * 3
FEAT_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662

# flat index ranges for sub-blocks in (1662,) vector
POSE_START, POSE_END = 0, POSE_DIM
FACE_START, FACE_END = POSE_END, POSE_END + FACE_DIM
LH_START, LH_END = FACE_END, FACE_END + L_HAND_DIM
RH_START, RH_END = LH_END, LH_END + R_HAND_DIM

DEFAULT_SEQ_LEN = 48
AUTOTUNE = tf.data.AUTOTUNE

# --------------- Repro / threading -----------------


def set_env(num_threads=None):
    # Optional: limit intra/inter threads to avoid CPU oversubscription
    if num_threads:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(max(1, num_threads // 2))


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def maybe_enable_mixed_precision(enable: bool):
    if enable:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision enabled (float16 compute).")
        except Exception as e:
            print("[WARN] Mixed precision not available:", e)

# -------------------- Data discovery ----------------


def list_classes(split_dir: Path):
    train_dir = split_dir / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"[ERROR] '{split_dir}' must contain 'train/'.")
    classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not classes:
        raise SystemExit(f"[ERROR] No class folders in {train_dir}")
    return classes


def enumerate_clips(split_dir: Path, split: str, classes):
    items = []
    for ci, cname in enumerate(classes):
        cdir = split_dir / split / cname
        if not cdir.is_dir():
            continue
        for clip in sorted(cdir.iterdir()):
            if clip.is_dir() and clip.name.startswith("clip_"):
                seq = clip / "sequence.npy"
                if seq.is_file():  # fast path: we require sequence.npy
                    items.append((str(clip), ci))
    return items

# -------------------- Loaders ----------------------


def load_sequence_numpy(clip_dir: str, T: int) -> np.ndarray:
    seq_path = Path(clip_dir) / "sequence.npy"
    arr = np.load(seq_path)
    if arr.shape != (T, FEAT_DIM):
        raise ValueError(f"Bad sequence shape {arr.shape} in {seq_path}")
    return arr.astype(np.float32)


def build_index_masks():
    """Precompute flat indices for x, y, z in the flattened (1662,) vector."""
    x_idx, y_idx, z_idx = [], [], []
    base = 0
    for i in range(POSE_LM):  # stride 4: x,y,z,v
        x_idx.append(base + i*4 + 0)
        y_idx.append(base + i*4 + 1)
        z_idx.append(base + i*4 + 2)
    base += POSE_DIM
    for i in range(FACE_LM):  # stride 3
        x_idx.append(base + i*3 + 0)
        y_idx.append(base + i*3 + 1)
        z_idx.append(base + i*3 + 2)
    base += FACE_DIM
    for i in range(HAND_LM):  # LH stride 3
        x_idx.append(base + i*3 + 0)
        y_idx.append(base + i*3 + 1)
        z_idx.append(base + i*3 + 2)
    base += L_HAND_DIM
    for i in range(HAND_LM):  # RH stride 3
        x_idx.append(base + i*3 + 0)
        y_idx.append(base + i*3 + 1)
        z_idx.append(base + i*3 + 2)
    return np.array(x_idx, np.int32), np.array(y_idx, np.int32), np.array(z_idx, np.int32)


X_IDX, Y_IDX, Z_IDX = build_index_masks()

# -------------------- Augmentation -----------------


def _np_seq_aug_xy(x_np, xy_scale=0.01, xy_shift=0.01, z_noise=0.003):
    """Conservative spatial jitter per frame; keep zeros intact."""
    x = np.array(x_np, dtype=np.float32)
    T = x.shape[0]
    for t in range(T):
        s = 1.0 + np.random.uniform(-xy_scale, xy_scale)
        dx = np.random.uniform(-xy_shift, xy_shift)
        dy = np.random.uniform(-xy_shift, xy_shift)
        # x,y in [0,1]
        x[t, X_IDX] = np.clip(x[t, X_IDX] * s + dx, 0.0, 1.0)
        x[t, Y_IDX] = np.clip(x[t, Y_IDX] * s + dy, 0.0, 1.0)
        # z unconstrained
        x[t, Z_IDX] = x[t, Z_IDX] + \
            np.random.normal(0.0, z_noise, size=len(Z_IDX))
    return x


def _np_time_shift(x, max_shift=4):
    """Random temporal shift ±max_shift with edge-padding."""
    if max_shift <= 0:
        return x
    s = np.random.randint(-max_shift, max_shift + 1)
    if s == 0:
        return x
    if s > 0:
        return np.concatenate([x[s:], np.repeat(x[-1:], s, axis=0)], axis=0)
    else:
        s = -s
        return np.concatenate([np.repeat(x[:1], s, axis=0), x[:-s]], axis=0)


def _np_temporal_cutout(x, max_cut=4, prob=0.5):
    """Zero a random contiguous block of frames of length [1, max_cut]."""
    if max_cut <= 0 or np.random.rand() > prob:
        return x
    T = x.shape[0]
    L = np.random.randint(1, max_cut + 1)
    if L >= T:
        return x
    start = np.random.randint(0, T - L + 1)
    x = x.copy()
    x[start:start+L, :] = 0.0
    return x


def _np_apply_face_ops(x, face_scale=0.4, face_dropout=0.5):
    """Reduce face dominance and randomly drop face to avoid identity leakage."""
    # scale face features down
    x[:, FACE_START:FACE_END] *= face_scale
    # randomly drop face entirely to force hand/pose learning
    if np.random.rand() < face_dropout:
        x[:, FACE_START:FACE_END] = 0.0
    return x


def build_seq_augment_fn(
    xy_scale=0.01, xy_shift=0.01, z_noise=0.003,
    time_shift=4, face_scale=0.4, face_dropout=0.5,
    temporal_cutout=4  # NEW
):
    def tf_aug(x, y):
        def _aug(a):
            a = _np_seq_aug_xy(a, xy_scale, xy_shift, z_noise)
            a = _np_time_shift(a, max_shift=time_shift)
            a = _np_temporal_cutout(
                a, max_cut=temporal_cutout, prob=0.5)  # NEW
            a = _np_apply_face_ops(a, face_scale, face_dropout)
            return a.astype(np.float32)
        x_out = tf.py_function(_aug, inp=[x], Tout=tf.float32)
        x_out.set_shape([None, FEAT_DIM])
        return x_out, y
    return tf_aug


# -------------------- tf.data ----------------------


def make_dataset(
    items, seq_len, batch, shuffle, seed,
    augment=False, cache=None, deterministic=False,
    time_shift=4, face_scale=0.4, face_dropout=0.5,
    xy_scale=0.01, xy_shift=0.01, z_noise=0.003,
    temporal_cutout=4,            # NEW
    shuffle_buf=2048              # NEW
):
    def gen():
        for clip_dir, y in items:
            try:
                x = load_sequence_numpy(clip_dir, seq_len)
                yield x, y
            except Exception:
                continue

    sig = (
        tf.TensorSpec(shape=(seq_len, FEAT_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(shuffle_buf, max(1024, len(items))),
                        seed=seed, reshuffle_each_iteration=True)

    if cache:
        # ensure cache parent exists
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(cache)  # path (disk) or RAM if cache=True

    if augment:
        ds = ds.map(
            build_seq_augment_fn(xy_scale, xy_shift, z_noise,
                                 time_shift, face_scale, face_dropout,
                                 temporal_cutout),               # pass cutout
            num_parallel_calls=AUTOTUNE,
            deterministic=deterministic
        )

    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# -------------------- Models -----------------------


def temporal_attention(x, units=128):
    """
    Functional temporal attention over time dimension.
    x: (batch, T, C) -> (batch, C)
    """
    scores = tf.keras.layers.Dense(
        units, activation="tanh")(x)  # (B, T, units)
    scores = tf.keras.layers.Dense(1, activation=None)(scores)   # (B, T, 1)
    weights = tf.keras.layers.Softmax(axis=1)(scores)            # (B, T, 1)
    context = tf.keras.layers.Multiply()([x, weights])           # (B, T, C)
    context = tf.keras.layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1))(context)  # (B, C)
    return context


def make_sparse_ls_loss(num_classes: int, label_smoothing: float):
    """
    If label_smoothing > 0, wrap CategoricalCrossentropy(label_smoothing=..)
    and convert sparse labels to one-hot on the fly.
    Else, use standard SparseCategoricalCrossentropy.
    """
    if label_smoothing is None or label_smoothing <= 1e-8:
        return tf.keras.losses.SparseCategoricalCrossentropy()

    cce = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing)

    def loss(y_true, y_pred):
        # y_true: (B,) int
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        return cce(y_true_oh, y_pred)
    return loss


def make_sparse_ce_with_smoothing(num_classes: int, epsilon: float):
    """
    Stable sparse-friendly loss with manual label smoothing.
    Works in TF/Keras 3 on Windows (avoids optree/unknown-rank issues).
    """
    epsilon = float(max(0.0, epsilon))

    def loss(y_true, y_pred):
        # y_true: (B,) int
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)           # (B,)
        # one-hot -> (B, C)
        y_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=y_pred.dtype)

        if epsilon > 0.0:
            eps = tf.cast(epsilon, y_pred.dtype)
            c = tf.cast(num_classes, y_pred.dtype)
            # y_smooth = (1-ε)*one_hot + ε/C
            y_smooth = y_one_hot * (1.0 - eps) + eps / c
        else:
            y_smooth = y_one_hot

        # Keras backend categorical CE (returns per-example loss)
        losses = tf.keras.backend.categorical_crossentropy(
            y_smooth, y_pred, from_logits=False, axis=-1
        )
        return tf.reduce_mean(losses)   # Let Keras reduction handle weights
    return loss


class TemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = tf.keras.layers.Dense(units, activation="tanh")
        self.score = tf.keras.layers.Dense(1, activation=None)
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, x):
        # x: (B, T, C)
        s = self.proj(x)                  # (B, T, units)
        s = self.score(s)                 # (B, T, 1)
        w = self.softmax(s)               # (B, T, 1)
        # Weighted sum over time without Lambda:
        # (B, C) = sum_t (w_t * x_t)
        return tf.reduce_sum(x * w, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


def build_lstm_model(
    num_classes,
    seq_len=DEFAULT_SEQ_LEN,
    feat_dim=FEAT_DIM,
    lr=5e-4,
    use_attention=True,
    dropout=0.5,
    label_smoothing=0.05,
    lstm_w1=128, lstm_w2=64,      # NEW
    l2_reg=1e-4                   # NEW
):
    K = tf.keras
    reg = regularizers.l2(l2_reg)

    inp = K.Input(shape=(seq_len, feat_dim), name="seq")

    x = K.layers.Dense(512, activation=None, kernel_regularizer=reg)(inp)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    x = K.layers.Dropout(0.35)(x)

    x = K.layers.Bidirectional(
        K.layers.LSTM(lstm_w1, return_sequences=True,
                      kernel_regularizer=reg, recurrent_regularizer=reg)
    )(x)
    x = K.layers.Dropout(0.3)(x)  # extra dropout between LSTMs

    x = K.layers.Bidirectional(
        K.layers.LSTM(lstm_w2, return_sequences=use_attention,
                      kernel_regularizer=reg, recurrent_regularizer=reg)
    )(x)

    if use_attention:
        x = TemporalAttentionLayer(units=128, name="temporal_attention")(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = K.layers.Dense(256, activation=None, kernel_regularizer=reg)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    x = K.layers.Dropout(dropout)(x)

    x = K.layers.Dense(128, activation=None, kernel_regularizer=reg)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)

    out = K.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = K.Model(inp, out, name="ISL_BiLSTM_Attn")
    loss_fn = make_sparse_ce_with_smoothing(num_classes, label_smoothing)
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=loss_fn,
        metrics=["accuracy", K.metrics.SparseTopKCategoricalAccuracy(
            k=3, name="top3")]
    )
    return model


def TCNBlock(filters, kernel_size=5, dilation_base=2, n_stacks=2, dropout=0.2):
    def f(x):
        for s in range(n_stacks):
            dil = dilation_base ** s
            y = tf.keras.layers.Conv1D(
                filters, kernel_size, padding="causal", dilation_rate=dil)(x)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.Activation("relu")(y)
            y = tf.keras.layers.Dropout(dropout)(y)
            if x.shape[-1] != y.shape[-1]:
                x = tf.keras.layers.Conv1D(filters, 1, padding="same")(x)
            x = tf.keras.layers.Add()([x, y])
        return x
    return f


def build_tcn_model(
    num_classes,
    seq_len=DEFAULT_SEQ_LEN,
    feat_dim=FEAT_DIM,
    lr=5e-4,
    dropout=0.5,
    label_smoothing=0.05
):
    K = tf.keras
    inp = K.Input(shape=(seq_len, feat_dim), name="seq")

    x = K.layers.Dense(256, activation="relu")(inp)
    x = K.layers.Dropout(0.25)(x)
    x = TCNBlock(256, kernel_size=5, dilation_base=2,
                 n_stacks=3, dropout=0.25)(x)
    x = TCNBlock(256, kernel_size=3, dilation_base=2,
                 n_stacks=2, dropout=0.25)(x)

    x = K.layers.GlobalAveragePooling1D()(x)
    x = K.layers.Dense(256, activation=None)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    x = K.layers.Dropout(dropout)(x)

    x = K.layers.Dense(128, activation=None)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)

    out = K.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = K.Model(inp, out, name="ISL_TCN")
    loss_fn = make_sparse_ce_with_smoothing(num_classes, label_smoothing)
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=loss_fn,
        metrics=["accuracy", K.metrics.SparseTopKCategoricalAccuracy(
            k=3, name="top3")]
    )

    return model

# -------------------- Class weights ----------------


def compute_cw(y_labels, n_classes):
    classes = np.arange(n_classes)
    w = compute_class_weight(class_weight="balanced",
                             classes=classes, y=y_labels)
    return {i: float(w[i]) for i in range(n_classes)}


def derive_class_weights(train_items, n_classes):
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, ci in train_items:
        counts[ci] += 1
    y = np.concatenate([np.full(c, i, dtype=np.int32)
                       for i, c in enumerate(counts)])
    return compute_cw(y, n_classes)

# -------------------- Exporters --------------------


def export_all(model, save_dir, classes, export_tfjs=True, export_tflite=True):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Keras native
    model.save(save_dir / "final_model.keras")
    with open(save_dir / "model_arch.json", "w", encoding="utf-8") as f:
        f.write(model.to_json())
    # FIX: Keras requires .weights.h5 suffix
    model.save_weights(save_dir / "model.weights.h5")

    # SavedModel (good for tfjs converter and serving)
    tf.saved_model.save(model, str(save_dir / "saved_model"))

    # Labels
    with open(save_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "label2idx": {
                  c: i for i, c in enumerate(classes)}}, f, indent=2)

    # TFJS
    if export_tfjs:
        try:
            import tensorflowjs as tfjs
            tfjs_dir = save_dir / "tfjs_model"
            tfjs_dir.mkdir(exist_ok=True)
            tfjs.converters.save_keras_model(model, str(tfjs_dir))
            print(f"[OK] TFJS model -> {tfjs_dir}")
        except Exception as e:
            print("[WARN] TFJS export failed:", e)

    # TFLite (float16 + dynamic-int8)
    if export_tflite:
        try:
            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_int8 = conv.convert()
            (save_dir / "model_int8.tflite").write_bytes(tflite_int8)
            print("[OK] TFLite dynamic-int8 -> model_int8.tflite")

            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            conv.target_spec.supported_types = [tf.float16]
            tflite_f16 = conv.convert()
            (save_dir / "model_float16.tflite").write_bytes(tflite_f16)
            print("[OK] TFLite float16 -> model_float16.tflite")
        except Exception as e:
            print("[WARN] TFLite export failed:", e)

# -------------------- Main ------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Train ISL sequence model (LSTM/TCN) and export for web/mobile.")
    ap.add_argument("--split_root", type=str, default="Dataset_Split")
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str,
                    choices=["lstm", "tcn"], default="lstm")
    ap.add_argument("--use_attention", action="store_true",
                    help="Temporal attention (LSTM only)")
    ap.add_argument("--no_aug", action="store_true",
                    help="Disable light augmentation")
    ap.add_argument("--save_dir", type=str, default="models/isl_seq_model")
    ap.add_argument("--mixed_precision", action="store_true",
                    help="Enable float16 mixed precision (GPU)")
    ap.add_argument("--cache_dir", type=str, default="",
                    help="Optional tf.data cache directory (disk).")
    ap.add_argument("--num_threads", type=int, default=0,
                    help="Optional thread hint for CPU ops.")
    ap.add_argument("--no_tfjs", action="store_true")
    ap.add_argument("--no_tflite", action="store_true")
    # Regularization & Augmentation controls
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--face_scale", type=float, default=0.4)
    ap.add_argument("--face_dropout", type=float, default=0.5)
    ap.add_argument("--time_shift", type=int, default=4)
    ap.add_argument("--xy_scale", type=float, default=0.01)
    ap.add_argument("--xy_shift", type=float, default=0.01)
    ap.add_argument("--z_noise", type=float, default=0.003)

    # NEW: model capacity & regularization knobs
    ap.add_argument("--lstm_w1", type=int, default=128,
                    help="First BiLSTM width (per direction)")
    ap.add_argument("--lstm_w2", type=int, default=64,
                    help="Second BiLSTM width (per direction)")
    ap.add_argument("--l2", type=float, default=1e-4, help="L2 weight decay")
    ap.add_argument("--temporal_cutout", type=int, default=4,
                    help="Max frames to zero in a contiguous block")
    ap.add_argument("--shuffle_buf", type=int, default=2048,
                    help="Shuffle buffer size for training ds")

    args = ap.parse_args()

    # env & reproducibility
    set_env(args.num_threads if args.num_threads > 0 else None)
    set_seeds(args.seed)
    maybe_enable_mixed_precision(args.mixed_precision)

    split_root = Path(args.split_root)
    classes = list_classes(split_root)
    n_classes = len(classes)
    print(
        f"[INFO] classes={n_classes}  seq_len={args.seq_len}  feat_dim={FEAT_DIM}")

    train_items = enumerate_clips(split_root, "train", classes)
    val_items = enumerate_clips(split_root, "val", classes)
    test_items = enumerate_clips(split_root, "test", classes)
    print(
        f"[INFO] train={len(train_items)}  val={len(val_items)}  test={len(test_items)}")

    # Cache paths
    cache_tr = str(Path(args.cache_dir) /
                   "train.cache") if args.cache_dir else None
    cache_va = str(Path(args.cache_dir) /
                   "val.cache") if args.cache_dir else None
    cache_te = None  # not needed for test

    ds_tr = make_dataset(
        train_items, args.seq_len, args.batch, shuffle=True,
        seed=args.seed, augment=not args.no_aug, cache=cache_tr, deterministic=False,
        time_shift=args.time_shift, face_scale=args.face_scale, face_dropout=args.face_dropout,
        xy_scale=args.xy_scale, xy_shift=args.xy_shift, z_noise=args.z_noise,
        temporal_cutout=args.temporal_cutout, shuffle_buf=args.shuffle_buf
    )
    ds_va = make_dataset(
        val_items, args.seq_len, args.batch, shuffle=False,
        seed=args.seed, augment=False, cache=cache_va, deterministic=True,
        time_shift=0, face_scale=1.0, face_dropout=0.0,
        temporal_cutout=0, shuffle_buf=1024
    )
    ds_te = make_dataset(
        test_items, args.seq_len, args.batch, shuffle=False,
        seed=args.seed, augment=False, cache=cache_te, deterministic=True,
        time_shift=0, face_scale=1.0, face_dropout=0.0,
        temporal_cutout=0, shuffle_buf=1024
    )

    # Model
    if args.model == "lstm":
        model = build_lstm_model(
            n_classes, seq_len=args.seq_len, feat_dim=FEAT_DIM,
            lr=args.lr, use_attention=args.use_attention,
            dropout=args.dropout, label_smoothing=args.label_smoothing,
            lstm_w1=args.lstm_w1, lstm_w2=args.lstm_w2, l2_reg=args.l2
        )
    else:
        model = build_tcn_model(
            n_classes, seq_len=args.seq_len, feat_dim=FEAT_DIM,
            lr=args.lr, dropout=args.dropout, label_smoothing=args.label_smoothing
        )
    model.summary()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "labels.json"), "w") as f:
        json.dump({"classes": classes, "label2idx": {
                  c: i for i, c in enumerate(classes)}}, f, indent=2)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.save_dir, "best.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-5
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(
            args.save_dir, "training_log.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    cw = derive_class_weights(train_items, n_classes)
    print("[INFO] class_weights:", cw)

    history = model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=cw,
        verbose=1
    )

    # Save training history
    hist_dict = {k: [float(vv) for vv in vals]
                 for k, vals in history.history.items()}
    with open(Path(args.save_dir) / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=2)

    # Load best checkpoint (ensures best export)
    best_path = Path(args.save_dir) / "best.keras"
    if best_path.is_file():
        try:
            model = tf.keras.models.load_model(
                best_path,
                compile=False,
                custom_objects={
                    "TemporalAttentionLayer": TemporalAttentionLayer}
            )
            print("[INFO] Loaded best checkpoint from disk.")
        except Exception as e:
            print(
                "[WARN] Could not load best.keras, exporting in-memory model instead:", e)

    # Test evaluation
    y_true, y_pred, y_prob_all = [], [], []
    for xb, yb in ds_te:
        prob = model.predict(xb, verbose=0)
        pred = np.argmax(prob, axis=1)
        y_true.append(yb.numpy())
        y_pred.append(pred)
        y_prob_all.append(prob)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob_all, axis=0)
    top3 = np.any(np.argsort(-y_prob, axis=1)
                  [:, :3] == y_true[:, None], axis=1).mean()
    acc = (y_true == y_pred).mean()
    print(f"\n[TEST] acc={acc:.4f}  top3={top3:.4f}")

    # Classification report & confusion matrix
    report = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)
    with open(Path(args.save_dir) / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(args.save_dir, "confusion_matrix.csv"),
               cm, fmt="%d", delimiter=",")

    # Save & export (Keras, SavedModel, TFJS, TFLite)
    export_all(
        model,
        save_dir=args.save_dir,
        classes=classes,
        export_tfjs=not args.no_tfjs,
        export_tflite=not args.no_tflite
    )

    print(f"[OK] Artifacts saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
