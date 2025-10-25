#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

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

POSE_START, POSE_END = 0, POSE_DIM
FACE_START, FACE_END = POSE_END, POSE_END + FACE_DIM
LH_START, LH_END = FACE_END, FACE_END + L_HAND_DIM
RH_START, RH_END = LH_END, LH_END + R_HAND_DIM

DEFAULT_SEQ_LEN = 48
AUTOTUNE = tf.data.AUTOTUNE

# -------------------- Repro / threading -----------------


def set_env(num_threads=None):
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
                if seq.is_file():
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
    x_idx, y_idx, z_idx = [], [], []
    base = 0
    for i in range(POSE_LM):
        x_idx.append(base + i*4 + 0)
        y_idx.append(base + i*4 + 1)
        z_idx.append(base + i*4 + 2)
    base += POSE_DIM
    for i in range(FACE_LM):
        x_idx.append(base + i*3 + 0)
        y_idx.append(base + i*3 + 1)
        z_idx.append(base + i*3 + 2)
    base += FACE_DIM
    for i in range(HAND_LM):
        x_idx.append(base + i*3 + 0)
        y_idx.append(base + i*3 + 1)
        z_idx.append(base + i*3 + 2)
    base += L_HAND_DIM
    for i in range(HAND_LM):
        x_idx.append(base + i*3 + 0)
        y_idx.append(base + i*3 + 1)
        z_idx.append(base + i*3 + 2)
    return np.array(x_idx, np.int32), np.array(y_idx, np.int32), np.array(z_idx, np.int32)


X_IDX, Y_IDX, Z_IDX = build_index_masks()

# -------------------- Augmentation -----------------


def _np_seq_aug_xy(x_np, xy_scale=0.01, xy_shift=0.01, z_noise=0.003):
    x = np.array(x_np, dtype=np.float32)
    T = x.shape[0]
    for t in range(T):
        s = 1.0 + np.random.uniform(-xy_scale, xy_scale)
        dx = np.random.uniform(-xy_shift, xy_shift)
        dy = np.random.uniform(-xy_shift, xy_shift)
        x[t, X_IDX] = np.clip(x[t, X_IDX] * s + dx, 0.0, 1.0)
        x[t, Y_IDX] = np.clip(x[t, Y_IDX] * s + dy, 0.0, 1.0)
        x[t, Z_IDX] = x[t, Z_IDX] + \
            np.random.normal(0.0, z_noise, size=len(Z_IDX))
    return x


def _np_time_shift(x, max_shift=4):
    if max_shift <= 0:
        return x
    s = np.random.randint(-max_shift, max_shift + 1)
    if s == 0:
        return x
    if s > 0:
        return np.concatenate([x[s:], np.repeat(x[-1:], s, axis=0)], axis=0)
    s = -s
    return np.concatenate([np.repeat(x[:1], s, axis=0), x[:-s]], axis=0)


def _np_temporal_cutout(x, max_cut=4, prob=0.5):
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
    x[:, FACE_START:FACE_END] *= face_scale
    if np.random.rand() < face_dropout:
        x[:, FACE_START:FACE_END] = 0.0
    return x


def build_seq_augment_fn(
    xy_scale=0.01, xy_shift=0.01, z_noise=0.003,
    time_shift=4, face_scale=0.4, face_dropout=0.5,
    temporal_cutout=4
):
    def tf_aug(x, y):
        def _aug(a):
            a = _np_seq_aug_xy(a, xy_scale, xy_shift, z_noise)
            a = _np_time_shift(a, max_shift=time_shift)
            a = _np_temporal_cutout(a, max_cut=temporal_cutout, prob=0.5)
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
    temporal_cutout=4,
    shuffle_buf=2048
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
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(cache)

    if augment:
        ds = ds.map(
            build_seq_augment_fn(xy_scale, xy_shift, z_noise,
                                 time_shift, face_scale, face_dropout,
                                 temporal_cutout),
            num_parallel_calls=AUTOTUNE,
            deterministic=deterministic
        )

    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# -------------------- Loss -------------------------


def make_sparse_ce_with_smoothing(num_classes: int, epsilon: float):
    if epsilon is None or epsilon <= 1e-8:
        return tf.keras.losses.SparseCategoricalCrossentropy()
    cce = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=float(epsilon))

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=y_pred.dtype)
        return cce(y_one_hot, y_pred)
    return loss

# -------------------- Attention Layer --------------


class TemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = tf.keras.layers.Dense(units, activation="tanh")
        self.score = tf.keras.layers.Dense(1, activation=None)
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, x):
        s = self.proj(x)                 # (B, T, units)
        s = self.score(s)                # (B, T, 1)
        w = self.softmax(s)              # (B, T, 1)
        return tf.reduce_sum(x * w, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# -------------------- Models -----------------------


def build_lstm_model(
    num_classes,
    seq_len=DEFAULT_SEQ_LEN,
    feat_dim=FEAT_DIM,
    lr=5e-4,
    use_attention=True,
    dropout=0.5,
    label_smoothing=0.05,
    lstm_w1=128, lstm_w2=64,
    l2_reg=1e-4
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
    x = K.layers.Dropout(0.3)(x)
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
    # NOTE: No metrics here (we use EvalCallback to avoid Keras bug)
    model.compile(optimizer=K.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
                  loss=loss_fn, metrics=[])
    return model


def TCNBlock(filters, kernel_size=5, dilation_base=2, n_stacks=2, dropout=0.2, l2_reg=1e-4):
    reg = regularizers.l2(l2_reg)

    def f(x):
        for s in range(n_stacks):
            dil = dilation_base ** s
            y = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                       dilation_rate=dil, kernel_regularizer=reg)(x)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.Activation("relu")(y)
            y = tf.keras.layers.Dropout(dropout)(y)
            if x.shape[-1] != y.shape[-1]:
                x = tf.keras.layers.Conv1D(
                    filters, 1, padding="same", kernel_regularizer=reg)(x)
            x = tf.keras.layers.Add()([x, y])
        return x
    return f


def build_tcn_model(
    num_classes,
    seq_len=DEFAULT_SEQ_LEN,
    feat_dim=FEAT_DIM,
    lr=5e-4,
    dropout=0.5,
    label_smoothing=0.05,
    l2_reg=1e-4
):
    K = tf.keras
    inp = K.Input(shape=(seq_len, feat_dim), name="seq")
    x = K.layers.Dense(256, activation="relu")(inp)
    x = K.layers.Dropout(0.25)(x)
    x = TCNBlock(256, kernel_size=5, dilation_base=2,
                 n_stacks=3, dropout=0.25, l2_reg=l2_reg)(x)
    x = TCNBlock(256, kernel_size=3, dilation_base=2,
                 n_stacks=2, dropout=0.25, l2_reg=l2_reg)(x)
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
    model.compile(optimizer=K.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
                  loss=loss_fn, metrics=[])
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

# -------------------- Custom Eval Callback ----------


class EvalCallback(tf.keras.callbacks.Callback):
    """Compute val_accuracy and val_top3 at end of each epoch and inject into logs."""

    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        acc = tf.keras.metrics.SparseCategoricalAccuracy()
        top3 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
        for xb, yb in self.val_ds:
            preds = self.model(xb, training=False)
            acc.update_state(yb, preds)
            top3.update_state(yb, preds)
        logs["val_accuracy"] = float(acc.result().numpy())
        logs["val_top3"] = float(top3.result().numpy())
        print(
            f"\n[EvalCallback] val_accuracy={logs['val_accuracy']:.4f}  val_top3={logs['val_top3']:.4f}")

# -------------------- Exporters --------------------


def export_all(model, save_dir, classes, export_tfjs=True, export_tflite=True):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save(save_dir / "final_model.keras", include_optimizer=False)
    with open(save_dir / "model_arch.json", "w", encoding="utf-8") as f:
        f.write(model.to_json())
    model.save_weights(save_dir / "model.weights.h5")
    tf.saved_model.save(model, str(save_dir / "saved_model"))

    with open(save_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "label2idx": {
                  c: i for i, c in enumerate(classes)}}, f, indent=2)

    if export_tfjs:
        try:
            import tensorflowjs as tfjs
            tfjs_dir = save_dir / "tfjs_model"
            tfjs_dir.mkdir(exist_ok=True)
            tfjs.converters.save_keras_model(model, str(tfjs_dir))
            print(f"[OK] TFJS model -> {tfjs_dir}")
        except Exception as e:
            print("[WARN] TFJS export failed:", e)

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
                    help="Disable augmentation")
    ap.add_argument("--save_dir", type=str, default="models/isl_seq_model")
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--cache_dir", type=str, default="")
    ap.add_argument("--num_threads", type=int, default=0)
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
    ap.add_argument("--temporal_cutout", type=int, default=4)
    ap.add_argument("--shuffle_buf", type=int, default=2048)

    # Capacity & L2
    ap.add_argument("--lstm_w1", type=int, default=128)
    ap.add_argument("--lstm_w2", type=int, default=64)
    ap.add_argument("--l2", type=float, default=1e-4)

    # Train subset controls
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="Cap per-class train items (0=disable)")
    ap.add_argument("--subset_frac", type=float, default=1.0,
                    help="Random fraction of train set (<=1.0)")

    args = ap.parse_args()

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

    # Balanced sub-sampling (optional)
    if args.max_per_class > 0 or args.subset_frac < 0.999:
        # do class-balanced cap and/or random fraction
        by_class = {}
        rng = random.Random(args.seed)
        for p, c in train_items:
            by_class.setdefault(c, []).append(p)
        new_items = []
        for c, paths in by_class.items():
            rng.shuffle(paths)
            if args.max_per_class > 0:
                paths = paths[:args.max_per_class]
            new_items.extend([(p, c) for p in paths])
        if args.subset_frac < 0.999:
            rng.shuffle(new_items)
            k = max(1, int(len(new_items) * args.subset_frac))
            new_items = new_items[:k]
        train_items = new_items
        # stats
        counts = {}
        for _, c in train_items:
            counts[c] = counts.get(c, 0) + 1
        print(f"[INFO] After sub-sampling: {len(train_items)} train items")
        print("[INFO] Per-class min/max:",
              min(counts.values()), max(counts.values()))

    # Datasets
    cache_tr = str(Path(args.cache_dir) /
                   "train.cache") if args.cache_dir else None
    cache_va = str(Path(args.cache_dir) /
                   "val.cache") if args.cache_dir else None
    cache_te = None

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

    # Build model
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
            lr=args.lr, dropout=args.dropout, label_smoothing=args.label_smoothing,
            l2_reg=args.l2
        )
    model.summary()

    # Class weights (balanced)
    cw = derive_class_weights(train_items, n_classes)
    print("[INFO] class_weights:", cw)

    # Save labels
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "labels.json"), "w") as f:
        json.dump({"classes": classes, "label2idx": {
                  c: i for i, c in enumerate(classes)}}, f, indent=2)

    # Callbacks (Eval first so checkpoints see the metrics in logs)
    eval_cb = EvalCallback(ds_va)
    callbacks = [
        eval_cb,
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.save_dir, "best.weights.h5"),
            monitor="val_accuracy", save_best_only=True, mode="max",
            save_weights_only=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.save_dir, "best_top3.weights.h5"),
            monitor="val_top3", save_best_only=True, mode="max",
            save_weights_only=True
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

    history = model.fit(
        ds_tr,
        validation_data=ds_va,  # for val_loss; acc/top3 come from EvalCallback
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

    # Load best weights (by val_accuracy)
    best_w = Path(args.save_dir) / "best.weights.h5"
    if best_w.is_file():
        try:
            model.load_weights(str(best_w))
            print("[INFO] Loaded best weights from disk.")
        except Exception as e:
            print("[WARN] Could not load best weights:", e)

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

    # Reports
    report = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)
    with open(Path(args.save_dir) / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(args.save_dir, "confusion_matrix.csv"),
               cm, fmt="%d", delimiter=",")

    # Export final model (weights currently best)
    export_all(
        model, save_dir=args.save_dir, classes=classes,
        export_tfjs=not args.no_tfjs, export_tflite=not args.no_tflite
    )
    print(f"[OK] Artifacts saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
