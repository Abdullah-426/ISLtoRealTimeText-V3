#!/usr/bin/env python3
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


def _np_seq_aug(x_np, xy_scale=0.01, xy_shift=0.01, z_noise=0.003):
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


def build_seq_augment_fn(xy_scale=0.01, xy_shift=0.01, z_noise=0.003):
    def tf_aug(x, y):
        x = tf.py_function(lambda a: _np_seq_aug(a, xy_scale, xy_shift, z_noise),
                           inp=[x], Tout=tf.float32)
        x.set_shape([None, FEAT_DIM])
        return x, y
    return tf_aug


def make_dataset(items, seq_len, batch, shuffle, seed,
                 augment=False, cache=None, deterministic=False):
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
        ds = ds.shuffle(buffer_size=min(8192, len(items)),
                        seed=seed, reshuffle_each_iteration=True)
    if cache:
        ds = ds.cache(cache)  # path (disk) or RAM if cache=True
    if augment:
        ds = ds.map(build_seq_augment_fn(),
                    num_parallel_calls=AUTOTUNE, deterministic=deterministic)
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# -------------------- Models -----------------------


def temporal_attention(x, units=128):
    """
    Functional temporal attention over time dimension.
    x: (batch, T, C)
    returns: (batch, C)
    """
    # Score per time step
    scores = tf.keras.layers.Dense(
        units, activation="tanh")(x)   # (B, T, units)
    scores = tf.keras.layers.Dense(1, activation=None)(scores)    # (B, T, 1)
    weights = tf.keras.layers.Softmax(axis=1)(scores)             # (B, T, 1)

    # Weighted sum over time
    context = tf.keras.layers.Multiply()([x, weights])            # (B, T, C)
    context = tf.keras.layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1))(context)  # (B, C)
    return context


def build_lstm_model(num_classes, seq_len=DEFAULT_SEQ_LEN, feat_dim=FEAT_DIM, lr=1e-3,
                     use_attention=True, dropout=0.35):
    K = tf.keras
    inp = K.Input(shape=(seq_len, feat_dim), name="seq")

    # Input projection + norm
    x = K.layers.Dense(512, activation=None)(inp)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    x = K.layers.Dropout(0.25)(x)

    # BiLSTM stack
    x = K.layers.Bidirectional(K.layers.LSTM(256, return_sequences=True))(x)
    x = K.layers.Bidirectional(K.layers.LSTM(
        128, return_sequences=use_attention))(x)

    if use_attention:
        x = temporal_attention(x)   # << use the functional block here
    else:
        x = K.layers.GlobalAveragePooling1D()(x)

    x = K.layers.Dense(256, activation=None)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    x = K.layers.Dropout(dropout)(x)

    x = K.layers.Dense(128, activation=None)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)

    out = K.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = K.Model(inp, out, name="ISL_BiLSTM_Attn")
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
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


def build_tcn_model(num_classes, seq_len=DEFAULT_SEQ_LEN, feat_dim=FEAT_DIM, lr=1e-3, dropout=0.25):
    K = tf.keras
    inp = K.Input(shape=(seq_len, feat_dim), name="seq")

    x = K.layers.Dense(256, activation="relu")(inp)
    x = K.layers.Dropout(0.2)(x)
    x = TCNBlock(256, kernel_size=5, dilation_base=2,
                 n_stacks=3, dropout=0.2)(x)
    x = TCNBlock(256, kernel_size=3, dilation_base=2,
                 n_stacks=2, dropout=0.2)(x)

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
    model.compile(optimizer=K.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", K.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")])
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
    model.save_weights(save_dir / "weights.h5")

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
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
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
                    help="Optional tf.data cache path (disk).")
    ap.add_argument("--num_threads", type=int, default=0,
                    help="Optional thread hint for CPU ops.")
    ap.add_argument("--no_tfjs", action="store_true")
    ap.add_argument("--no_tflite", action="store_true")
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

    cache_tr = str(Path(args.cache_dir) /
                   "train.cache") if args.cache_dir else None
    cache_va = str(Path(args.cache_dir) /
                   "val.cache") if args.cache_dir else None
    cache_te = None  # usually no need to cache test

    ds_tr = make_dataset(train_items, args.seq_len, args.batch, shuffle=True,
                         seed=args.seed, augment=not args.no_aug, cache=cache_tr, deterministic=False)
    ds_va = make_dataset(val_items, args.seq_len, args.batch, shuffle=False,
                         seed=args.seed, augment=False, cache=cache_va)
    ds_te = make_dataset(test_items, args.seq_len, args.batch, shuffle=False,
                         seed=args.seed, augment=False, cache=cache_te)

    if args.model == "lstm":
        model = build_lstm_model(n_classes, seq_len=args.seq_len, feat_dim=FEAT_DIM,
                                 lr=args.lr, use_attention=args.use_attention)
    else:
        model = build_tcn_model(
            n_classes, seq_len=args.seq_len, feat_dim=FEAT_DIM, lr=args.lr)
    model.summary()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "labels.json"), "w") as f:
        json.dump({"classes": classes, "label2idx": {
                  c: i for i, c in enumerate(classes)}}, f, indent=2)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.save_dir, "best.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-5),
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

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred,
          target_names=classes, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(args.save_dir, "confusion_matrix.csv"),
               cm, fmt="%d", delimiter=",")

    # Save & export
    export_all(model,
               save_dir=args.save_dir,
               classes=classes,
               export_tfjs=not args.no_tfjs,
               export_tflite=not args.no_tflite)

    print(f"[OK] Artifacts saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
