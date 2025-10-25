#!/usr/bin/env python3
import os
import json
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# -------------------- Constants --------------------
NUM_LM_PER_HAND = 21
DIMS_PER_LM = 3
FEAT_DIM = NUM_LM_PER_HAND * DIMS_PER_LM * 2  # 126
EPS = 1e-6

# -------------------- Utils -----------------------


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_classes(root: Path):
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def load_split(split_root: Path, split: str, classes):
    """
    Load (X, y, paths) for a given split from MP_Dataset_Split/{split}/{class}/*.npy
    """
    X, y, paths = [], [], []
    for i, c in enumerate(classes):
        cdir = split_root / split / c
        files = sorted(glob(str(cdir / "*.npy")))
        for p in files:
            try:
                v = np.load(p)
            except Exception:
                continue
            v = v.reshape(-1).astype("float32")
            if v.shape[0] != FEAT_DIM:
                continue
            X.append(v)
            y.append(i)
            paths.append(p)
    if not X:
        return np.empty((0, FEAT_DIM), dtype="float32"), np.empty((0,), dtype="int32"), []
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="int32"), paths


def prepare_data(split_root: str, seed: int = 42, val_from_train: float = 0.1):
    """
    If val/ exists in MP_Dataset_Split -> use it.
    Else take val_from_train portion out of train.
    """
    root = Path(split_root)
    tr = root / "train"
    ts = root / "test"

    if not tr.is_dir() or not ts.is_dir():
        raise SystemExit(
            f"[ERROR] '{split_root}' must contain /train and /test")

    classes = list_classes(tr)
    if not classes:
        raise SystemExit(f"[ERROR] No class folders found under {tr}")

    # Load train; if val/ exists use it, else split from train
    Xtr, ytr, _ = load_split(root, "train", classes)

    val_dir = root / "val"
    if val_dir.is_dir():
        Xva, yva, _ = load_split(root, "val", classes)
    else:
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr, ytr, test_size=val_from_train, random_state=seed, stratify=ytr
        )

    Xte, yte, _ = load_split(root, "test", classes)
    return classes, (Xtr, ytr), (Xva, yva), (Xte, yte)

# --------------- Augmentation ---------------------


def _np_aug_126(x_np, xy_scale=0.02, xy_shift=0.01, z_noise=0.005):
    """
    x_np: (126,) -> numpy array. Apply per-hand scale/shift (x,y) and z noise.
    Keeps 0-hand blocks intact (important for 'blank').
    (LIGHT augmentation)
    """
    x = np.array(x_np, dtype=np.float32).reshape(
        2, NUM_LM_PER_HAND, DIMS_PER_LM)
    for h in range(2):
        hand = x[h]
        if np.allclose(hand, 0.0):  # missing hand -> leave zeros (esp. blank)
            continue
        s = 1.0 + np.random.uniform(-xy_scale, xy_scale)
        tx = np.random.uniform(-xy_shift, xy_shift)
        ty = np.random.uniform(-xy_shift, xy_shift)
        hand[:, 0] = np.clip(hand[:, 0] * s + tx, 0.0, 1.0)
        hand[:, 1] = np.clip(hand[:, 1] * s + ty, 0.0, 1.0)
        hand[:, 2] = hand[:, 2] + \
            np.random.normal(0.0, z_noise, size=hand.shape[0])
    return x.reshape(-1).astype(np.float32)


def build_augment_map_fn(xy_scale=0.02, xy_shift=0.01, z_noise=0.005):
    def tf_aug(x, y):
        x = tf.py_function(
            func=lambda a: _np_aug_126(a, xy_scale, xy_shift, z_noise),
            inp=[x],
            Tout=tf.float32,
        )
        x.set_shape([FEAT_DIM])
        return x, y
    return tf_aug

# --------------- Model (Keras Functional) ----------


def build_model(num_classes: int, lr: float = 1e-3):
    """
    Two-branch MLP with wrist-centered scale normalization (WCS) from RAW input.
      - Input raw (126) -> reshape to (2,21,3)
      - Presence flags per hand (B,2) computed from RAW (no normalization layer)
      - WCS on RAW -> per hand flatten (63) -> shared encoder -> concat with flags
      - Classifier head
    """
    K = tf.keras
    inp = K.Input(shape=(FEAT_DIM,), name="keypoints_raw")  # RAW input

    # reshape to (B, 2, 21, 3) from RAW
    hands = K.layers.Reshape(
        (2, NUM_LM_PER_HAND, DIMS_PER_LM), name="reshape_raw")(inp)

    # presence flags from RAW: (B,2)
    def pres_fn(t):
        s = tf.reduce_sum(tf.abs(t), axis=[2, 3])  # (B,2)
        return tf.cast(s > 0.0, tf.float32)

    flags = K.layers.Lambda(pres_fn, name="presence_flags")(hands)  # (B,2)

    # wrist-centered + scale-normalized on RAW
    def wcs_fn(t):
        wrist = t[:, :, 0:1, :]                        # (B,2,1,3)
        centered = t - wrist                           # (B,2,21,3)
        dist = tf.norm(centered, axis=-1)              # (B,2,21)
        span = tf.reduce_max(dist, axis=-1, keepdims=True)  # (B,2,1)
        span = tf.maximum(span, EPS)
        centered = centered / span[..., None]          # (B,2,21,3)
        return centered

    centered = K.layers.Lambda(wcs_fn, name="wrist_center_scale")(hands)

    # Flatten per hand
    L = K.layers.Lambda(lambda z: z[:, 0, :, :], name="left_hand")(centered)
    R = K.layers.Lambda(lambda z: z[:, 1, :, :], name="right_hand")(centered)
    L = K.layers.Flatten()(L)  # (B,63)
    R = K.layers.Flatten()(R)  # (B,63)

    # Shared hand encoder
    def hand_encoder():
        ii = K.Input(shape=(63,))
        x = K.layers.Dense(256, activation=None)(ii)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation("relu")(x)
        x = K.layers.Dropout(0.25)(x)

        x = K.layers.Dense(128, activation=None)(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation("relu")(x)
        x = K.layers.Dropout(0.25)(x)
        return K.Model(ii, x, name="hand_encoder")

    enc = hand_encoder()
    Lh = enc(L)
    Rh = enc(R)

    h = K.layers.Concatenate(name="concat_with_flags")(
        [Lh, Rh, flags])  # (B,258)

    # Classifier head
    h = K.layers.Dense(256, activation=None)(h)
    h = K.layers.BatchNormalization()(h)
    h = K.layers.Activation("relu")(h)
    h = K.layers.Dropout(0.35)(h)

    h = K.layers.Dense(128, activation=None)(h)
    h = K.layers.BatchNormalization()(h)
    h = K.layers.Activation("relu")(h)

    out = K.layers.Dense(num_classes, activation="softmax")(h)

    model = K.Model(inputs=inp, outputs=out, name="ISL_MLP_WCS_RAW")
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            K.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )
    return model

# --------------- Dataset builders ------------------


def make_datasets(Xtr, ytr, Xva, yva, augment=True, batch=128, seed=42,
                  aug_xy_scale=0.02, aug_xy_shift=0.01, aug_z_noise=0.005):
    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, ytr))
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, yva))

    if augment:
        ds_tr = ds_tr.shuffle(len(Xtr), seed=seed)
        ds_tr = ds_tr.map(build_augment_map_fn(
            xy_scale=aug_xy_scale, xy_shift=aug_xy_shift, z_noise=aug_z_noise),
            num_parallel_calls=tf.data.AUTOTUNE)

    ds_tr = ds_tr.batch(batch).prefetch(tf.data.AUTOTUNE)
    ds_va = ds_va.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds_tr, ds_va


def compute_cw(y, n_classes):
    classes = np.arange(n_classes)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {i: float(w[i]) for i in range(n_classes)}

# --------------- Main --------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Train ISL keypoints MLP (WCS on RAW) on MP_Dataset_Split."
    )
    ap.add_argument("--split_root", type=str, default="MP_Dataset_Split")
    ap.add_argument("--val_from_train", type=float, default=0.10,
                    help="If no val folder, split this much from train")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_aug", action="store_true",
                    help="Disable augmentation")
    ap.add_argument("--aug_xy_scale", type=float, default=0.02,
                    help="Augment: +/- scale on x,y (default 0.02)")
    ap.add_argument("--aug_xy_shift", type=float, default=0.01,
                    help="Augment: +/- shift on x,y (default 0.01)")
    ap.add_argument("--aug_z_noise", type=float, default=0.005,
                    help="Augment: z noise std (default 0.005)")
    ap.add_argument("--save_dir", type=str,
                    default="models/isl_keypoints_wcs_raw")
    args = ap.parse_args()

    set_seeds(args.seed)

    # Load data from MP_Dataset_Split (train/test; optional val/)
    classes, (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare_data(
        args.split_root, seed=args.seed, val_from_train=args.val_from_train
    )
    print(
        f"[INFO] #classes={len(classes)} | train={Xtr.shape} | val={Xva.shape} | test={Xte.shape}")

    ds_tr, ds_va = make_datasets(
        Xtr, ytr, Xva, yva,
        augment=not args.no_aug,
        batch=args.batch,
        seed=args.seed,
        aug_xy_scale=args.aug_xy_scale,
        aug_xy_shift=args.aug_xy_shift,
        aug_z_noise=args.aug_z_noise,
    )
    model = build_model(len(classes), lr=args.lr)
    model.summary()

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, "best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(os.path.join(
            args.save_dir, "training_log.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    cw = compute_cw(ytr, len(classes))
    print("[INFO] Class weights:", cw)

    model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=args.epochs,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # --------- Evaluation & save ----------
    y_prob = model.predict(Xte, batch_size=args.batch, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    top3 = np.any(np.argsort(-y_prob, axis=1)
                  [:, :3] == yte[:, None], axis=1).mean()
    acc = (y_pred == yte).mean()
    print(f"\n[TEST] accuracy={acc:.4f}  top3={top3:.4f}")

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(yte, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(yte, y_pred)
    np.savetxt(os.path.join(args.save_dir, "confusion_matrix.csv"),
               cm, fmt="%d", delimiter=",")
    with open(os.path.join(args.save_dir, "labels.json"), "w") as f:
        json.dump({"classes": classes, "label2idx": {
                  c: i for i, c in enumerate(classes)}}, f, indent=2)

    # Save both legacy H5 and modern Keras format
    model.save(os.path.join(args.save_dir, "model.h5"))
    model.save(os.path.join(args.save_dir, "model.keras"))
    with open(os.path.join(args.save_dir, "model.json"), "w") as f:
        f.write(model.to_json())

    print(f"[INFO] Saved artifacts to: {args.save_dir}")
    print(f"[DONE] Best checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
