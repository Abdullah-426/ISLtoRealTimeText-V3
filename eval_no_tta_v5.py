import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path  # <-- add this
from train_model_v5 import (list_classes, enumerate_clips, make_dataset,
                            build_lstm_model, build_tcn_model, build_transformer_model,
                            FEAT_DIM, DEFAULT_SEQ_LEN)

ap = argparse.ArgumentParser()
ap.add_argument("--split_root", default="Dataset_Split")
ap.add_argument("--model", choices=["lstm",
                "tcn", "transformer"], default="lstm")
ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--weights", required=True)
ap.add_argument("--add_deltas", action="store_true")
args = ap.parse_args()

split_root = Path(args.split_root)  # <-- add this

classes = list_classes(split_root)  # <-- use split_root (Path) here
feat_dim_in = FEAT_DIM * (2 if args.add_deltas else 1)

if args.model == "lstm":
    model = build_lstm_model(len(classes), args.seq_len, feat_dim_in, lr=1e-5,
                             use_attention=True, dropout=0.45, label_smoothing=0.05,
                             lstm_w1=224, lstm_w2=128, l2_reg=1e-4,
                             optimizer_name="adamw", weight_decay=1e-4, soft_labels=False)
elif args.model == "tcn":
    model = build_tcn_model(len(classes), args.seq_len, feat_dim_in, lr=1e-5,
                            dropout=0.45, label_smoothing=0.05, l2_reg=1e-4,
                            optimizer_name="adamw", weight_decay=1e-4, soft_labels=False)
else:
    model = build_transformer_model(len(classes), args.seq_len, feat_dim_in, lr=1e-5,
                                    dropout=0.35, label_smoothing=0.05, l2_reg=1e-4,
                                    layers=3, heads=4, d_model=256, ff_dim=512,
                                    optimizer_name="adamw", weight_decay=1e-4, soft_labels=False)

model.load_weights(args.weights)

test_items = enumerate_clips(split_root, "test", classes)  # <-- use Path
ds_te = make_dataset(test_items, args.seq_len, args.batch, shuffle=False, seed=42,
                     augment=False, cache=None, deterministic=True,
                     time_shift=0, face_scale=1.0, face_dropout=0.0,
                     xy_scale=0.0, xy_shift=0.0, z_noise=0.0,
                     temporal_cutout=0, shuffle_buf=1024,
                     hand_dropout=0.0, landmark_dropout=0.0,
                     enable_time_warp=False, enable_temporal_crop=False,
                     add_deltas=args.add_deltas)

y_true = np.concatenate([y.numpy() for _, y in ds_te])
probs = np.concatenate([model(x, training=False).numpy()
                       for x, _ in ds_te], axis=0)
y_pred = probs.argmax(axis=1)
top3 = np.any(np.argsort(-probs, axis=1)
              [:, :3] == y_true[:, None], axis=1).mean()
acc = (y_true == y_pred).mean()
print(f"[EVAL no-TTA] acc={acc:.4f}  top3={top3:.4f}")
