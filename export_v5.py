import tensorflowjs as tfjs
import json
import tensorflow as tf
from pathlib import Path
from train_model_v5 import (
    list_classes, build_lstm_model, FEAT_DIM, DEFAULT_SEQ_LEN)

SAVE_DIR = Path("models/isl_v5_lstm_mild_aw_deltas")
classes = json.load(open(SAVE_DIR/"labels.json"))["classes"]
feat_dim_in = FEAT_DIM * 2  # because you trained with --add_deltas

model = build_lstm_model(len(classes), DEFAULT_SEQ_LEN, feat_dim_in, lr=1e-5,
                         use_attention=True, dropout=0.45, label_smoothing=0.05,
                         lstm_w1=224, lstm_w2=128, l2_reg=1e-4,
                         optimizer_name="adamw", weight_decay=1e-4, soft_labels=False)
model.load_weights(SAVE_DIR/"best.weights.h5")

# Keras & SavedModel
model.save(SAVE_DIR/"final_model.keras", include_optimizer=False)
tf.saved_model.save(model, str(SAVE_DIR/"saved_model"))

# TFJS
tfjs_path = SAVE_DIR/"tfjs_model"
tfjs_path.mkdir(exist_ok=True)
tfjs.converters.save_keras_model(model, str(tfjs_path))
print("[OK] TFJS exported ->", tfjs_path)

# TFLite int8 (dynamic range) + float16
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
(SAVE_DIR/"model_int8.tflite").write_bytes(conv.convert())
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.target_spec.supported_types = [tf.float16]
(SAVE_DIR/"model_float16.tflite").write_bytes(conv.convert())
print("[OK] TFLite exported")
