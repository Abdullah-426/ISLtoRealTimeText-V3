# export_best.py
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# ---- Import your custom attention layer if used ----
from Useless.train_model_v2 import TemporalAttentionLayer

SAVE_DIR = Path("models/isl_phrases_v2")
BEST = SAVE_DIR / "best.keras"
LABELS = SAVE_DIR / "labels.json"

# Load classes
with open(LABELS, "r", encoding="utf-8") as f:
    classes = json.load(f)["classes"]

# Load best model (no compile)
model = tf.keras.models.load_model(
    BEST,
    compile=False,
    custom_objects={"TemporalAttentionLayer": TemporalAttentionLayer}
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 1) Keras formats
model.save(SAVE_DIR / "final_model.keras")
with open(SAVE_DIR / "model_arch.json", "w", encoding="utf-8") as f:
    f.write(model.to_json())
model.save_weights(SAVE_DIR / "model.weights.h5")

# 2) SavedModel (for TFJS converter and TFLite fallback)
tf.saved_model.save(model, str(SAVE_DIR / "saved_model"))

# 3) TFJS export (works with numpy<2.0)
try:
    import tensorflowjs as tfjs
    tfjs_dir = SAVE_DIR / "tfjs_model"
    tfjs_dir.mkdir(exist_ok=True)
    tfjs.converters.save_keras_model(model, str(tfjs_dir))
    print("[OK] TFJS model ->", tfjs_dir)
except Exception as e:
    print("[WARN] TFJS export failed:", e)

# 4) TFLite exports
# Try dynamic-int8 and float16 first; if LSTM TensorList blocks it, fall back to Select TF Ops


def try_tflite_builtins_dynamic_int8():
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl = conv.convert()
    (SAVE_DIR / "model_int8.tflite").write_bytes(tfl)
    print("[OK] TFLite dynamic-int8 -> model_int8.tflite")


def try_tflite_builtins_float16():
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    (SAVE_DIR / "model_float16.tflite").write_bytes(tfl)
    print("[OK] TFLite float16 -> model_float16.tflite")


def try_tflite_select_tf_ops():
    conv = tf.lite.TFLiteConverter.from_saved_model(
        str(SAVE_DIR / "saved_model"))
    conv.experimental_enable_resource_variables = True
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    # Guard for private flag
    try:
        conv._experimental_lower_tensor_list_ops = False
    except Exception:
        pass

    # plain float32
    tfl = conv.convert()
    (SAVE_DIR / "model_select_tfops.tflite").write_bytes(tfl)
    print("[OK] TFLite (Select TF Ops) -> model_select_tfops.tflite")

    # float16 with Select TF Ops
    conv.target_spec.supported_types = [tf.float16]
    tfl_f16 = conv.convert()
    (SAVE_DIR / "model_select_tfops_float16.tflite").write_bytes(tfl_f16)
    print("[OK] TFLite (Select TF Ops + float16) -> model_select_tfops_float16.tflite")


# Run the attempts
try:
    try_tflite_builtins_dynamic_int8()
except Exception as e:
    print("[WARN] Builtins INT8 failed; will try others:", e)

try:
    try_tflite_builtins_float16()
except Exception as e:
    print("[WARN] Builtins float16 failed; will try Select TF Ops:", e)

try:
    try_tflite_select_tf_ops()
except Exception as e:
    print("[WARN] Select TF Ops export failed:", e)

print("[OK] All export attempts done. Check:", SAVE_DIR)
