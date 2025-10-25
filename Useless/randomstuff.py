import tensorflowjs as tfjs
import keras
from custom_layers import TemporalAttentionLayer
import os
os.environ["KERAS_BACKEND"] = "tensorflow"


keras_model_path = r".\models\isl_phrases_v2\final_model.keras"
saved_model_dir = r".\models\isl_phrases_v2\saved_model"
tfjs_out_dir = r".\models\isl_phrases_v2\tfjs_model"

model = keras.models.load_model(
    keras_model_path,
    custom_objects={"TemporalAttentionLayer": TemporalAttentionLayer},
    compile=False,
)
print("Loaded Keras model ✅")

# Export TF SavedModel (robust for TFJS)
model.save(saved_model_dir, save_format="tf")
print("Exported SavedModel:", saved_model_dir)

# Convert to TFJS
tfjs.converters.convert_tf_saved_model(saved_model_dir, tfjs_out_dir)
print("✅ TFJS model written to:", tfjs_out_dir)
