import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Custom")
class TemporalAttentionLayer(layers.Layer):
    """Temporal attention over time. Input (B,T,D) -> Output (B,D)."""

    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.W = layers.Dense(self.units, use_bias=True, name="attn_W")
        self.v = layers.Dense(1,     use_bias=False, name="attn_v")
        self._D = None

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected (B,T,D); got {input_shape}")
        self._D = int(input_shape[-1])
        super().build(input_shape)

    def call(self, x, training=None):
        scores = self.v(tf.nn.tanh(self.W(x)))  # (B,T,1)
        alpha = tf.nn.softmax(scores, axis=1)  # (B,T,1)
        context = tf.reduce_sum(alpha * x, axis=1)  # (B,D)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg
