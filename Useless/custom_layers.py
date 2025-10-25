# custom_layers.py
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="custom", name="TemporalAttentionLayer")
class TemporalAttentionLayer(keras.layers.Layer):
    """
    Mirrors saved weights:
      layers/temporal_attention_layer/proj/vars/{0,1}
      layers/temporal_attention_layer/score/vars/{0,1}
    """

    def __init__(self, units, return_attention=False, **kwargs):
        kwargs = dict(kwargs)
        # match layer scope
        kwargs.setdefault("name", "temporal_attention_layer")
        super().__init__(**kwargs)
        self.units = int(units)
        self.return_attention = return_attention
        self.proj = []
        self.score = []

    def build(self, input_shape):
        F = int(input_shape[-1])
        with tf.name_scope("proj"):
            W = self.add_weight(name="vars", shape=(F, self.units),
                                initializer="glorot_uniform", trainable=True)
            b = self.add_weight(name="vars", shape=(self.units,),
                                initializer="zeros", trainable=True)
            self.proj = [W, b]
        with tf.name_scope("score"):
            u = self.add_weight(name="vars", shape=(self.units,),
                                initializer="glorot_uniform", trainable=True)
            c = self.add_weight(name="vars", shape=(1,),
                                initializer="zeros", trainable=True)
            self.score = [u, c]
        super().build(input_shape)

    def call(self, x):
        W, b = self.proj
        u, c = self.score
        uit = tf.tanh(tf.tensordot(x, W, axes=1) + b)   # (B,T,U)
        ait = tf.tensordot(uit, u, axes=1) + c          # (B,T)
        alpha = tf.nn.softmax(ait, axis=1)              # (B,T)
        context = tf.reduce_sum(x * tf.expand_dims(alpha, -1), axis=1)  # (B,F)
        return (context, alpha) if self.return_attention else context

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {"units": self.units, "return_attention": self.return_attention})
        return cfg
