# import os
# import tensorflow as tf
# print("TF:", tf.__version__)
# print("GPUs:", tf.config.list_physical_devices("GPU"))

# # Optional: prefer dynamic growth to avoid VRAM pre-allocation
# print("TF_FORCE_GPU_ALLOW_GROWTH =", os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"))

# import tensorflow as tf
# import numpy as np
# import cv2
# import ml_dtypes
# print("TF:", tf.__version__, " | numpy:", np.__version__, " | opencv:",
#       cv2.__version__, " | ml_dtypes:", ml_dtypes.__version__)
# try:
#     import mediapipe as mp
#     print("mediapipe:", mp.__version__)
#     # Try Hands init
#     with mp.solutions.hands.Hands(model_complexity=0, max_num_hands=2) as h:
#         print("MediaPipe Hands OK")
# except Exception as e:
#     print("Mediapipe import error:", e)

# quick_check.py
# quick_check.py
# import numpy as np
# import tensorflow as tf
# from pathlib import Path
# from custom_layers import TemporalAttentionLayer


# def try_load(dir_):
#     print("Trying:", dir_)
#     m = tf.keras.models.load_model(
#         str(Path(dir_)/"best.keras"),
#         compile=False,
#         custom_objects={"TemporalAttentionLayer": TemporalAttentionLayer},
#         safe_mode=False,
#     )
#     print("Model input_shape:", m.input_shape)
#     # Expect something like: (None, 48, 1662)
#     _, T, D = m.input_shape
#     x = np.zeros((1, T, D), dtype="float32")
#     y = m.predict(x, verbose=0)
#     print("OK -> out shape:", y.shape)


# try_load("models/isl_phrases_v3_lstm")
# try_load("models/isl_phrases_v3_tcn")


# test.py  ─── Minimal WebRTC smoke test (no STUN, Windows loop policy)

import platform
import asyncio
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.title("Minimal WebRTC camera")
cfg = RTCConfiguration({"iceServers": []})  # NO STUN (localhost)

ctx = webrtc_streamer(
    key="min",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=cfg,
    media_stream_constraints={
        "video": {"width": {"ideal": 960}, "height": {"ideal": 540}, "frameRate": {"ideal": 30}},
        "audio": False,
    },
    video_html_attrs={"autoPlay": True, "muted": True,
                      "playsInline": True, "controls": True},
)

st.write("State:", getattr(ctx.state, "value", ctx.state))
st.info("If this shows your camera, the environment is OK. Then use the main app below.")
