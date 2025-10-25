# app_opencv.py
# ==============  ISL Real-time Translator (OpenCV - WORKING)  ==================
# - Three models: MLP (letters), LSTM + TCN (phrases) + ensemble
# - Direct OpenCV camera access - GUARANTEED TO WORK
# - No WebRTC complications
# =====================================================================

import os
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ---------- Custom LSTM attention layer ----------
from custom_layers import TemporalAttentionLayer

# ---------- Paths ----------
PHRASE_LSTM_DIR = "models/isl_phrases_v3_lstm"
PHRASE_TCN_DIR = "models/isl_phrases_v3_tcn"
MLP_DIR = "models/isl_wcs_raw_aug_light_v2"

# ---------- Constants ----------
DEFAULT_SEQ_LEN = 48
DEFAULT_FEAT_DIM = 1662
NUM_LM_HAND = 21
DIMS_PER_LM = 3
MLP_FEAT_DIM = NUM_LM_HAND * DIMS_PER_LM * 2

# ---------- Custom objects for MLP ----------


@tf.keras.utils.register_keras_serializable(package="Custom", name="wcs_fn")
def wcs_fn(t):
    EPS = 1e-6
    wrist = t[:, :, 0:1, :]
    centered = t - wrist
    dist = tf.norm(centered, axis=-1)
    span = tf.reduce_max(dist, axis=-1, keepdims=True)
    span = tf.maximum(span, EPS)
    centered = centered / span[..., None]
    present = tf.reduce_sum(tf.abs(t), axis=[2, 3])
    present = tf.cast(tf.not_equal(present, 0.0), tf.float32)
    return centered, present


@tf.keras.utils.register_keras_serializable(package="Custom", name="pres_fn")
def pres_fn(t):
    present = tf.reduce_sum(tf.abs(t), axis=[2, 3])
    return tf.cast(tf.not_equal(present, 0.0), tf.float32)


@tf.keras.utils.register_keras_serializable(package="Custom", name="lhand_fn")
def lhand_fn(z): return z[:, 0, :, :]


@tf.keras.utils.register_keras_serializable(package="Custom", name="rhand_fn")
def rhand_fn(z): return z[:, 1, :, :]

# ---------- Label loader ----------


def load_labels(path):
    obj = json.load(open(path, "r", encoding="utf-8"))
    if "classes" in obj:
        return obj["classes"]
    if isinstance(obj, dict) and "label2idx" in obj:
        return [c for c, _ in sorted(obj["label2idx"].items(), key=lambda kv: kv[1])]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Bad labels file: {path}")

# ---------- Model loaders ----------


def load_phrase_model(dir_):
    custom = {"TemporalAttentionLayer": TemporalAttentionLayer}
    model = tf.keras.models.load_model(
        str(Path(dir_) / "best.keras"),
        compile=False, custom_objects=custom, safe_mode=False
    )
    labels = load_labels(str(Path(dir_) / "labels.json"))
    return model, labels


def load_mlp_model(dir_):
    custom = {"wcs_fn": wcs_fn, "pres_fn": pres_fn,
              "lhand_fn": lhand_fn, "rhand_fn": rhand_fn}
    model = tf.keras.models.load_model(
        str(Path(dir_) / "best.keras"),
        compile=False, custom_objects=custom, safe_mode=False
    )
    labels = load_labels(str(Path(dir_) / "labels.json"))
    return model, labels

# ---------- Cache models across reruns ----------


@st.cache_resource
def get_models_and_meta():
    mlp, labels_mlp = load_mlp_model(MLP_DIR)
    lstm, labels_ph = load_phrase_model(PHRASE_LSTM_DIR)
    tcn, labels_ph2 = load_phrase_model(PHRASE_TCN_DIR)
    assert labels_ph == labels_ph2, "Phrase labels mismatch!"

    _, T_lstm, D_lstm = lstm.input_shape
    _, T_tcn, D_tcn = tcn.input_shape
    return {
        "mlp": mlp, "labels_mlp": labels_mlp,
        "lstm": lstm, "tcn": tcn, "labels_ph": labels_ph,
        "T_lstm": int(T_lstm), "D_lstm": int(D_lstm),
        "T_tcn": int(T_tcn), "D_tcn": int(D_tcn),
        "T_phrase_max": int(max(T_lstm, T_tcn)),
    }


# ---------- MediaPipe ----------
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    HAS_MP = True
except Exception:
    HAS_MP = False

# ---------- Feature extractors ----------


def extract_2hand_126(frame_bgr, hands_ctx):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands_ctx.process(rgb)
    rgb.flags.writeable = True

    left = np.zeros((21, 3), np.float32)
    right = np.zeros((21, 3), np.float32)
    bbox = (0, 0, w, h)

    if getattr(res, "multi_hand_landmarks", None):
        hand_map = {}
        if getattr(res, "multi_handedness", None):
            for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label
                hand_map[label] = lm
        else:
            if len(res.multi_hand_landmarks) >= 1:
                hand_map["Right"] = res.multi_hand_landmarks[0]
            if len(res.multi_hand_landmarks) >= 2:
                hand_map["Left"] = res.multi_hand_landmarks[1]

        if "Left" in hand_map:
            for i, p in enumerate(hand_map["Left"].landmark):
                left[i, :] = [p.x, p.y, p.z]
        if "Right" in hand_map:
            for i, p in enumerate(hand_map["Right"].landmark):
                right[i, :] = [p.x, p.y, p.z]

        xs, ys = [], []
        for lm in hand_map.values():
            for p in lm.landmark:
                xs.append(int(p.x * w))
                ys.append(int(p.y * h))
        if xs and ys:
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w, max(xs)), min(h, max(ys))
            m = int(0.10 * max(1, max(x2 - x1, y2 - y1)))
            bbox = (max(0, x1 - m), max(0, y1 - m),
                    min(w, x2 + m), min(h, y2 + m))

    vec = np.concatenate([left.reshape(-1), right.reshape(-1)], axis=0)
    return vec.astype(np.float32), bbox, res


def extract_1662(frame_bgr, holistic_ctx):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = holistic_ctx.process(rgb)
    rgb.flags.writeable = True

    pose = np.zeros((33, 4), np.float32)
    face = np.zeros((468, 3), np.float32)
    lh = np.zeros((21, 3), np.float32)
    rh = np.zeros((21, 3), np.float32)
    xs, ys = [], []

    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark):
            pose[i] = [lm.x, lm.y, lm.z, lm.visibility]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

    if res.face_landmarks:
        for i, lm in enumerate(res.face_landmarks.landmark[:468]):
            face[i] = [lm.x, lm.y, lm.z]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

    if res.left_hand_landmarks:
        for i, lm in enumerate(res.left_hand_landmarks.landmark):
            lh[i] = [lm.x, lm.y, lm.z]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

    if res.right_hand_landmarks:
        for i, lm in enumerate(res.right_hand_landmarks.landmark):
            rh[i] = [lm.x, lm.y, lm.z]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

    if xs and ys:
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))
        m = int(0.06 * max(1, max(x2 - x1, y2 - y1)))
        bbox = (max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m))
    else:
        bbox = (0, 0, w, h)

    vec = np.concatenate(
        [pose.reshape(-1), face.reshape(-1), lh.reshape(-1), rh.reshape(-1)], axis=0)
    return vec.astype(np.float32), bbox, res

# ---------- Helpers ----------


def adapt_sequence_dim(seq_td: np.ndarray, target_dim: int) -> np.ndarray:
    T, d0 = seq_td.shape
    out = np.zeros((T, target_dim), dtype=seq_td.dtype)
    m = min(d0, target_dim)
    out[:, :m] = seq_td[:, :m]
    return out


class ProbSmoother:
    def __init__(self, n_classes, alpha=0.8):
        self.p = np.zeros(n_classes, np.float32)
        self.alpha = float(alpha)
        self.count = 0
        self.ready = False

    def reset(self):
        self.p[:] = 0.0
        self.count = 0
        self.ready = False

    def update(self, probs):
        self.p = self.alpha * self.p + (1.0 - self.alpha) * probs
        self.count += 1
        self.ready = self.count >= 3
        return self.p.copy(), self.ready


class CommitState:
    def __init__(self, threshold=0.55, hold_ms=3000, cooldown_ms=800):
        self.th = float(threshold)
        self.hold = int(hold_ms)
        self.cool = int(cooldown_ms)
        self.cand = None
        self.since = None
        self.last_commit = 0

    def step(self, t_ms, top_idx, top_conf):
        commit = None
        if top_conf >= self.th:
            if self.cand == top_idx:
                if self.since is None:
                    self.since = t_ms
                if (t_ms - self.since) >= self.hold and (t_ms - self.last_commit) >= self.cool:
                    commit = top_idx
                    self.last_commit = t_ms
                    self.cand = None
                    self.since = None
            else:
                self.cand = top_idx
                self.since = t_ms
        else:
            self.cand = None
            self.since = None
        frac = 0.0 if self.since is None else (t_ms - self.since) / self.hold
        return commit, self.cand, frac


def softmax_T(probs, T=1.0):
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 1:
        probs = probs[None, :]
    p = np.clip(probs, 1e-8, 1.0)
    q = np.exp(np.log(p) / float(T))
    q = q / q.sum(axis=1, keepdims=True)
    return q


def top_k(probs, labels, k=3):
    idx = np.argsort(-probs)[:k]
    return [(labels[i], float(probs[i])) for i in idx], int(idx[0]), float(probs[idx[0]])

# ===================== Streamlit UI =====================


st.set_page_config(page_title="ISL Real-time", layout="wide")
st.title("🤟 ISL Real-time Translator (OpenCV)")
st.markdown("**Models:** MLP (Letters) + LSTM/TCN/Ensemble (Phrases)")

if not HAS_MP:
    st.error("⚠️ mediapipe is not installed. Run: `pip install mediapipe==0.10.14`")
    st.stop()

# Initialize session state
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "typed_text" not in st.session_state:
    st.session_state.typed_text = ""
if "top3_preds" not in st.session_state:
    st.session_state.top3_preds = []

# Load models
with st.spinner("Loading models..."):
    meta = get_models_and_meta()

# Sidebar
st.sidebar.header("⚙️ Controls")
mode = st.sidebar.selectbox(
    "Recognition Mode",
    ["Letters (MLP)", "Phrases – LSTM", "Phrases – TCN", "Phrases – Ensemble"],
    key="mode_select"
)

camera_index = st.sidebar.number_input("Camera Index", 0, 10, 0, key="cam_idx")
auto_commit = st.sidebar.checkbox(
    "Auto commit (hold-to-commit)", value=True, key="auto_commit")

with st.sidebar.expander("🎯 Commit Settings"):
    conf_th = st.slider("Confidence threshold", 0.4, 0.9, 0.55, 0.01, key="th")
    hold_s = st.slider("Hold duration (s)", 0.5, 4.0, 3.0, 0.1, key="hold")
    cool_s = st.slider("Cooldown (s)", 0.2, 2.0, 0.8, 0.1, key="cool")

with st.sidebar.expander("🔧 Advanced"):
    ema_alpha = st.slider("Smoothing α", 0.6, 0.95, 0.80, 0.01, key="ema")

if mode == "Phrases – Ensemble":
    with st.sidebar.expander("🎛️ Ensemble Settings"):
        alpha_mix = st.slider("LSTM weight α", 0.0, 1.0, 0.5, 0.05, key="mix")
        T_lstm_temp = st.slider("LSTM temperature", 0.5,
                                2.0, 1.0, 0.05, key="Tl")
        T_tcn_temp = st.slider("TCN temperature", 0.5,
                               2.0, 1.0, 0.05, key="Tt")
else:
    alpha_mix, T_lstm_temp, T_tcn_temp = 0.5, 1.0, 1.0

# Layout
col_video, col_side = st.columns([2, 1])

with col_video:
    st.markdown("### 📹 Camera Feed")
    video_placeholder = st.empty()

    btn_col1, btn_col2 = st.columns(2)
    start_btn = btn_col1.button(
        "▶️ START CAMERA", use_container_width=True, type="primary")
    stop_btn = btn_col2.button("⏹️ STOP CAMERA", use_container_width=True)

with col_side:
    st.markdown("### 📝 Live Output")
    typed_placeholder = st.empty()
    topk_placeholder = st.empty()

    st.markdown("### 🎮 Controls")
    ctrl_col1, ctrl_col2 = st.columns(2)
    ctrl_col3, ctrl_col4 = st.columns(2)

    save_status = st.empty()

# Button handlers
if start_btn:
    st.session_state.camera_running = True

if stop_btn:
    st.session_state.camera_running = False

# Control buttons
if ctrl_col1.button("➕ Space", key="btn_space", use_container_width=True):
    st.session_state.typed_text += " "

if ctrl_col2.button("⬅️ Backspace", key="btn_bksp", use_container_width=True):
    st.session_state.typed_text = st.session_state.typed_text[:-1]

if ctrl_col3.button("🗑️ Clear", key="btn_clear", use_container_width=True):
    st.session_state.typed_text = ""

if ctrl_col4.button("💾 Save", key="btn_save", use_container_width=True):
    os.makedirs("transcripts", exist_ok=True)
    path = f"transcripts/isl_{int(time.time())}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(st.session_state.typed_text)
    save_status.success(f"✅ Saved to {path}")

# Main camera loop
if st.session_state.camera_running:
    cap = cv2.VideoCapture(int(camera_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        st.error("❌ Cannot open camera! Try a different camera index.")
        st.session_state.camera_running = False
    else:
        # Initialize MediaPipe
        if mode.startswith("Letters"):
            mp_ctx = mp_hands.Hands(
                static_image_mode=False,
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            smoother = ProbSmoother(len(meta["labels_mlp"]), ema_alpha)
            committer = CommitState(conf_th, int(
                hold_s*1000), int(cool_s*1000))
            labels = meta["labels_mlp"]
        else:
            mp_ctx = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            smoother = ProbSmoother(len(meta["labels_ph"]), ema_alpha)
            committer = CommitState(conf_th, int(
                hold_s*1000), int(cool_s*1000))
            labels = meta["labels_ph"]
            seqbuf = deque(maxlen=meta["T_phrase_max"])

        frame_count = 0

        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from camera!")
                break

            frame = cv2.flip(frame, 1)
            now_ms = int(time.time() * 1000)

            try:
                if mode.startswith("Letters"):
                    # Letters mode
                    vec126, bbox, res = extract_2hand_126(frame, mp_ctx)
                    probs = meta["mlp"].predict(
                        vec126.reshape(1, -1), verbose=0)[0]
                    smp, ready = smoother.update(probs)
                    use = smp if ready else probs
                    top3, tidx, tconf = top_k(use, labels, k=3)

                    # Draw landmarks
                    if res and getattr(res, "multi_hand_landmarks", None):
                        for lm in res.multi_hand_landmarks:
                            mp_draw.draw_landmarks(
                                frame, lm, mp_hands.HAND_CONNECTIONS,
                                mp_styles.get_default_hand_landmarks_style(),
                                mp_styles.get_default_hand_connections_style()
                            )

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Auto commit
                    if auto_commit:
                        commit, _, _ = committer.step(now_ms, tidx, tconf)
                        if commit is not None:
                            lbl = labels[commit]
                            st.session_state.typed_text += (
                                " " if lbl.lower() == "blank" else lbl)
                            smoother.reset()

                    st.session_state.top3_preds = top3

                else:
                    # Phrases mode
                    vec1662, bbox, res = extract_1662(frame, mp_ctx)
                    seqbuf.append(vec1662)

                    if mode == "Phrases – LSTM":
                        need_T, need_D = meta["T_lstm"], meta["D_lstm"]
                    elif mode == "Phrases – TCN":
                        need_T, need_D = meta["T_tcn"], meta["D_tcn"]
                    else:
                        need_T = max(meta["T_lstm"], meta["T_tcn"])

                    if len(seqbuf) >= need_T:
                        seq_arr = np.stack(list(seqbuf)[-need_T:], axis=0)

                        if mode == "Phrases – LSTM":
                            p = meta["lstm"].predict(adapt_sequence_dim(
                                seq_arr, meta["D_lstm"])[None, ...], verbose=0)
                            p = softmax_T(p, 1.0)[0]
                        elif mode == "Phrases – TCN":
                            p = meta["tcn"].predict(adapt_sequence_dim(
                                seq_arr, meta["D_tcn"])[None, ...], verbose=0)
                            p = softmax_T(p, 1.0)[0]
                        else:
                            x_l = adapt_sequence_dim(
                                seq_arr[-meta["T_lstm"]:], meta["D_lstm"])[None, ...]
                            x_t = adapt_sequence_dim(
                                seq_arr[-meta["T_tcn"]:], meta["D_tcn"])[None, ...]
                            p1 = softmax_T(meta["lstm"].predict(
                                x_l, verbose=0), T_lstm_temp)
                            p2 = softmax_T(meta["tcn"].predict(
                                x_t, verbose=0), T_tcn_temp)
                            p = (alpha_mix * p1 + (1.0 - alpha_mix) * p2)[0]

                        smp, ready = smoother.update(p)
                        use = smp if ready else p
                        top3, tidx, tconf = top_k(use, labels, k=3)

                        # Auto commit
                        if auto_commit:
                            commit, _, _ = committer.step(now_ms, tidx, tconf)
                            if commit is not None:
                                lbl = labels[commit]
                                st.session_state.typed_text += (
                                    " " if lbl.lower() == "blank" else f" {lbl} ")
                                smoother.reset()

                        st.session_state.top3_preds = top3
                    else:
                        st.session_state.top3_preds = [("…buffering…", 0.0)]

                    # Draw landmarks
                    if res and res.face_landmarks:
                        mp_draw.draw_landmarks(
                            frame, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                        )
                    if res and res.pose_landmarks:
                        mp_draw.draw_landmarks(
                            frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    if res and res.left_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if res and res.right_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add typed text overlay
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (20, h-80),
                              (w-20, h-20), (30, 30, 30), -1)
                cv2.putText(frame, f"Typed: {st.session_state.typed_text[-50:]}", (30, h-35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            except Exception as e:
                cv2.putText(frame, f"ERR: {str(e)[:40]}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(
                frame_rgb, channels="RGB", use_container_width=True)

            # Update output (only every 10 frames to reduce overhead)
            frame_count += 1
            if frame_count % 10 == 0:
                typed_placeholder.markdown(
                    f"**Typed:** `{st.session_state.typed_text}`")
                with topk_placeholder.container():
                    st.markdown("**Top-3 Predictions:**")
                    for lbl, p in st.session_state.top3_preds:
                        st.write(f"- {lbl}: {p*100:.1f}%")

            # Small delay
            time.sleep(0.03)

        # Cleanup
        cap.release()
        mp_ctx.close()

else:
    video_placeholder.info("👆 Click 'START CAMERA' to begin")
    st.markdown("""
    ### 📋 Instructions:
    1. **Click START CAMERA** to activate your webcam
    2. **Select mode** in the sidebar (Letters or Phrases)
    3. **Perform signs** in front of the camera
    4. **Hold steady** for 3 seconds to commit a prediction
    5. Use **controls** to edit or save your text
    
    ### 🔧 Troubleshooting:
    - If camera doesn't open, try different **Camera Index** (0, 1, 2...)
    - Ensure good lighting for best results
    - Keep hands visible in frame
    - Click STOP before changing settings
    """)
