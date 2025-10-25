#!/usr/bin/env python3
import os
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

# ---------------------- Defaults (Editable) ----------------------
RAW_ROOT = "RAW DATA"
T = 48
CLIPS_PER_CLASS = 50
CAM_INDEX = 0
FPS_TARGET = 20

# Holistic thresholds
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
MODEL_COMPLEXITY = 1

# UI
WINDOW = "ISL Words Collector (Holistic)"
COUNTDOWN_SEC = 3.0
BETWEEN_CLIPS_SEC = 1.2
PREVIEW_SCALE = 1.0
# -----------------------------------------------------------------

# ------------- MediaPipe Holistic -------------
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    USE_MP = True
except Exception:
    print("[ERROR] mediapipe not installed. Run: pip install mediapipe")
    USE_MP = False

# ------------------ Class List (104) with Source Video ------------------
# Mapping uses playlist index offset: original 1–8 -> here 3–10 (add +2)
# Colours are from Video 6 (as per your note).
CLASS_ITEMS = [
    # Greetings & social (Video 3)
    ("Hello", 3), ("Indian", 3), ("Namaste", 3), ("Bye-bye", 3),
    ("Thank you", 3), ("Please", 3), ("Sorry", 3), ("Welcome", 3),
    ("How are you?", 3), ("I'm fine", 3), ("My name is", 3), ("Again", 3),

    # Yes/No & daily basics (Video 4)
    ("Yes", 4), ("No", 4), ("Good", 4), ("Bad", 4), ("Correct", 4), ("Wrong", 4),
    ("Child", 4), ("Boy", 4), ("Girl", 4), ("Food", 4), ("Morning", 4),
    ("Good morning", 4), ("Good afternoon", 4), ("Good evening", 4),
    ("Good night", 4), ("Peace", 4), ("No fear", 4), ("Understand", 4),
    ("I don't understand", 4), ("Remember", 4),

    # Questions / deictics / time (Video 5)
    ("What", 5), ("Why", 5), ("How", 5), ("Where", 5), ("Who", 5),
    ("When", 5), ("Which", 5), ("This", 5), ("Time", 5), ("Place", 5),

    # People & pronouns (Video 3)
    ("I", 3), ("You", 3), ("He", 3), ("She", 3),
    ("Man", 3), ("Woman", 3), ("Deaf", 3), ("Hearing", 3), ("Teacher", 3),

    # Family & relations (Video 7)
    ("Family", 7), ("Mother", 7), ("Father", 7), ("Wife", 7), ("Husband", 7),
    ("Daughter", 7), ("Son", 7), ("Sister", 7), ("Brother", 7),
    ("Grandmother", 7), ("Grandfather", 7), ("Aunt", 7), ("Uncle", 7),

    # Calendar (Video 8 & 9)
    ("Day", 8), ("Week", 8), ("Monday", 8), ("Tuesday", 8), ("Wednesday", 8),
    ("Thursday", 8), ("Friday", 8), ("Saturday", 8), ("Sunday", 8),
    ("Month", 9), ("Year", 9),

    # Home / objects / states (Video 10)
    ("House", 10), ("Apartment", 10), ("Car", 10), ("Chair", 10), ("Table", 10),
    ("Happy", 10), ("Beautiful", 10), ("Ugly", 10), ("Tall", 10), ("Short", 10),
    ("Clever", 10), ("Sweet", 10), ("Bright", 10), ("Dark", 10),
    ("Camera", 10), ("Photo", 10), ("Work", 10),

    # Colours (Video 6)
    ("Colours", 6), ("Black", 6), ("Green", 6), ("Brown", 6), ("Red", 6),
    ("Pink", 6), ("Blue", 6), ("Yellow", 6), ("Orange", 6),
    ("Golden", 6), ("Silver", 6), ("Grey", 6),
]

# Derived lists/maps
LABELS = [name for name, _v in CLASS_ITEMS]
VIDEO_OF = {name: vid for name, vid in CLASS_ITEMS}

# ------------- Feature Layout -----------------
POSE_LM = 33     # (x,y,z,v)
FACE_LM = 468    # (x,y,z)
HAND_LM = 21     # (x,y,z)

POSE_DIM = POSE_LM * 4        # 132
FACE_DIM = FACE_LM * 3        # 1404
L_HAND_DIM = HAND_LM * 3      # 63
R_HAND_DIM = HAND_LM * 3      # 63
FRAME_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662

# ---- Face drawing style (UI only) ----
FACE_POINT_SPEC = mp_drawing.DrawingSpec(
    color=(180, 180, 180), thickness=1, circle_radius=1)
FACE_LINE_SPEC = mp_drawing.DrawingSpec(
    color=(140, 140, 140), thickness=1, circle_radius=0)
FACE_CONTOUR_SPEC = mp_drawing.DrawingSpec(
    color=(200, 200, 200), thickness=1, circle_radius=0)

# ------------------ Helpers ------------------
INVALID_FS_CHARS = set('<>:"/\\|?*')


def sanitize_dirname(label: str) -> str:
    """Convert a label to a safe folder name (Windows/macOS/Linux)."""
    s = "".join('_' if ch in INVALID_FS_CHARS else ch for ch in label)
    # Collapse spaces/underscores and trim
    s = s.replace("  ", " ").strip()
    s = s.replace("?", "")  # extra guard
    return s


def extract_keypoints(results) -> np.ndarray:
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        pose = np.zeros((POSE_DIM,), dtype=np.float32)

    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
                        dtype=np.float32).flatten()
    else:
        face = np.zeros((FACE_DIM,), dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        lh = np.zeros((L_HAND_DIM,), dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                      dtype=np.float32).flatten()
    else:
        rh = np.zeros((R_HAND_DIM,), dtype=np.float32)

    return np.concatenate([pose, face, lh, rh], axis=0)


def draw_overlay(img, action, video_id, clip_idx, frame_idx, total_clips, help_on=True):
    h, w = img.shape[:2]
    panel = img.copy()
    cv2.rectangle(panel, (0, 0), (w, 92), (0, 0, 0), -1)
    img = cv2.addWeighted(panel, 0.35, img, 0.65, 0)

    head = f"Class: {action}  |  Video: {video_id}"
    cv2.putText(img, head, (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Clip {clip_idx} / {total_clips}  |  Frame {frame_idx}/{T}",
                (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)

    if help_on:
        msg = u"[←/→]=select class  [SPACE]=start/next clip  [U]=undo  [P]=pause  [Q]=quit"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.putText(img, msg, (w - tw - 16, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (220, 220, 255), 2, cv2.LINE_AA)
    return img


def draw_countdown(img, secs_left):
    h, w = img.shape[:2]
    txt = f"Starting in {secs_left:.1f}s"
    cv2.putText(img, txt, (w//2 - 180, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (w//2 - 180, 110), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)


def next_clip_index(class_dir: Path) -> int:
    if not class_dir.is_dir():
        class_dir.mkdir(parents=True, exist_ok=True)
        return 0
    existing = [p.name for p in class_dir.iterdir() if p.is_dir()
                and p.name.startswith("clip_")]
    if not existing:
        return 0
    nums = []
    for name in existing:
        try:
            nums.append(int(name.replace("clip_", "")))
        except:
            pass
    return (max(nums) + 1) if nums else 0


def ensure_clip_folder(action_dir: Path, clip_idx: int) -> Path:
    d = action_dir / f"clip_{clip_idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_sequence_stack(clip_dir: Path, frames: list):
    arr = np.stack(frames, axis=0).astype(np.float32)
    np.save(str(clip_dir / "sequence.npy"), arr)


def main():
    if not USE_MP:
        return

    ap = argparse.ArgumentParser(
        description="Collect ISL word/phrase sequences with MediaPipe Holistic.")
    ap.add_argument("--root", type=str, default=RAW_ROOT,
                    help="Output root (RAW DATA)")
    ap.add_argument("--clips", type=int,
                    default=CLIPS_PER_CLASS, help="Clips per class")
    ap.add_argument("--frames", type=int, default=T, help="Frames per clip")
    ap.add_argument("--cam", type=int, default=CAM_INDEX, help="Camera index")
    ap.add_argument("--fps", type=float, default=FPS_TARGET,
                    help="Approx preview FPS")
    ap.add_argument("--scale", type=float, default=PREVIEW_SCALE,
                    help="Preview scaling (display only)")
    args = ap.parse_args()

    classes = LABELS[:]  # keep order
    T_frames = int(args.frames)
    clips_per_class = int(args.clips)
    scale = float(args.scale)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # Prepare class directories (sanitized)
    label_to_dir = {label: root / sanitize_dirname(label) for label in classes}
    for d in label_to_dir.values():
        d.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Classes (with source video):")
    for i, label in enumerate(classes, 1):
        print(f"  {i:3d}. {label}    (Video {VIDEO_OF[label]})")
    print("\n[INFO] Controls:")
    print("  ← / → : select class")
    print("  SPACE : start/next clip")
    print("  U     : undo last clip")
    print("  P     : pause/resume")
    print("  Q/ESC : quit")
    print("----------------------------------------------------")

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # Extended key codes (cv2.waitKeyEx)
    KEY_LEFT = 2424832
    KEY_RIGHT = 2555904

    with mp_holistic.Holistic(
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    ) as holistic:

        current_class_idx = None
        pause = False
        countdown_t0 = None
        collecting = False
        frames_collected = []
        last_saved_clip_dir = None

        # Count existing clips per label
        per_class_counts = {}
        for label in classes:
            cdir = label_to_dir[label]
            existing = [d for d in cdir.iterdir() if d.is_dir()
                        and d.name.startswith("clip_")]
            per_class_counts[label] = len(existing)

        target_gap = 1.0 / max(1e-6, float(args.fps))
        last_t = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ---- Draw landmarks (UI only) ----
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    panel, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_styles.get_default_pose_landmarks_style()
                )

            if results.face_landmarks:
                try:
                    mp_drawing.draw_landmarks(
                        panel, results.face_landmarks, mp_face.FACEMESH_TESSELATION,
                        FACE_POINT_SPEC, FACE_LINE_SPEC
                    )
                    mp_drawing.draw_landmarks(
                        panel, results.face_landmarks, mp_face.FACEMESH_CONTOURS,
                        FACE_POINT_SPEC, FACE_CONTOUR_SPEC
                    )
                except Exception:
                    mp_drawing.draw_landmarks(panel, results.face_landmarks)

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    panel, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    panel, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

            # HUD
            if current_class_idx is not None:
                label = classes[current_class_idx]
                video_id = VIDEO_OF[label]
                cur_count = per_class_counts.get(label, 0)
                panel = draw_overlay(
                    panel, label, video_id,
                    clip_idx=cur_count,
                    frame_idx=len(frames_collected) if collecting else 0,
                    total_clips=clips_per_class,
                    help_on=True
                )
            else:
                panel = draw_overlay(
                    panel, "<none>", "-", clip_idx=0, frame_idx=0,
                    total_clips=clips_per_class, help_on=True
                )

            # Countdown
            if countdown_t0 is not None and not pause:
                elapsed = time.time() - countdown_t0
                remain = max(0.0, COUNTDOWN_SEC - elapsed)
                draw_countdown(panel, remain)
                if remain <= 0.0:
                    countdown_t0 = None
                    collecting = True
                    frames_collected = []

            # Collect frames
            if collecting and current_class_idx is not None:
                now = time.time()
                if (now - last_t) >= target_gap:
                    last_t = now
                    vec = extract_keypoints(results)
                    frames_collected.append(vec)

                if len(frames_collected) >= T_frames:
                    label = classes[current_class_idx]
                    class_dir = label_to_dir[label]
                    idx = next_clip_index(class_dir)
                    clip_dir = ensure_clip_folder(class_dir, idx)

                    for k, v in enumerate(frames_collected[:T_frames]):
                        np.save(
                            str(clip_dir / f"f_{k:03d}.npy"), v.astype(np.float32))

                    save_sequence_stack(clip_dir, frames_collected[:T_frames])

                    last_saved_clip_dir = clip_dir
                    per_class_counts[label] = per_class_counts.get(
                        label, 0) + 1
                    collecting = False
                    frames_collected = []
                    time.sleep(BETWEEN_CLIPS_SEC)

            # Resize preview
            disp = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
                if abs(scale - 1.0) > 1e-3 else panel

            cv2.imshow(WINDOW, disp)
            key = cv2.waitKeyEx(1)

            # -------- Keys --------
            if key in (27, ord('q'), ord('Q')):
                break

            elif key in (ord('p'), ord('P')):
                pause = not pause

            elif key in (ord('u'), ord('U')):
                if last_saved_clip_dir and last_saved_clip_dir.exists():
                    try:
                        for f in last_saved_clip_dir.iterdir():
                            f.unlink()
                        last_saved_clip_dir.rmdir()
                        if current_class_idx is not None:
                            cname = classes[current_class_idx]
                            per_class_counts[cname] = max(
                                0, per_class_counts[cname] - 1)
                        print(f"[UNDO] Removed {last_saved_clip_dir}")
                        last_saved_clip_dir = None
                    except Exception as e:
                        print(f"[WARN] Undo failed: {e}")

            elif key == ord(' '):
                if current_class_idx is None:
                    print("[INFO] Select a class with ←/→ first.")
                else:
                    cname = classes[current_class_idx]
                    if per_class_counts[cname] >= clips_per_class:
                        print(
                            "[INFO] Target reached for this class. Choose another.")
                    else:
                        countdown_t0 = time.time()
                        collecting = False
                        frames_collected = []

            elif key == 2555904:  # RIGHT
                if len(classes) > 0:
                    current_class_idx = 0 if current_class_idx is None else (
                        current_class_idx + 1) % len(classes)
                    cname = classes[current_class_idx]
                    print(f"[SELECT] Class: {cname} (Video {VIDEO_OF[cname]})")
                    print(
                        f"         Count so far: {per_class_counts[cname]}/{clips_per_class}")

            elif key == 2424832:  # LEFT
                if len(classes) > 0:
                    current_class_idx = (len(
                        classes) - 1) if current_class_idx is None else (current_class_idx - 1) % len(classes)
                    cname = classes[current_class_idx]
                    print(f"[SELECT] Class: {cname} (Video {VIDEO_OF[cname]})")
                    print(
                        f"         Count so far: {per_class_counts[cname]}/{clips_per_class}")

        cap.release()
        cv2.destroyAllWindows()

    # Summary
    print("\n[SUMMARY]")
    total = 0
    for label in classes:
        cnt = per_class_counts.get(label, 0)
        total += cnt
        print(f"  {label}: {cnt} clips")
    print(f"Total clips: {total}\n")
    meta = {
        "root": str(root),
        "classes": classes,
        "video_map": VIDEO_OF,
        "frames_per_clip": T_frames,
        "clips_requested": clips_per_class,
        "counts": per_class_counts,
        "frame_dim": FRAME_DIM
    }
    with open(str(root / "collection_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[OK] collection_manifest.json written.")


if __name__ == "__main__":
    print("[INFO] Launching ISL Holistic Collector...")
    main()
