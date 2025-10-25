import os
import sys
import glob
import time
import numpy as np
import cv2

# ====================== Config ======================
INPUT_ROOT = "RAW Images"     # source: 0-9, A-Z, blank (full images)
OUTPUT_ROOT = "MP_Dataset"    # destination: same structure, .npy per image
ALLOWED_EXT = (".jpg", ".jpeg", ".png")

# MediaPipe / detection
MAX_HANDS = 2
# First pass thresholds
MIN_DET_CONF = 0.35
MIN_TRK_CONF = 0.35
MODEL_COMPLEXITY = 1
# Fallback thresholds (if <2 hands in first pass)
FB_MIN_DET_CONF = 0.25
FB_MIN_TRK_CONF = 0.25
FB_MODEL_COMPLEXITY = 0
# Optional upscale fallback (helps when hands are small in the image)
ENABLE_UPSCALE_FALLBACK = True
# only for the fallback pass preview & detection, data saved from original
UPSCALE_FACTOR = 1.6

# Preview / GUI
PREVIEW = True
PREVIEW_WINDOW_NAME = "MP Keypoint Extractor (SPACE=pause, N=step, Q=quit)"
# only scales the GUI view (no effect on saved data)
PREVIEW_TARGET_WIDTH = 1100
SLOW_MODE_DELAY_MS = 20     # delay per processed image so you can actually see it

# Controls overlay
HELP_TEXT = "Keys: [SPACE]=pause/resume  [N]=step  [Q]/[ESC]=quit"

# Output vector layout: Left then Right (21 landmarks * (x,y,z) each)
VEC_LEN_PER_HAND = 21 * 3
TOTAL_VEC_LEN = VEC_LEN_PER_HAND * 2

# Create output root
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ====================== MediaPipe ======================
try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe is not installed. Run: python -m pip install mediapipe")
    sys.exit(1)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ====================== Helpers ======================
def list_class_dirs(root):
    if not os.path.isdir(root):
        return []
    names = [d for d in os.listdir(
        root) if os.path.isdir(os.path.join(root, d))]
    return sorted(names)


def ensure_outdir_for_class(cls):
    out_dir = os.path.join(OUTPUT_ROOT, cls)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def lm_to_vec(lm_obj):
    arr = np.zeros(VEC_LEN_PER_HAND, dtype=np.float32)
    if lm_obj is None:
        return arr
    idx = 0
    for p in lm_obj.landmark:
        arr[idx+0] = p.x
        arr[idx+1] = p.y
        arr[idx+2] = p.z
        idx += 3
    return arr


def extract_two_hand_keypoints(results):
    """
    Build fixed-length 126-dim vector:
      Left(63) then Right(63).
    Missing hands -> zeros.
    """
    vec_left = np.zeros(VEC_LEN_PER_HAND, dtype=np.float32)
    vec_right = np.zeros(VEC_LEN_PER_HAND, dtype=np.float32)

    if not results or not getattr(results, "multi_hand_landmarks", None):
        return np.concatenate([vec_left, vec_right], axis=0)

    # Build handedness map
    hand_map = {}  # 'Left' -> lm, 'Right' -> lm
    if getattr(results, "multi_handedness", None):
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handed.classification[0].label  # 'Left' or 'Right'
            hand_map[label] = lm
    else:
        # Fallback: just assign order
        lm_list = results.multi_hand_landmarks
        if len(lm_list) >= 1:
            hand_map["Right"] = lm_list[0]
        if len(lm_list) >= 2:
            hand_map["Left"] = lm_list[1]

    vec_left = lm_to_vec(hand_map.get('Left',  None))
    vec_right = lm_to_vec(hand_map.get('Right', None))
    return np.concatenate([vec_left, vec_right], axis=0)


def already_done(out_path):
    return os.path.isfile(out_path)


def resize_for_preview(img, target_w):
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / float(w)
    nh = int(h * scale)
    return cv2.resize(img, (target_w, nh), interpolation=cv2.INTER_AREA)


def draw_info_panel(img, cls, fname, idx, total, saved_for_class,
                    left_found, right_found, scores_text):
    """Overlay status text + simple progress on the frame."""
    h, w = img.shape[:2]
    pad = 10
    # translucent top bar
    panel = img.copy()
    cv2.rectangle(panel, (0, 0), (w, 88), (0, 0, 0), -1)
    img = cv2.addWeighted(panel, 0.35, img, 0.65, 0)

    cv2.putText(img, f"Class: {cls}", (pad+10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"File:  {fname}", (pad+10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    text = f"{idx}/{total}  saved_in_class: {saved_for_class}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(img, text, (w - tw - pad - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    lr_text = f"Left: {'yes' if left_found else 'no '} | Right: {'yes' if right_found else 'no '}  {scores_text}"
    (tw2, _), _ = cv2.getTextSize(lr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, lr_text, (w - tw2 - pad - 10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2, cv2.LINE_AA)

    # help text
    cv2.putText(img, HELP_TEXT, (pad+10, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
    return img


def draw_landmarks_with_labels(disp, results):
    left_found = False
    right_found = False
    scores_text = ""

    if not getattr(results, "multi_hand_landmarks", None):
        return disp, left_found, right_found, scores_text

    # Derive labels and scores (if available)
    labels, scores = [], []
    if getattr(results, "multi_handedness", None):
        for h in results.multi_handedness:
            labels.append(h.classification[0].label)     # 'Left' or 'Right'
            scores.append(h.classification[0].score)     # confidence
    else:
        labels = ["Right", "Left"][:len(results.multi_hand_landmarks)]
        scores = [None] * len(labels)

    for hand_lms, label, score in zip(results.multi_hand_landmarks, labels, scores):
        mp_draw.draw_landmarks(
            disp,
            hand_lms,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )
        h, w = disp.shape[:2]
        wx = int(hand_lms.landmark[0].x * w)
        wy = int(hand_lms.landmark[0].y * h)
        t = f"{label}{'' if score is None else f' {score:.2f}'}"
        cv2.putText(disp, t, (wx+8, wy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if label == "Left" else (255, 200, 0),
                    2, cv2.LINE_AA)
        left_found = left_found or (label == "Left")
        right_found = right_found or (label == "Right")

    # Compose a small compact score string for the header
    if scores:
        try:
            scores_text = " | ".join(
                f"{lab}:{(s if s is None else f'{s:.2f}')}"
                for lab, s in zip(labels, scores)
            )
        except Exception:
            scores_text = ""
    return disp, left_found, right_found, scores_text


# ====================== Main ======================
def main():
    classes = list_class_dirs(INPUT_ROOT)
    if not classes:
        print(f"ERROR: No class folders found under '{INPUT_ROOT}'.")
        return

    print("[INFO] Classes found:", classes)

    # Collect all files first (so we know totals)
    per_class_files = {}
    total_images = 0
    for cls in classes:
        in_dir = os.path.join(INPUT_ROOT, cls)
        files = []
        for ext in ALLOWED_EXT:
            files.extend(glob.glob(os.path.join(in_dir, f"*{ext}")))
        files = sorted(files)
        per_class_files[cls] = files
        total_images += len(files)

    print(f"[INFO] Total images to scan: {total_images}")

    processed = 0
    saved = 0
    paused = False
    step_mode = False  # when True, advance only when user presses 'N'

    # Primary detector
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    )

    if PREVIEW:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for cls in classes:
            out_dir = ensure_outdir_for_class(cls)
            image_paths = per_class_files[cls]
            if not image_paths:
                continue

            print(f"\n[CLASS] {cls}  |  images: {len(image_paths)}")
            saved_for_class = len(glob.glob(os.path.join(out_dir, "*.npy")))

            for i, img_path in enumerate(image_paths, start=1):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(out_dir, f"{stem}.npy")

                processed += 1

                bgr = cv2.imread(img_path)
                if bgr is None:
                    print(f"  [warn] Could not read {img_path}")
                    continue

                # ---------- First pass ----------
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                # ---------- Fallback pass (lower thresholds + simpler model) ----------
                need_fallback = (not getattr(results, "multi_hand_landmarks", None)) or \
                                (len(results.multi_hand_landmarks) < 2)

                if need_fallback:
                    with mp_hands.Hands(
                        static_image_mode=True,
                        max_num_hands=MAX_HANDS,
                        model_complexity=FB_MODEL_COMPLEXITY,
                        min_detection_confidence=FB_MIN_DET_CONF,
                        min_tracking_confidence=FB_MIN_TRK_CONF
                    ) as hands_fb:

                        # Optionally upscale for detection visibility
                        if ENABLE_UPSCALE_FALLBACK:
                            big = cv2.resize(
                                bgr, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
                            big_rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
                            big_rgb.flags.writeable = False
                            results_fb = hands_fb.process(big_rgb)
                            big_rgb.flags.writeable = True
                            # We only *display* the upscaled preview below; features are still saved from original coordinates
                        else:
                            rgb.flags.writeable = False
                            results_fb = hands_fb.process(rgb)
                            rgb.flags.writeable = True

                        # Use fallback if it’s better
                        count_orig = len(results.multi_hand_landmarks) if getattr(
                            results, "multi_hand_landmarks", None) else 0
                        count_fb = len(results_fb.multi_hand_landmarks) if getattr(
                            results_fb, "multi_hand_landmarks", None) else 0
                        if count_fb > count_orig:
                            results = results_fb

                # ---------- Save features ----------
                if not already_done(out_path):
                    vec = extract_two_hand_keypoints(results)
                    np.save(out_path, vec)
                    saved += 1
                    saved_for_class += 1

                # ---------- GUI preview ----------
                if PREVIEW:
                    disp = bgr.copy()
                    disp, left_found, right_found, scores_text = draw_landmarks_with_labels(
                        disp, results)
                    disp = draw_info_panel(
                        disp, cls, os.path.basename(img_path),
                        processed, total_images, saved_for_class,
                        left_found, right_found, scores_text
                    )
                    disp = resize_for_preview(disp, PREVIEW_TARGET_WIDTH)
                    cv2.imshow(PREVIEW_WINDOW_NAME, disp)

                    # slow down so the window is visible
                    wait_ms = SLOW_MODE_DELAY_MS
                    # If paused or step mode, handle keys in a small loop
                    end_time = time.time() + (wait_ms / 1000.0)
                    while True:
                        key = cv2.waitKey(20) & 0xFF
                        if key in (27, ord('q'), ord('Q')):
                            raise KeyboardInterrupt
                        elif key == 32:  # space -> pause/resume
                            paused = not paused
                        elif key in (ord('n'), ord('N')):  # step one image when paused
                            step_mode = True
                            paused = False
                            break

                        if not paused and not step_mode and time.time() >= end_time:
                            break

                        # When paused: keep waiting until user toggles or steps
                        if paused:
                            continue

                    # If in step mode, process only one image per 'N' press
                    if step_mode:
                        # enter a small loop to wait for the next 'N' or quit
                        while True:
                            key2 = cv2.waitKey(20) & 0xFF
                            if key2 in (27, ord('q'), ord('Q')):
                                raise KeyboardInterrupt
                            elif key2 == 32:  # space also resumes normal flow
                                paused = False
                                step_mode = False
                                break
                            elif key2 in (ord('n'), ord('N')):
                                # advance to next image
                                break

    except KeyboardInterrupt:
        # Quit from preview
        pass
    finally:
        hands.close()
        if PREVIEW:
            cv2.destroyAllWindows()

    print(
        f"\n[DONE] Scanned: {processed}  |  Saved new: {saved}  |  Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
