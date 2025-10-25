import os
import time
import cv2

# ---------------- Config ----------------
ROOT_DIR = 'RAW Images'
# 0-9 + A-Z + 'blank'
SUBFOLDERS = [str(d) for d in range(10)] + [chr(i)
                                            for i in range(65, 91)] + ['blank']
TARGET_PER_CLASS = 100          # <-- change this if you want a different target
START_DELAY_SEC = 7.0           # countdown before first capture
BETWEEN_CAP_SEC = 0.1          # delay between captures
WINDOW_NAME = "ISL Data Collector"
# ---------------------------------------

# Prepare folders (append-only if present)
os.makedirs(ROOT_DIR, exist_ok=True)
for name in SUBFOLDERS:
    os.makedirs(os.path.join(ROOT_DIR, name), exist_ok=True)


def safe_next_index(folder):
    """Return next numeric filename index (0.jpg, 1.jpg, ...) without gaps if possible."""
    existing = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    if not existing:
        return 0
    nums = []
    for f in existing:
        base, ext = os.path.splitext(f)
        if base.isdigit():
            nums.append(int(base))
    return (max(nums) + 1) if nums else len(existing)


def current_count(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])

# Map keys to class names


def key_to_class(k):
    # digits 0-9
    if ord('0') <= k <= ord('9'):
        return chr(k)
    # '.' -> blank
    if k == ord('.'):
        return 'blank'
    # letters a-z -> A-Z
    if ord('a') <= k <= ord('z'):
        return chr(k).upper()
    return None

# UI helpers


def draw_badge(img, text, org=(20, 35), color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)


def draw_countdown(img, secs_left):
    h, w = img.shape[:2]
    msg = f"Starting in {secs_left:.1f}s"
    cv2.putText(img, msg, (w//2 - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, msg, (w//2 - 200, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 255), 2, cv2.LINE_AA)


def draw_status_light(img, is_green):
    # Draw a small circle top-right: green if capture just happened, else red (cooldown)
    h, w = img.shape[:2]
    center = (w - 40, 40)
    color = (0, 200, 0) if is_green else (0, 0, 255)
    cv2.circle(img, center, 15, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(img, center, 15, color, -1, cv2.LINE_AA)
    cv2.putText(img, "Status", (w - 120, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_help(img):
    lines = [
        "Keys: [0-9], [a-z] for classes, '.' for blank",
        "ESC to quit, ';' to cancel current class",
    ]
    y = 80
    for ln in lines:
        cv2.putText(img, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


# Capture state
current_class = None
countdown_start_t = None
last_capture_t = None
session_done = True
just_captured = False  # for green flash after capture

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open camera 0")

print("Press [0-9], [a-z] or '.' (blank) to start collecting.")
print("ESC to quit, ';' to cancel current class.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for friendly UX; saved image is the full frame
        frame = cv2.flip(frame, 1)

        # If no active session, we are waiting for a key
        if session_done:
            draw_badge(
                frame, f"Target per class: {TARGET_PER_CLASS}", (20, 35))
            draw_help(frame)
            draw_badge(frame, "Waiting for key...", (20, 70), (0, 255, 255))
            draw_status_light(frame, is_green=False)

        # Handle countdown
        if current_class is not None and countdown_start_t is not None:
            elapsed = time.time() - countdown_start_t
            remaining = max(0.0, START_DELAY_SEC - elapsed)
            draw_badge(frame, f"Class: {current_class}",
                       (20, 35), (0, 255, 255))
            draw_countdown(frame, remaining)

            # show current count
            folder = os.path.join(ROOT_DIR, current_class)
            cnt = current_count(folder)
            draw_badge(frame, f"{cnt}/{TARGET_PER_CLASS}",
                       (20, 105), (200, 255, 200))

            if remaining <= 0:
                # start capture session
                countdown_start_t = None
                last_capture_t = 0.0
                session_done = False
                just_captured = False

        # Active capture session
        if current_class is not None and not session_done and countdown_start_t is None:
            folder = os.path.join(ROOT_DIR, current_class)
            cnt = current_count(folder)
            draw_badge(frame, f"Class: {current_class}",
                       (20, 35), (0, 255, 255))
            draw_badge(frame, f"{cnt}/{TARGET_PER_CLASS}",
                       (20, 105), (200, 255, 200))

            now = time.time()
            # Are we allowed to capture a new image (time gap met and not over target)?
            if cnt < TARGET_PER_CLASS and (last_capture_t == 0.0 or (now - last_capture_t) >= BETWEEN_CAP_SEC):
                # Capture and save full frame
                idx = safe_next_index(folder)
                out_path = os.path.join(folder, f"{idx}.jpg")
                cv2.imwrite(out_path, frame)
                last_capture_t = now
                just_captured = True
            else:
                # In cooldown / or already done
                just_captured = False

            # Status light: green right after capture, red during waiting
            draw_status_light(frame, is_green=just_captured)

            # If target met, end session
            if current_count(folder) >= TARGET_PER_CLASS:
                draw_badge(frame, "Completed this class!",
                           (20, 145), (0, 255, 0))
                session_done = True
                current_class = None
                countdown_start_t = None
                last_capture_t = None

        # Show window
        cv2.imshow(WINDOW_NAME, frame)

        # Key handling
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        elif k in (ord(';'), ord(';')):
            # cancel current class session
            current_class = None
            session_done = True
            countdown_start_t = None
            last_capture_t = None
        else:
            chosen = key_to_class(k)
            if chosen is not None:
                # Start a new countdown for that class
                current_class = chosen
                countdown_start_t = time.time()
                session_done = False
                last_capture_t = None
                just_captured = False

finally:
    cap.release()
    cv2.destroyAllWindows()
