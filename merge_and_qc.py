#!/usr/bin/env python3
import os
import re
import sys
import json
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# ---------------- Config Defaults ----------------
FRAMES_TARGET = 48               # standard target frames per clip
MIN_KEEP = 40                    # minimum raw frames to attempt padding
EXPECT_SEQ = True                # prefer sequence.npy if present
USE_HARDLINK_WHEN_POSSIBLE = False  # only for unmodified pass-through clips
MAX_WORKERS = max(1, os.cpu_count() // 2)
# -------------------------------------------------

# ---- Class list (104) identical to your collector ----
CLASS_ITEMS = [
    ("Hello", 3), ("Indian", 3), ("Namaste", 3), ("Bye-bye", 3),
    ("Thank you", 3), ("Please", 3), ("Sorry", 3), ("Welcome", 3),
    ("How are you?", 3), ("I'm fine", 3), ("My name is", 3), ("Again", 3),

    ("Yes", 4), ("No", 4), ("Good", 4), ("Bad", 4), ("Correct", 4), ("Wrong", 4),
    ("Child", 4), ("Boy", 4), ("Girl", 4), ("Food", 4), ("Morning", 4),
    ("Good morning", 4), ("Good afternoon", 4), ("Good evening", 4),
    ("Good night", 4), ("Peace", 4), ("No fear", 4), ("Understand", 4),
    ("I don't understand", 4), ("Remember", 4),

    ("What", 5), ("Why", 5), ("How", 5), ("Where", 5), ("Who", 5),
    ("When", 5), ("Which", 5), ("This", 5), ("Time", 5), ("Place", 5),

    ("I", 3), ("You", 3), ("He", 3), ("She", 3),
    ("Man", 3), ("Woman", 3), ("Deaf", 3), ("Hearing", 3), ("Teacher", 3),

    ("Family", 7), ("Mother", 7), ("Father", 7), ("Wife", 7), ("Husband", 7),
    ("Daughter", 7), ("Son", 7), ("Sister", 7), ("Brother", 7),
    ("Grandmother", 7), ("Grandfather", 7), ("Aunt", 7), ("Uncle", 7),

    ("Day", 8), ("Week", 8), ("Monday", 8), ("Tuesday", 8), ("Wednesday", 8),
    ("Thursday", 8), ("Friday", 8), ("Saturday", 8), ("Sunday", 8),
    ("Month", 9), ("Year", 9),

    ("House", 10), ("Apartment", 10), ("Car", 10), ("Chair", 10), ("Table", 10),
    ("Happy", 10), ("Beautiful", 10), ("Ugly", 10), ("Tall", 10), ("Short", 10),
    ("Clever", 10), ("Sweet", 10), ("Bright", 10), ("Dark", 10),
    ("Camera", 10), ("Photo", 10), ("Work", 10),

    ("Colours", 6), ("Black", 6), ("Green", 6), ("Brown", 6), ("Red", 6),
    ("Pink", 6), ("Blue", 6), ("Yellow", 6), ("Orange", 6),
    ("Golden", 6), ("Silver", 6), ("Grey", 6),
]
LABELS = [n for n, _ in CLASS_ITEMS]

# ---- Feature layout (1662-dim/frame) ----
POSE_LM = 33
FACE_LM = 468
HAND_LM = 21

POSE_DIM = POSE_LM * 4          # 132
FACE_DIM = FACE_LM * 3          # 1404
L_HAND_DIM = HAND_LM * 3        # 63
R_HAND_DIM = HAND_LM * 3        # 63
FRAME_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662

# ---- Sanitizer (same as collector) ----
INVALID_FS_CHARS = set('<>:"/\\|?*')


def sanitize(label: str) -> str:
    s = "".join('_' if ch in INVALID_FS_CHARS else ch for ch in label)
    s = s.replace("  ", " ").strip()
    s = s.replace("?", "")
    return s

# ------------ Presence / QC helpers -------------


def split_streams(frame_vec):
    idx = 0
    pose = frame_vec[idx:idx+POSE_DIM].reshape(POSE_LM, 4)
    idx += POSE_DIM
    face = frame_vec[idx:idx+FACE_DIM].reshape(FACE_LM, 3)
    idx += FACE_DIM
    lh = frame_vec[idx:idx+L_HAND_DIM].reshape(HAND_LM, 3)
    idx += L_HAND_DIM
    rh = frame_vec[idx:idx+R_HAND_DIM].reshape(HAND_LM, 3)
    return pose, face, lh, rh


def presence_flags(frame_vec):
    pose, face, lh, rh = split_streams(frame_vec)
    pose_present = np.sum(np.abs(pose)) > 0
    face_present = np.sum(np.abs(face)) > 0
    lh_present = np.sum(np.abs(lh)) > 0
    rh_present = np.sum(np.abs(rh)) > 0
    any_hand = lh_present or rh_present
    return pose_present, face_present, any_hand


def norm01_out_of_range_ratio(arr):
    T, D = arr.shape
    idxs = []
    base = 0
    for i in range(POSE_LM):
        idxs.extend([base + i*4 + 0, base + i*4 + 1])
    base += POSE_DIM
    for i in range(FACE_LM):
        idxs.extend([base + i*3 + 0, base + i*3 + 1])
    base += FACE_DIM
    for i in range(HAND_LM):
        idxs.extend([base + i*3 + 0, base + i*3 + 1])
    base += L_HAND_DIM
    for i in range(HAND_LM):
        idxs.extend([base + i*3 + 0, base + i*3 + 1])
    xy = arr[:, idxs]
    total = xy.size
    if total == 0:
        return 0.0
    out = (xy < 0.0) | (xy > 1.0)
    return float(out.sum()) / float(total)


def hand_span(frame_vec):
    _, _, lh, rh = split_streams(frame_vec)

    def span(hand):
        if np.allclose(hand, 0.0):
            return 0.0
        wrist = hand[0, :2]
        d = hand[:, :2] - wrist[None, :]
        dist = np.sqrt((d**2).sum(axis=1))
        return float(np.max(dist))
    return span(lh), span(rh)


def clip_motion_energy(arr):
    dif = np.diff(arr, axis=0)
    return float(np.sum(dif**2))


# ---------- Class-wise QC overrides ----------
HEAD_ONLY = {"Yes", "No"}          # head-only nod/shake
LOW_MOTION = {"Namaste", "Boy"}    # small/short movements


def qc_clip(arr, params):
    """
    params dict:
      min_pose_ratio, min_face_ratio, min_anyhand_ratio,
      max_gap, max_oob_ratio, min_hand_span, min_motion
    """
    T = arr.shape[0]
    pres = [presence_flags(arr[t]) for t in range(T)]
    pose_present = np.array([p[0] for p in pres], dtype=bool)
    face_present = np.array([p[1] for p in pres], dtype=bool)
    any_hand = np.array([p[2] for p in pres], dtype=bool)

    pose_ratio = pose_present.mean()
    face_ratio = face_present.mean()
    hand_ratio = any_hand.mean()

    none_present = ~(pose_present | face_present | any_hand)
    max_consec = 0
    cur = 0
    for v in none_present:
        if v:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    oob = norm01_out_of_range_ratio(arr)
    spans = [hand_span(arr[t]) for t in range(T)]
    max_any_span = max(max(s) for s in spans)
    motion = clip_motion_energy(arr)

    reasons = []
    if pose_ratio < params["min_pose_ratio"]:
        reasons.append(f"low_pose_ratio={pose_ratio:.2f}")
    if face_ratio < params["min_face_ratio"]:
        reasons.append(f"low_face_ratio={face_ratio:.2f}")
    if hand_ratio < params["min_anyhand_ratio"]:
        reasons.append(f"low_anyhand_ratio={hand_ratio:.2f}")
    if max_consec > params["max_gap"]:
        reasons.append(f"long_missing_gap={max_consec}")
    if oob > params["max_oob_ratio"]:
        reasons.append(f"out_of_bounds_ratio={oob:.2f}")
    if max_any_span < params["min_hand_span"]:
        reasons.append(f"tiny_hand_span={max_any_span:.3f}")
    if motion < params["min_motion"]:
        reasons.append(f"near_zero_motion={motion:.2e}")

    ok = (len(reasons) == 0)
    metrics = {
        "pose_ratio": pose_ratio, "face_ratio": face_ratio, "anyhand_ratio": hand_ratio,
        "max_missing_gap": max_consec, "out_of_bounds_ratio": oob,
        "max_hand_span": max_any_span, "motion_energy": motion
    }
    return ok, reasons, metrics


def get_qc_params_for_label(label, base):
    """
    Returns params dict per label by applying overrides on top of base.
    """
    p = dict(base)  # copy
    if label in HEAD_ONLY:
        # Head-only nod/shake: don't require hands or hand-span; allow tiny motion threshold.
        p["min_anyhand_ratio"] = 0.0
        p["min_hand_span"] = 0.0
        p["min_motion"] = 1e-8
        # Slightly more tolerant of missing pose/face frames
        p["min_pose_ratio"] = min(p["min_pose_ratio"], 0.50)
        p["min_face_ratio"] = min(p["min_face_ratio"], 0.50)
        p["max_gap"] = max(p["max_gap"], 16)
    elif label in LOW_MOTION:
        # Low-motion signs: reduce motion & hand-span requirement, and hand presence a bit.
        p["min_motion"] = min(p["min_motion"], 5e-8)
        p["min_hand_span"] = min(p["min_hand_span"], 0.005)
        p["min_anyhand_ratio"] = min(p["min_anyhand_ratio"], 0.40)
        p["max_gap"] = max(p["max_gap"], 14)
    return p


# ---- Merge helpers ----
clip_pat = re.compile(r"^clip_(\d{3})$")


def list_clip_dirs(class_dir: Path):
    out = []
    if not class_dir.is_dir():
        return out
    for d in class_dir.iterdir():
        if d.is_dir():
            m = clip_pat.match(d.name)
            if m:
                out.append((d, int(m.group(1))))
    out.sort(key=lambda x: x[1])
    return out


def load_clip_to_array(clip_dir: Path, expect_seq=True):
    """
    Loads raw array from sequence.npy if present (any T), else stacks f_*.npy.
    Returns (arr_raw (T,D) float32, frames_paths[list or None], used_seq_path or None)
    """
    seq_path = clip_dir / "sequence.npy"
    if expect_seq and seq_path.is_file():
        try:
            arr = np.load(seq_path)
            if arr.ndim == 2 and arr.shape[1] == FRAME_DIM and np.isfinite(arr).all():
                return arr.astype(np.float32), None, str(seq_path)
        except Exception:
            pass

    frames = sorted([p for p in clip_dir.iterdir()
                     if p.is_file() and p.suffix.lower() == ".npy" and p.name.startswith("f_")])
    if not frames:
        return None, None, None
    rows = []
    for f in frames:
        try:
            v = np.load(f)
            v = v.reshape(-1)
            if v.shape[0] != FRAME_DIM:
                return None, None, None
            if not np.isfinite(v).all():
                return None, None, None
            rows.append(v.astype(np.float32))
        except Exception:
            return None, None, None
    arr = np.stack(rows, axis=0)
    return arr, [str(p) for p in frames], None


def standardize_T(arr_raw: np.ndarray, T_target=48, min_keep=40):
    T = arr_raw.shape[0]
    if T == T_target:
        return arr_raw, "pass", None  # no change
    if T > T_target:
        idx = np.linspace(0, T - 1, T_target).round().astype(int)
        return arr_raw[idx], "downsample", idx.tolist()
    if T < min_keep:
        return None, "too_short", None
    pad_len = T_target - T
    last = arr_raw[-1:]
    pad = np.repeat(last, pad_len, axis=0)
    return np.concatenate([arr_raw, pad], axis=0), "pad", None


def write_clip(dest_class_dir: Path, arr_std: np.ndarray, new_index: int,
               from_frames_paths=None, unchanged_pass=False, hardlink=False):
    out_dir = dest_class_dir / f"clip_{new_index:03d}"
    out_dir.mkdir(parents=True, exist_ok=False)
    np.save(str(out_dir / "sequence.npy"), arr_std.astype(np.float32))

    if unchanged_pass and (from_frames_paths is not None) and hardlink:
        for k, src in enumerate(from_frames_paths[:FRAMES_TARGET]):
            dst = out_dir / f"f_{k:03d}.npy"
            try:
                os.link(src, str(dst))
            except Exception:
                shutil.copy2(src, str(dst))
    else:
        for k in range(arr_std.shape[0]):
            np.save(str(out_dir / f"f_{k:03d}.npy"),
                    arr_std[k].astype(np.float32))
    return out_dir


def process_one_clip(clip_dir: str,
                     dest_class_dir: str,
                     next_index: int,
                     qc_base_params: dict,
                     label: str,
                     frames_target=48,
                     min_keep=40,
                     expect_seq=True,
                     hardlink_passthrough=False):
    """
    Worker: loads -> standardizes -> QC(with overrides) -> writes if OK.
    Returns dict with status.
    """
    clip_dir = Path(clip_dir)
    dest_class_dir = Path(dest_class_dir)
    info = {"ok": False, "written": False, "reason": None, "new_index": None}

    arr_raw, frames_paths, _ = load_clip_to_array(
        clip_dir, expect_seq=expect_seq)
    if arr_raw is None:
        info["reason"] = "load_fail"
        return info

    arr_std, how, _ = standardize_T(
        arr_raw, T_target=frames_target, min_keep=min_keep)
    if arr_std is None:
        info["reason"] = how  # too_short
        return info

    qc_params = get_qc_params_for_label(label, qc_base_params)
    ok, reasons, _ = qc_clip(arr_std, qc_params)
    if not ok:
        info["reason"] = "qc_fail:" + ",".join(reasons[:3])
        return info

    unchanged_pass = (how == "pass")
    out_dir = write_clip(
        dest_class_dir, arr_std, next_index,
        from_frames_paths=frames_paths, unchanged_pass=unchanged_pass,
        hardlink=hardlink_passthrough
    )
    info["ok"] = True
    info["written"] = True
    info["new_index"] = next_index
    return info


def collect_all_source_clips_for_label(label: str, sources_roots):
    s_lab = sanitize(label)
    flat = []
    for si, s in enumerate(sources_roots):
        cdir = s / s_lab
        if not cdir.is_dir():
            continue
        for (clip_path, idx) in list_clip_dirs(cdir):
            flat.append((si, idx, clip_path))
    flat.sort(key=lambda t: (t[0], t[1]))  # by source order, then clip index
    return [p for (_si, _idx, p) in flat]


def merge_and_qc_main():
    ap = argparse.ArgumentParser(
        description="One-pass Merge + Standardize + QC holistic sequences into a CLEAN root (with class-wise QC overrides)."
    )
    ap.add_argument("--sources", nargs="+", required=True,
                    help="Contributor roots (e.g., 'RAW Data Aryan' 'RAW Data Pranav' ...)")
    ap.add_argument("--dest_clean", type=str, default="RAW DATA_CLEAN",
                    help="Destination cleaned root (default: 'RAW DATA_CLEAN')")
    ap.add_argument("--frames", type=int, default=FRAMES_TARGET,
                    help="Target frames per clip (default 48)")
    ap.add_argument("--min_keep", type=int, default=MIN_KEEP,
                    help="Minimum raw frames to allow padding (default 40)")
    ap.add_argument("--expect_seq", action="store_true",
                    help="Prefer sequence.npy if present")
    ap.add_argument("--hardlink", action="store_true",
                    help="If a clip is already 48 and passes QC, hardlink frames instead of copying")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help=f"Parallel workers (default {MAX_WORKERS})")

    # Base QC thresholds (overrides may apply per class)
    ap.add_argument("--min_pose_ratio", type=float, default=0.80)
    ap.add_argument("--min_face_ratio", type=float, default=0.60)
    ap.add_argument("--min_anyhand_ratio", type=float, default=0.70)
    ap.add_argument("--max_gap", type=int, default=12)
    ap.add_argument("--max_oob_ratio", type=float, default=0.10)
    ap.add_argument("--min_hand_span", type=float, default=0.02)
    ap.add_argument("--min_motion", type=float, default=1e-6)

    args = ap.parse_args()

    sources = [Path(s) for s in args.sources]
    if not any(s.is_dir() for s in sources):
        raise SystemExit("[ERROR] No valid source directories.")

    dest_clean = Path(args.dest_clean)
    if dest_clean.exists():
        shutil.rmtree(dest_clean)
    dest_clean.mkdir(parents=True, exist_ok=True)

    # Prepare dest class dirs
    label_to_dir = {lab: dest_clean / sanitize(lab) for lab in LABELS}
    for d in label_to_dir.values():
        d.mkdir(parents=True, exist_ok=True)

    qc_base = dict(
        min_pose_ratio=args.min_pose_ratio,
        min_face_ratio=args.min_face_ratio,
        min_anyhand_ratio=args.min_anyhand_ratio,
        max_gap=args.max_gap,
        max_oob_ratio=args.max_oob_ratio,
        min_hand_span=args.min_hand_span,
        min_motion=args.min_motion
    )

    summary = {"classes": {}, "frames_target": args.frames,
               "min_keep": args.min_keep}
    grand_written = 0
    grand_dropped = 0

    for lab in LABELS:
        cdir = label_to_dir[lab]
        clips_paths = collect_all_source_clips_for_label(lab, sources)
        if not clips_paths:
            summary["classes"][lab] = {
                "source_clips": 0, "written": 0, "dropped": 0}
            continue

        # Submit tasks in order; allocate indices in submission order.
        futures = []
        next_idx = 0
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            for cp in clips_paths:
                futures.append(
                    ex.submit(
                        process_one_clip,
                        str(cp),
                        str(cdir),
                        next_idx,
                        qc_base,
                        lab,
                        frames_target=args.frames,
                        min_keep=args.min_keep,
                        expect_seq=args.expect_seq,
                        hardlink_passthrough=args.hardlink
                    )
                )
                next_idx += 1

            # Gather results
            results = [f.result() for f in as_completed(futures)]

        # Compact indices (remove gaps due to drops)
        created = list_clip_dirs(cdir)
        for new_i, (clip_path, _) in enumerate(created):
            if clip_path.name != f"clip_{new_i:03d}":
                target = cdir / f"clip_{new_i:03d}"
                clip_path.rename(target)
        written = len(created)
        dropped = len(clips_paths) - written
        grand_written += written
        grand_dropped += dropped
        summary["classes"][lab] = {"source_clips": len(clips_paths),
                                   "written": written, "dropped": dropped}
        print(
            f"[CLASS] {lab:20s}  source={len(clips_paths):4d}  kept={written:4d}  dropped={dropped:4d}")

    summary["sources"] = [str(s.resolve()) for s in sources]
    summary["dest_clean"] = str(dest_clean.resolve())
    with open(dest_clean / "merge_qc_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n[RESULT]")
    print(f"  Total written (clean): {grand_written}")
    print(f"  Total dropped:         {grand_dropped}")
    print(f"  Manifest: {dest_clean / 'merge_qc_manifest.json'}")


if __name__ == "__main__":
    merge_and_qc_main()
