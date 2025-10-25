#!/usr/bin/env python3
import os
import re
import sys
import json
import shutil
import argparse
from pathlib import Path

import numpy as np

# ---------------- Config Defaults ----------------
DEST_ROOT = "RAW DATA"                 # final merged folder
FRAMES_PER_CLIP = 48                   # target frames per clip
MIN_KEEP_FRAMES = 40                   # drop if fewer than this
EXPECT_SEQ = True                      # require sequence.npy (fast path only)
OVERWRITE_DEST = True                  # if True, delete/recreate DEST_ROOT
FEAT_DIM = 1662                        # 33*4 + 468*3 + 21*3 + 21*3
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

# ---- Sanitizer (same as collector) ----
INVALID_FS_CHARS = set('<>:"/\\|?*')


def sanitize(label: str) -> str:
    s = "".join('_' if ch in INVALID_FS_CHARS else ch for ch in label)
    s = s.replace("  ", " ").strip()
    s = s.replace("?", "")  # extra guard
    return s


# ---- Patterns & helpers ----
clip_pat = re.compile(r"^clip_(\d{3})$")
FRAME_PAT = "f_{:03d}.npy"


def list_clip_dirs(class_dir: Path):
    """Return [(clip_path, idx), ...] sorted by idx (numeric)."""
    out = []
    if not class_dir.is_dir():
        return out
    # Using scandir via iterdir() is quite fast already
    for d in class_dir.iterdir():
        if d.is_dir():
            m = clip_pat.match(d.name)
            if m:
                out.append((d, int(m.group(1))))
    out.sort(key=lambda x: x[1])
    return out


def list_frame_files(clip_dir: Path):
    """Return sorted list of frame files f_XXX.npy."""
    frames = [p for p in clip_dir.iterdir()
              if p.is_file() and p.suffix.lower() == ".npy" and p.name.startswith("f_")]
    frames.sort(key=lambda p: p.name)
    return frames


def load_frame_vecs(frame_files):
    """Load frames into list[(1662,)] with basic validation."""
    vecs = []
    for f in frame_files:
        try:
            v = np.load(str(f))
            v = v.reshape(-1)
            if v.shape[0] == FEAT_DIM:
                vecs.append(v.astype(np.float32))
        except Exception:
            # skip broken frame
            pass
    return vecs


def temporal_standardize(vecs, target_T=48, min_keep=40):
    """
    vecs: list of (1662,) arrays
    Returns (T,1662) float32 or None if too short/empty.
    """
    n = len(vecs)
    if n == 0:
        return None
    if n >= target_T:
        # Uniform downsample to exactly T
        idx = np.linspace(0, n - 1, target_T).round().astype(int)
        out = np.stack([vecs[i] for i in idx], axis=0)
        return out.astype(np.float32)
    else:
        # n < target_T
        if n < min_keep:
            return None
        pad = target_T - n
        last = vecs[-1][None, :]
        out = np.vstack([np.stack(vecs, axis=0),
                         np.repeat(last, pad, axis=0)]).astype(np.float32)
        return out


def write_canonical_clip(dst_dir: Path, seq: np.ndarray):
    """
    seq: (T,1662)
    Writes f_000..f_047.npy + sequence.npy into dst directory.
    """
    dst_dir.mkdir(parents=True, exist_ok=False)
    T = seq.shape[0]
    for i in range(T):
        np.save(str(dst_dir / FRAME_PAT.format(i)), seq[i])
    np.save(str(dst_dir / "sequence.npy"), seq)


def fast_copy_clip(src_clip: Path, dst_dir: Path, hardlink=False):
    """
    Fast path for perfect clip (48 frames + sequence.npy).
    hardlink=True attempts os.link to save IO if on same filesystem.
    Returns True/False.
    """
    dst_dir.mkdir(parents=True, exist_ok=False)

    # sequence.npy first
    seq_src = src_clip / "sequence.npy"
    if not seq_src.is_file():
        return False

    def cp(src: Path, dst: Path):
        if hardlink:
            try:
                os.link(str(src), str(dst))
                return
            except Exception:
                pass
        shutil.copy2(str(src), str(dst))

    cp(seq_src, dst_dir / "sequence.npy")

    # copy frames f_000..f_047 strictly in order
    for i in range(FRAMES_PER_CLIP):
        fsrc = src_clip / FRAME_PAT.format(i)
        if not fsrc.is_file():
            return False
        cp(fsrc, dst_dir / fsrc.name)
    return True


def prepare_dest(dest_root: Path, labels):
    if OVERWRITE_DEST and dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    for lab in labels:
        (dest_root / sanitize(lab)).mkdir(parents=True, exist_ok=True)


def read_contributor_manifest(root: Path):
    mf = root / "collection_manifest.json"
    if mf.is_file():
        try:
            with open(mf, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def merge_main():
    ap = argparse.ArgumentParser(
        description="Merge RAW Data folders (Holistic sequences) into single 'RAW DATA' with standardization to 48 frames."
    )
    ap.add_argument("--sources", nargs="+", required=True,
                    help="Contributor roots (e.g., 'RAW Data Aryan' 'RAW Data Pranav' ...)")
    ap.add_argument("--dest", type=str, default=DEST_ROOT,
                    help="Destination root (default: 'RAW DATA')")
    ap.add_argument("--frames", type=int, default=FRAMES_PER_CLIP,
                    help="Target frames per clip (default 48)")
    ap.add_argument("--min_keep", type=int, default=MIN_KEEP_FRAMES,
                    help="Minimum frames to keep (pad up to target); else drop (default 40)")
    ap.add_argument("--no_require_seq", action="store_true",
                    help="Do not require sequence.npy for fast copy; will rebuild if missing")
    ap.add_argument("--keep_dest", action="store_true",
                    help="Append to existing dest instead of rebuilding from zero")
    ap.add_argument("--hardlink", action="store_true",
                    help="Try hardlink instead of copy for fast path (same filesystem)")
    args = ap.parse_args()

    target_T = int(args.frames)
    min_keep = int(args.min_keep)
    require_seq_fast = not args.no_require_seq
    dest_root = Path(args.dest)

    global OVERWRITE_DEST
    OVERWRITE_DEST = not args.keep_dest

    # Validate sources
    sources = [Path(s) for s in args.sources]
    any_source = False
    manifests = {}
    for s in sources:
        if s.is_dir():
            any_source = True
            manifests[str(s)] = read_contributor_manifest(s)
        else:
            print(f"[WARN] Missing source: {s}")
    if not any_source:
        print("[ERROR] None of the source folders exist; abort.")
        sys.exit(1)

    print(f"[INFO] Dest: {dest_root.resolve()}")
    print(
        f"[INFO] Overwrite dest: {OVERWRITE_DEST} | Target T: {target_T} | Min keep: {min_keep}")
    print(
        f"[INFO] Require sequence for fast-copy: {require_seq_fast} | Hardlink: {args.hardlink}")

    # Prepare dest structure
    prepare_dest(dest_root, LABELS)

    # Merge per class
    grand_total = 0
    class_summary = {}
    for lab in LABELS:
        s_lab = sanitize(lab)
        dest_class_dir = dest_root / s_lab

        # Start index depending on append mode
        start_idx = 0
        if args.keep_dest:
            existing = list_clip_dirs(dest_class_dir)
            start_idx = (max([i for _, i in existing]) + 1) if existing else 0

        copied = 0
        bad = 0
        fast_copied = 0
        standardized = 0
        dropped_short = 0
        rebuilt_seq = 0

        per_source_counts = {}

        # Stable ordering: by source position, then clip index
        for si, src in enumerate(sources):
            src_class_dir = src / s_lab
            if not src_class_dir.is_dir():
                per_source_counts[str(src)] = 0
                continue

            clip_list = list_clip_dirs(src_class_dir)
            accept_count = 0

            for (clip_dir, _) in clip_list:
                # Inspect frames
                frames = list_frame_files(clip_dir)
                n = len(frames)
                seq_path = clip_dir / "sequence.npy"

                # Fast path: exactly target_T frames + sequence.npy (unless disabled)
                if n == target_T and (seq_path.is_file() or not require_seq_fast):
                    # If sequence is missing but allowed, we still can fast-copy frames and rebuild seq in dest
                    new_dir = dest_class_dir / f"clip_{start_idx + copied:03d}"
                    # If require_seq_fast True but seq missing, we won't fast-copy; we'll rebuild below.
                    if seq_path.is_file() and require_seq_fast:
                        ok = fast_copy_clip(
                            clip_dir, new_dir, hardlink=args.hardlink)
                        if ok:
                            fast_copied += 1
                            copied += 1
                            accept_count += 1
                            continue
                        # If fast copy failed (rare), fall back to standardization route.

                    # Rebuild sequence from existing (exact) frames
                    vecs = load_frame_vecs(frames)
                    if len(vecs) == target_T:
                        seq = np.stack(vecs, axis=0).astype(np.float32)
                        write_canonical_clip(new_dir, seq)
                        rebuilt_seq += 1
                        copied += 1
                        accept_count += 1
                        continue
                    # If corrupt, fall through to standardize logic

                # Standardize: downsample/pad/drop
                if n >= target_T:
                    vecs = load_frame_vecs(frames)
                    if len(vecs) == 0:
                        bad += 1
                        continue
                    seq = temporal_standardize(
                        vecs, target_T=target_T, min_keep=min_keep)
                    if seq is None:
                        bad += 1
                        continue
                    new_dir = dest_class_dir / f"clip_{start_idx + copied:03d}"
                    write_canonical_clip(new_dir, seq)
                    standardized += 1
                    copied += 1
                    accept_count += 1
                elif min_keep <= n < target_T:
                    vecs = load_frame_vecs(frames)
                    if len(vecs) < min_keep:
                        dropped_short += 1
                        continue
                    seq = temporal_standardize(
                        vecs, target_T=target_T, min_keep=min_keep)
                    if seq is None:
                        dropped_short += 1
                        continue
                    new_dir = dest_class_dir / f"clip_{start_idx + copied:03d}"
                    write_canonical_clip(new_dir, seq)
                    standardized += 1
                    copied += 1
                    accept_count += 1
                else:
                    # too short
                    dropped_short += 1

            per_source_counts[str(src)] = accept_count

        class_summary[lab] = {
            "dest_class_dir": str(dest_class_dir),
            "copied_total": copied,
            "fast_copied": fast_copied,
            "standardized": standardized,
            "rebuilt_sequence": rebuilt_seq,
            "skipped_bad": bad,
            "dropped_too_short": dropped_short,
            "per_source": per_source_counts
        }
        grand_total += copied
        print(f"[CLASS] {lab:20s}  +{copied:3d}  (fast:{fast_copied} std:{standardized} rebuilt:{rebuilt_seq}  drop<{min_keep}:{dropped_short}  bad:{bad})")

    # Write merge manifest
    merge_manifest = {
        "dest_root": str(dest_root.resolve()),
        "target_frames": target_T,
        "min_keep": min_keep,
        "standardize_policy": {
            "downsample_if_gt": True,
            "pad_if_between": True,
            "drop_if_lt": True,
        },
        "fast_copy_requires_sequence": require_seq_fast,
        "hardlink_used": bool(args.hardlink),
        "sources": [str(s.resolve()) for s in sources],
        "class_list": LABELS,
        "summary": class_summary,
        "grand_total_clips": grand_total,
    }
    with open(dest_root / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(merge_manifest, f, indent=2)
    print("\n[RESULT]")
    print(f"  Grand total clips written: {grand_total}")
    print(f"  Manifest: {dest_root / 'merge_manifest.json'}")


if __name__ == "__main__":
    merge_main()
