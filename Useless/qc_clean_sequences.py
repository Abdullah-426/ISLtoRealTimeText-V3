#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import shutil
import numpy as np

POSE_LM = 33
FACE_LM = 468
HAND_LM = 21

POSE_DIM = POSE_LM * 4
FACE_DIM = FACE_LM * 3
L_HAND_DIM = HAND_LM * 3
R_HAND_DIM = HAND_LM * 3
FRAME_DIM = POSE_DIM + FACE_DIM + L_HAND_DIM + R_HAND_DIM  # 1662


def list_classes(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_clips(class_dir: Path):
    return sorted([p for p in class_dir.iterdir() if p.is_dir() and p.name.startswith("clip_")])


def load_sequence(clip_dir: Path, T: int):
    seq_path = clip_dir / "sequence.npy"
    if not seq_path.is_file():
        return None, "missing_sequence"
    try:
        arr = np.load(seq_path)
    except Exception:
        return None, "load_error"

    if arr.ndim != 2 or arr.shape[0] != T or arr.shape[1] != FRAME_DIM:
        return None, f"bad_shape_{arr.shape if arr is not None else 'none'}"
    if not np.isfinite(arr).all():
        return None, "nan_inf"
    return arr.astype(np.float32), None


def split_streams(frame_vec):
    """Return (pose(33,4), face(468,3), lh(21,3), rh(21,3)) from (1662,)"""
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
    return pose_present, face_present, any_hand, lh_present, rh_present


def norm01_out_of_range_ratio(arr):
    """arr is (T, D). Count coordinates outside [0,1] in x,y components only (z is free)."""
    T, D = arr.shape
    # Build mask to pick only x,y indices
    # Pose: 33*(x,y,z,v). x=0, y=1 in 4-stride
    # Face: 468*(x,y,z). x=0, y=1 in 3-stride
    # Hands: 21*(x,y,z). x=0, y=1 in 3-stride
    idxs = []
    # pose x,y
    base = 0
    for i in range(POSE_LM):
        idxs.extend([base + i*4 + 0, base + i*4 + 1])
    base += POSE_DIM
    # face x,y
    for i in range(FACE_LM):
        idxs.extend([base + i*3 + 0, base + i*3 + 1])
    base += FACE_DIM
    # lh x,y
    for i in range(HAND_LM):
        idxs.extend([base + i*3 + 0, base + i*3 + 1])
    base += L_HAND_DIM
    # rh x,y
    for i in range(HAND_LM):
        idxs.extend([base + i*3 + 0, base + i*3 + 1])

    xy = arr[:, idxs]
    total = xy.size
    if total == 0:
        return 0.0
    out = (xy < 0.0) | (xy > 1.0)
    return float(out.sum()) / float(total)


def hand_span(frame_vec):
    """Compute per-hand wrist-centered span (max L2 from wrist over 21 points in x,y)."""
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
    """Sum L2 diff over time across all dims."""
    dif = np.diff(arr, axis=0)
    return float(np.sum(dif**2))


def qc_clip(arr,
            min_pose_ratio=0.80,
            min_face_ratio=0.60,
            min_anyhand_ratio=0.70,
            max_gap=12,
            max_oob_ratio=0.10,
            min_hand_span=0.02,
            min_motion=1e-6):
    T = arr.shape[0]
    pres = [presence_flags(arr[t]) for t in range(T)]
    pose_present = np.array([p[0] for p in pres], dtype=bool)
    face_present = np.array([p[1] for p in pres], dtype=bool)
    any_hand = np.array([p[2] for p in pres], dtype=bool)

    pose_ratio = pose_present.mean()
    face_ratio = face_present.mean()
    hand_ratio = any_hand.mean()

    # longest missing streak (none of pose+face+hand present)
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
    # spans
    spans = [hand_span(arr[t]) for t in range(T)]
    max_lh_span = max(s[0] for s in spans)
    max_rh_span = max(s[1] for s in spans)
    max_any_span = max(max_lh_span, max_rh_span)

    motion = clip_motion_energy(arr)

    reasons = []
    if pose_ratio < min_pose_ratio:
        reasons.append(f"low_pose_ratio={pose_ratio:.2f}")
    if face_ratio < min_face_ratio:
        reasons.append(f"low_face_ratio={face_ratio:.2f}")
    if hand_ratio < min_anyhand_ratio:
        reasons.append(f"low_anyhand_ratio={hand_ratio:.2f}")
    if max_consec > max_gap:
        reasons.append(f"long_missing_gap={max_consec}")
    if oob > max_oob_ratio:
        reasons.append(f"out_of_bounds_ratio={oob:.2f}")
    if max_any_span < min_hand_span:
        reasons.append(f"tiny_hand_span={max_any_span:.3f}")
    if motion < min_motion:
        reasons.append(f"near_zero_motion={motion:.2e}")

    ok = len(reasons) == 0
    metrics = {
        "pose_ratio": pose_ratio,
        "face_ratio": face_ratio,
        "anyhand_ratio": hand_ratio,
        "max_missing_gap": max_consec,
        "out_of_bounds_ratio": oob,
        "max_hand_span": max_any_span,
        "motion_energy": motion
    }
    return ok, reasons, metrics


def copy_clip(src_clip: Path, dst_class_dir: Path, dst_index: int):
    new_dir = dst_class_dir / f"clip_{dst_index:03d}"
    new_dir.mkdir(parents=True, exist_ok=False)
    # copy sequence.npy
    seq = src_clip / "sequence.npy"
    if seq.is_file():
        shutil.copy2(str(seq), str(new_dir / "sequence.npy"))
    # frames
    frames = sorted([p for p in src_clip.iterdir()
                     if p.is_file() and p.suffix.lower() == ".npy" and p.name.startswith("f_")])
    for f in frames:
        shutil.copy2(str(f), str(new_dir / f.name))
    return new_dir


def main():
    ap = argparse.ArgumentParser(
        description="QC + Clean Holistic sequences into RAW DATA_CLEAN/")
    ap.add_argument("--src", type=str, default="RAW DATA", help="Input root")
    ap.add_argument("--dst", type=str,
                    default="RAW DATA_CLEAN", help="Output root")
    ap.add_argument("--frames", type=int, default=48, help="Frames/clip")
    ap.add_argument("--min_pose_ratio", type=float, default=0.80)
    ap.add_argument("--min_face_ratio", type=float, default=0.60)
    ap.add_argument("--min_anyhand_ratio", type=float, default=0.70)
    ap.add_argument("--max_gap", type=int, default=12)
    ap.add_argument("--max_oob_ratio", type=float, default=0.10)
    ap.add_argument("--min_hand_span", type=float, default=0.02)
    ap.add_argument("--min_motion", type=float, default=1e-6)
    ap.add_argument("--dry_run", action="store_true",
                    help="Do not copy, only report")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"[ERROR] Missing source: {src}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    manifest = {
        "src": str(src.resolve()),
        "dst": str(dst.resolve()),
        "frames": args.frames,
        "thresholds": {
            "min_pose_ratio": args.min_pose_ratio,
            "min_face_ratio": args.min_face_ratio,
            "min_anyhand_ratio": args.min_anyhand_ratio,
            "max_gap": args.max_gap,
            "max_oob_ratio": args.max_oob_ratio,
            "min_hand_span": args.min_hand_span,
            "min_motion": args.min_motion
        },
        "classes": {}
    }

    total_kept = 0
    total_bad = 0

    classes = list_classes(src)
    for cdir in classes:
        cname = cdir.name
        out_cdir = dst / cname
        out_cdir.mkdir(parents=True, exist_ok=True)

        clips = list_clips(cdir)
        kept = 0
        bad = 0
        clip_reports = []
        for clip_i, clip_dir in enumerate(clips):
            seq, err = load_sequence(clip_dir, args.frames)
            if err:
                bad += 1
                clip_reports.append(
                    {"clip": clip_dir.name, "ok": False, "reason": err})
                continue

            ok, reasons, metrics = qc_clip(
                seq,
                args.min_pose_ratio, args.min_face_ratio, args.min_anyhand_ratio,
                args.max_gap, args.max_oob_ratio, args.min_hand_span, args.min_motion
            )
            if ok:
                if not args.dry_run:
                    copy_clip(clip_dir, out_cdir, kept)
                kept += 1
                clip_reports.append(
                    {"clip": clip_dir.name, "ok": True, "metrics": metrics})
            else:
                bad += 1
                clip_reports.append(
                    {"clip": clip_dir.name, "ok": False, "reasons": reasons, "metrics": metrics})

        manifest["classes"][cname] = {
            "source_clips": len(clips),
            "kept": kept,
            "dropped": bad,
            "clips": clip_reports
        }
        total_kept += kept
        total_bad += bad
        print(f"[QC] {cname:20s} kept={kept:3d}  dropped={bad:3d}")

    manifest["summary"] = {
        "total_kept": total_kept, "total_dropped": total_bad}
    with open(dst / "qc_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[RESULT] Kept={total_kept}  Dropped={total_bad}")
    print(f"[OK] qc_manifest.json -> {dst/'qc_manifest.json'}")


if __name__ == "__main__":
    main()
