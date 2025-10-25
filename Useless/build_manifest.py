#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
import re

clip_pat = re.compile(r"^clip_(\d{3})$")


def list_classes(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_clips(class_dir: Path):
    return sorted([p for p in class_dir.iterdir() if p.is_dir() and clip_pat.match(p.name)])


def has_seq(clip_dir: Path):
    return (clip_dir / "sequence.npy").is_file()


def count_frames(clip_dir: Path):
    return len([p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() == ".npy" and p.name.startswith("f_")])


def main():
    ap = argparse.ArgumentParser(
        description="Build simple manifest for a cleaned dataset.")
    ap.add_argument("--root", type=str,
                    default="RAW DATA_CLEAN", help="Dataset root")
    ap.add_argument("--frames", type=int, default=48,
                    help="Expected frames per clip")
    args = ap.parse_args()

    root = Path(args.root)
    classes = list_classes(root)
    overall = {"root": str(
        root.resolve()), "frames_expected": args.frames, "classes": {}, "totals": {}}
    total_clips = 0
    bad_shape = 0
    missing_seq = 0

    for cdir in classes:
        cname = cdir.name
        clips = list_clips(cdir)
        c_count = 0
        c_bad_shape = 0
        c_missing_seq = 0
        c_report = []
        for clip in clips:
            nf = count_frames(clip)
            seq = has_seq(clip)
            ok = (nf == args.frames) and seq
            c_report.append({
                "clip": clip.name,
                "frames": nf,
                "has_sequence": seq,
                "ok": ok
            })
            if ok:
                c_count += 1
            else:
                if not seq:
                    c_missing_seq += 1
                if nf != args.frames:
                    c_bad_shape += 1

        overall["classes"][cname] = {
            "clips_total": len(clips),
            "clips_ok": c_count,
            "clips_bad_shape": c_bad_shape,
            "clips_missing_seq": c_missing_seq,
            "clips": c_report
        }
        total_clips += len(clips)
        bad_shape += c_bad_shape
        missing_seq += c_missing_seq

    overall["totals"] = {
        "classes": len(classes),
        "clips_total": total_clips,
        "clips_ok": sum(overall["classes"][n]["clips_ok"] for n in overall["classes"]),
        "clips_bad_shape": bad_shape,
        "clips_missing_seq": missing_seq
    }

    out = root / "final_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(f"[OK] Manifest -> {out}")


if __name__ == "__main__":
    main()
