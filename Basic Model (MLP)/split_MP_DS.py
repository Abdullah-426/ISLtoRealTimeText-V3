#!/usr/bin/env python3
import os
import json
import random
import shutil
import argparse
from pathlib import Path
from math import floor, ceil
from collections import defaultdict

import numpy as np

ALLOW_EXT = ".npy"  # only lowercase to avoid Windows dupes
FEAT_DIM = 126      # 2 hands * 21 * 3
NUM_LM = 21
DIMS = 3


# ---------- Helpers ----------
def list_classes(root: Path):
    if not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def files_in_class(root: Path, cls: str):
    """Return unique .npy files for class (case-insensitive), dedup by resolved path."""
    cls_dir = root / cls
    if not cls_dir.is_dir():
        return []
    files = [str(p.resolve()) for p in cls_dir.iterdir()
             if p.is_file() and p.suffix.lower() == ALLOW_EXT]
    files = list(dict.fromkeys(files))  # remove accidental duplicates
    return sorted(files)


def load_and_tag(path: str):
    """
    Load a .npy and return a tag among:
      - 'both': both hands present (neither block is all zeros)
      - 'one' : exactly one hand present (one block all zeros)
      - 'zero': both blocks zero (no hands detected)
      - 'bad' : invalid shape or load error
    """
    try:
        v = np.load(path)
    except Exception:
        return "bad"

    v = v.reshape(-1)
    if v.shape[0] != FEAT_DIM:
        return "bad"
    arr = v.reshape(2, NUM_LM, DIMS)
    left_zero = np.allclose(arr[0], 0.0)
    right_zero = np.allclose(arr[1], 0.0)

    if not left_zero and not right_zero:
        return "both"
    if left_zero and right_zero:
        return "zero"
    return "one"


def decide_two_hand(both_count, valid_count, min_ratio, min_abs):
    """
    Decide class is two-hand if both_count >= max(min_abs, ceil(min_ratio * valid_count)).
    """
    need = max(min_abs, ceil(min_ratio * valid_count))
    return both_count >= need


def prep_dirs(out_root: Path, classes, splits_keys):
    for split in splits_keys:
        for cls in classes:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)


def per_class_split(paths, train_r, val_r, test_r):
    """Shuffle + split array paths by ratios. If val_r=0 -> only train/test."""
    random.shuffle(paths)
    n = len(paths)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    n_train = floor(train_r * n)
    n_test = floor(test_r * n)
    n_val = n - n_train - n_test

    if n_val < 0:
        n_val = 0
        n_test = n - n_train

    if val_r == 0.0:
        n_val = 0
        n_test = n - n_train

    train_arr = paths[:n_train]
    val_arr = paths[n_train:n_train + n_val] if n_val > 0 else []
    test_arr = paths[n_train + n_val:]
    return {"train": train_arr, "val": val_arr, "test": test_arr}
# --------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Split MP_Dataset into train/(val)/test with hand-consistency filtering (keep zero-hand only for 'blank')."
    )
    ap.add_argument("--source", type=str, default="MP_Dataset",
                    help="Folder with class subfolders containing .npy files.")
    ap.add_argument("--out", type=str, default="MP_Dataset_Split",
                    help="Output folder for train/(val)/test.")
    ap.add_argument("--train", type=float, default=0.8,
                    help="Train ratio (default 0.8)")
    ap.add_argument("--val",   type=float, default=0.0,
                    help="Val ratio (set 0 to disable; default 0.0)")
    ap.add_argument("--test",  type=float, default=0.2,
                    help="Test ratio (default 0.2)")
    ap.add_argument("--seed",  type=int, default=42,
                    help="Random seed (default 42)")

    # 2-hand detection controls
    ap.add_argument("--twohand_min_ratio", type=float, default=0.25,
                    help="Min fraction of both-hand files in a class to auto-tag it as two_hand (default 0.25).")
    ap.add_argument("--twohand_min_abs", type=int, default=10,
                    help="Min absolute count of both-hand files to auto-tag as two_hand (default 10).")
    ap.add_argument("--min_keep_per_class", type=int, default=8,
                    help="Warn if, after filtering, a class has fewer than this number of files (default 8).")

    args = ap.parse_args()

    # Validate ratios
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 1e-6:
        raise SystemExit(
            f"[ERROR] Split ratios must sum to 1.0 (got {total_ratio}).")

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_root = Path(args.source)
    out_root = Path(args.out)

    if not src_root.is_dir():
        raise SystemExit(
            f"[ERROR] Missing source folder: {src_root.resolve()}")

    classes = list_classes(src_root)
    if not classes:
        raise SystemExit(
            f"[ERROR] No class folders found in {src_root.resolve()}")

    # Reset output
    if out_root.exists():
        print(f"[WARN] Removing existing {out_root.resolve()}")
        shutil.rmtree(out_root)

    # Which splits to create?
    splits_keys = ["train", "test"] if args.val == 0.0 else [
        "train", "val", "test"]
    prep_dirs(out_root, classes, splits_keys)

    manifest = {
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "twohand_min_ratio": args.twohand_min_ratio,
        "twohand_min_abs": args.twohand_min_abs,
        "classes": {},
        "summary": {}
    }

    totals = {"train": 0, "val": 0, "test": 0}
    grand_total = 0

    for cls in classes:
        paths = files_in_class(src_root, cls)
        if not paths:
            print(f"[SKIP] {cls}: 0 files")
            continue

        # Tag each file
        tags = {"both": [], "one": [], "zero": [], "bad": []}
        for p in paths:
            t = load_and_tag(p)
            tags[t].append(p)

        valid_count = len(tags["both"]) + len(tags["one"])
        both_count = len(tags["both"])

        # Special case: keep ZERO-hand only for the 'blank' class
        if cls.lower() == "blank":
            keep = tags["zero"].copy()
            cls_type = "blank_zero_hand"
            # Optional: you could also include some one-hand noise here if desired (we don't).
        else:
            # auto decide two-hand
            class_is_two = decide_two_hand(
                both_count, valid_count, args.twohand_min_ratio, args.twohand_min_abs
            )
            if class_is_two:
                keep = tags["both"]
                cls_type = "two_hand"
            else:
                keep = tags["one"]
                cls_type = "one_hand"

        print(
            f"[CHECK] {cls}: total={len(paths)} | both={len(tags['both'])} | one={len(tags['one'])} | zero={len(tags['zero'])} | bad={len(tags['bad'])} -> type={cls_type}, keep={len(keep)}"
        )
        if len(keep) < args.min_keep_per_class and cls.lower() != "blank":
            print(
                f"  [WARN] {cls}: only {len(keep)} usable files after filtering ({cls_type}).")

        # Split
        split_dict = per_class_split(keep, args.train, args.val, args.test)

        # Copy
        cls_counts = {"train": 0, "val": 0, "test": 0}
        for split, arr in split_dict.items():
            if args.val == 0.0 and split == "val":
                continue
            dst_dir = out_root / split / cls
            for p in arr:
                src = Path(p)
                shutil.copy2(src, dst_dir / src.name)
            cls_counts[split] = len(arr)
            totals[split] += len(arr)
            grand_total += len(arr)

        if args.val == 0.0:
            print(
                f"[CLASS {cls}] type={cls_type} | train={cls_counts['train']}  test={cls_counts['test']}")
        else:
            print(
                f"[CLASS {cls}] type={cls_type} | train={cls_counts['train']}  val={cls_counts['val']}  test={cls_counts['test']}")

        manifest["classes"][cls] = {
            "type": cls_type,
            "counts_before": {
                "total": len(paths),
                "both": len(tags["both"]),
                "one": len(tags["one"]),
                "zero": len(tags["zero"]),
                "bad": len(tags["bad"]),
            },
            "counts_after_kept": {
                "train": cls_counts["train"],
                "val": cls_counts["val"],
                "test": cls_counts["test"],
            }
        }

    # Summary
    print("\n[SUMMARY]")
    if args.val == 0.0:
        print(
            f"train={totals['train']}  test={totals['test']}  total={grand_total}")
    else:
        print(
            f"train={totals['train']}  val={totals['val']}  test={totals['test']}  total={grand_total}")

    manifest["summary"] = {"totals": totals, "grand_total": grand_total}

    # Save manifest
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] Split written to: {out_root.resolve()}")
    print("    Manifest: split_manifest.json")


if __name__ == "__main__":
    main()
