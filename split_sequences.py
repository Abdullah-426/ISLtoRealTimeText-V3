#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import shutil
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------- Helpers -----------------


def list_classes(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_clips(class_dir: Path):
    return sorted([p for p in class_dir.iterdir() if p.is_dir() and p.name.startswith("clip_")])


def prep(out_root: Path, splits, classes):
    for sp in splits:
        for c in classes:
            (out_root / sp / c.name).mkdir(parents=True, exist_ok=True)


def hardlink_clip(src_clip: Path, dst_class_dir: Path):
    """Create dst/clip_xxx with hard-links of all files from src/clip_xxx."""
    dst = dst_class_dir / src_clip.name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_clip.iterdir()):
        if p.is_file():
            target = dst / p.name
            try:
                # hardlink (fast & space efficient)
                os.link(str(p), str(target))
            except Exception:
                shutil.copy2(str(p), str(target))  # fallback copy on failure
    return dst


def copy_clip(src_clip: Path, dst_class_dir: Path):
    dst = dst_class_dir / src_clip.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_clip, dst)


def split_counts(n, train_r, val_r, test_r, min_val=1, min_test=1):
    """Compute (n_train, n_val, n_test) with minimal val/test if feasible."""
    n_train = int(round(train_r * n))
    n_val = int(round(val_r * n))
    if n_train + n_val > n:
        n_val = n - n_train
    n_test = n - n_train - n_val

    # Enforce minimum val/test if class has enough samples
    if n >= (min_val + min_test + 1):
        if n_val < min_val:
            take = min_val - n_val
            if n_train >= take:
                n_train -= take
                n_val += take
        if n_test < min_test:
            take = min_test - n_test
            if n_train >= take:
                n_train -= take
                n_test += take

    # Guards
    if n_train < 0:
        n_train = 0
    if n_val < 0:
        n_val = 0
    if n_test < 0:
        n_test = 0
    # Fix rounding mismatch
    s = n_train + n_val + n_test
    if s != n:
        n_train += (n - s)
    return n_train, n_val, n_test

# --------------- Main -----------------


def main():
    ap = argparse.ArgumentParser(
        description="Fast split (workers) cleaned sequences into train/val/test")
    ap.add_argument("--src", type=str, default="RAW DATA_CLEAN",
                    help="Input cleaned root")
    ap.add_argument("--out", type=str, default="Dataset_Split",
                    help="Output split root")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy_mode", type=str, default="link", choices=["copy", "link"],
                    help="Use 'link' to hard-link files (faster, space-saving; same drive). Default=link")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel workers (threads). Default=8")
    ap.add_argument("--min_val", type=int, default=1,
                    help="Min val clips per class if feasible")
    ap.add_argument("--min_test", type=int, default=1,
                    help="Min test clips per class if feasible")
    args = ap.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("[ERROR] train+val+test must sum to 1.0")

    random.seed(args.seed)
    src = Path(args.src)
    out = Path(args.out)

    # Recreate output
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    classes = list_classes(src)
    prep(out, ["train", "val", "test"], classes)

    use_link = (args.copy_mode == "link")
    copier = hardlink_clip if use_link else copy_clip

    # --------- Plan split & build manifest (synchronously; fast) ----------
    manifest = {
        "src": str(src.resolve()),
        "out": str(out.resolve()),
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "workers": args.workers,
        "classes": {},
        "summary": {}
    }
    totals = {"train": 0, "val": 0, "test": 0}

    # Per-class split and schedule tasks
    tasks = []  # (src_clip, dst_dir)
    for cdir in classes:
        cls_name = cdir.name
        clips = list_clips(cdir)
        n = len(clips)
        idxs = list(range(n))
        random.shuffle(idxs)

        n_train, n_val, n_test = split_counts(n, args.train, args.val, args.test,
                                              min_val=args.min_val, min_test=args.min_test)

        part_train = idxs[:n_train]
        part_val = idxs[n_train:n_train+n_val]
        part_test = idxs[n_train+n_val:]

        per = {"train": [], "val": [], "test": []}

        for i in part_train:
            per["train"].append(clips[i].name)
            tasks.append((clips[i], out / "train" / cls_name))
        for i in part_val:
            per["val"].append(clips[i].name)
            tasks.append((clips[i], out / "val" / cls_name))
        for i in part_test:
            per["test"].append(clips[i].name)
            tasks.append((clips[i], out / "test" / cls_name))

        manifest["classes"][cls_name] = {
            "total": n,
            "train": len(per["train"]),
            "val": len(per["val"]),
            "test": len(per["test"]),
            "clips": per
        }
        totals["train"] += len(per["train"])
        totals["val"] += len(per["val"])
        totals["test"] += len(per["test"])
        print(
            f"[SPLIT] {cls_name:20s}  train={len(per['train'])}  val={len(per['val'])}  test={len(per['test'])}")

    manifest["summary"] = {"totals": totals,
                           "grand_total": sum(totals.values())}

    # --------- Execute copying/linking in parallel ----------
    print(
        f"\n[INFO] Dispatching {len(tasks)} clip operations with {args.workers} workers (mode={args.copy_mode})...")
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(copier, src_clip, dst_dir)
                for (src_clip, dst_dir) in tasks]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                errors.append(str(e))

    # --------- Finalize manifest ----------
    with open(out / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n[RESULT] Split done.")
    print(
        f"  train={totals['train']}  val={totals['val']}  test={totals['test']}")
    print(f"  Manifest -> {out/'split_manifest.json'}")
    if errors:
        print(f"[WARN] {len(errors)} errors during copy/link (first 5 shown):")
        for e in errors[:5]:
            print("   -", e)


if __name__ == "__main__":
    main()
