#!/usr/bin/env python3
import os
import sys
import shutil
from glob import glob

# ------------- Config -------------
SOURCE = "RAW Images Aryan"   # new folder to merge from
DEST_ROOT = "RAW Images"      # existing merged folder (destination)

# Classes: 0-9, A-Z, blank (destination must have these; source may miss some like 'blank')
CLASSES = [str(d) for d in range(10)] + [chr(i)
                                         for i in range(65, 91)] + ["blank"]

# Accept only JPG/JPEG (case-insensitive)
ALLOW_EXT = (".jpg", ".jpeg")
# ----------------------------------


def is_image(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in ALLOW_EXT


def count_images_in_dir(d: str) -> int:
    if not os.path.isdir(d):
        return 0
    cnt = 0
    for f in os.listdir(d):
        if f.lower().endswith(ALLOW_EXT):
            cnt += 1
    return cnt


def list_images(dir_path: str):
    """Return absolute paths to images in a folder (stable sorted by name)."""
    if not os.path.isdir(dir_path):
        return []
    files = []
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if is_image(p):
            files.append(os.path.abspath(p))
    files = list(dict.fromkeys(files))  # dedupe if any
    files.sort(key=lambda x: os.path.basename(x).lower())
    return files


def ensure_dest_structure():
    """Ensure destination has all class folders; create missing ones."""
    if not os.path.isdir(DEST_ROOT):
        raise SystemExit(
            f"[ERROR] Destination not found: {os.path.abspath(DEST_ROOT)}")
    for c in CLASSES:
        os.makedirs(os.path.join(DEST_ROOT, c), exist_ok=True)


def next_index_in_class(dest_dir: str) -> int:
    """
    Find the next numeric index for filenames in dest_dir.
    Accepts any existing .jpg/.jpeg; uses the max(stem) if numeric, else counts.
    """
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        return 0
    nums = []
    count_other = 0
    for f in os.listdir(dest_dir):
        if f.lower().endswith(ALLOW_EXT):
            stem, _ = os.path.splitext(f)
            if stem.isdigit():
                nums.append(int(stem))
            else:
                count_other += 1
    if nums:
        return max(nums) + 1
    # If no numeric stems exist, append after existing count
    return count_other


def merge_class(cls_name: str) -> tuple[int, int, int]:
    """
    Merge images for one class from SOURCE into DEST_ROOT/cls_name.
    Returns (before_count, added_count, after_count).
    """
    dest_dir = os.path.join(DEST_ROOT, cls_name)
    os.makedirs(dest_dir, exist_ok=True)

    before = count_images_in_dir(dest_dir)

    # If source class dir missing, nothing to add (e.g., Aryan has no 'blank')
    src_dir = os.path.join(SOURCE, cls_name)
    if not os.path.isdir(src_dir):
        return (before, 0, before)

    src_files = list_images(src_dir)
    if not src_files:
        return (before, 0, before)

    # Copy with continued numbering, normalize to .jpg
    idx = next_index_in_class(dest_dir)
    added = 0
    for p in src_files:
        dest_path = os.path.join(dest_dir, f"{idx}.jpg")
        shutil.copy2(p, dest_path)
        idx += 1
        added += 1

    after = count_images_in_dir(dest_dir)
    return (before, added, after)


def total_images_in_tree(root: str) -> int:
    total = 0
    for c in CLASSES:
        d = os.path.join(root, c)
        total += count_images_in_dir(d)
    return total


def total_images_in_source(source_root: str) -> int:
    """Count all allowed images in the source (only under known class folders)."""
    total = 0
    for c in CLASSES:
        d = os.path.join(source_root, c)
        total += count_images_in_dir(d)
    return total


def main():
    # Validate paths
    if not os.path.isdir(SOURCE):
        print(f"[ERROR] Source not found: {os.path.abspath(SOURCE)}")
        sys.exit(1)
    if not os.path.isdir(DEST_ROOT):
        print(f"[ERROR] Destination not found: {os.path.abspath(DEST_ROOT)}")
        sys.exit(1)

    ensure_dest_structure()

    # Totals before
    dest_total_before = total_images_in_tree(DEST_ROOT)
    src_total = total_images_in_source(SOURCE)

    print(f"[INFO] Source:      {os.path.abspath(SOURCE)}")
    print(f"[INFO] Destination: {os.path.abspath(DEST_ROOT)}")
    print(f"[INFO] Total images in source ('{SOURCE}'): {src_total}")
    print(f"[INFO] Destination total BEFORE: {dest_total_before}\n")

    # Per-class merge
    grand_added = 0
    for cls in CLASSES:
        before, added, after = merge_class(cls)
        grand_added += added
        print(
            f"[CLASS {cls}] before={before:5d}  +added={added:5d}  after={after:5d}")

    # Totals after
    dest_total_after = total_images_in_tree(DEST_ROOT)

    print("\n[RESULT]")
    print(f"Total images in source ('{SOURCE}'): {src_total}")
    print(f"Destination total BEFORE: {dest_total_before}")
    print(f"Destination total ADDED:  {grand_added}")
    print(f"Destination total AFTER:  {dest_total_after}")

    # Sanity
    if dest_total_before + grand_added != dest_total_after:
        print(
            "[WARN] Totals mismatch – check for non-allowed extensions or unexpected files.")


if __name__ == "__main__":
    main()
