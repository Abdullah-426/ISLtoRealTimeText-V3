import os
import sys
import shutil
from glob import glob

# ------------- Config -------------
SOURCES = [
    "RAW Images Abdullah",
    "RAW Images Devyansh",
    "RAW Images Pranav",
]
DEST_ROOT = "RAW Images"  # final merged folder name

# Classes: 0-9, A-Z, blank
CLASSES = [str(d) for d in range(10)] + [chr(i)
                                         for i in range(65, 91)] + ["blank"]

# If True, DEST_ROOT will be (re)created fresh each run.
# If False and DEST_ROOT exists, images will append after existing ones (still renumbered).
OVERWRITE_DEST = True

# Accept only JPG/JPEG (case-insensitive)
ALLOW_EXT = (".jpg", ".jpeg")
# ----------------------------------


def is_image(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in ALLOW_EXT


def collect_images_for_class(cls_name: str):
    """
    Gather all images across SOURCES for a class folder name.
    - Normalizes to absolute paths
    - De-duplicates paths (handles *.jpg/*.JPG double matches on case-insensitive FS)
    - Stable order: by source, then numeric stem if possible, else by name
    """
    files = []
    for src in SOURCES:
        src_cls_dir = os.path.join(src, cls_name)
        if os.path.isdir(src_cls_dir):
            # Listdir + extension filter avoids duplicate globbing of same file in different cases
            for name in os.listdir(src_cls_dir):
                p = os.path.join(src_cls_dir, name)
                if is_image(p):
                    files.append(os.path.abspath(p))  # normalize

    # De-duplicate while preserving order
    files = list(dict.fromkeys(files))

    # Stable sort helpers
    abs_sources = [os.path.abspath(s) for s in SOURCES]

    def source_index(path: str) -> int:
        # Find which source the file belongs to (by common path)
        for i, s in enumerate(abs_sources):
            try:
                if os.path.commonpath([path, s]) == s:
                    return i
            except ValueError:
                # Different drives on Windows can raise ValueError
                pass
        return 999

    def sort_key(p: str):
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)
        try:
            n = int(stem)
        except ValueError:
            n = float("inf")
        return (source_index(p), n, base.lower())

    files.sort(key=sort_key)
    return files


def prepare_dest():
    """Create or reset destination structure."""
    if OVERWRITE_DEST and os.path.isdir(DEST_ROOT):
        shutil.rmtree(DEST_ROOT)
    os.makedirs(DEST_ROOT, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(DEST_ROOT, c), exist_ok=True)


def renumber_and_copy(files, dest_dir: str) -> int:
    """
    Copy files into dest_dir as 0.jpg, 1.jpg, ... (normalized to .jpg).
    If OVERWRITE_DEST is False and dest already has files, continue numbering.
    """
    existing = [f for f in os.listdir(dest_dir) if f.lower().endswith(".jpg")]
    start_idx = 0
    if not OVERWRITE_DEST and existing:
        nums = []
        for f in existing:
            stem, _ = os.path.splitext(f)
            if stem.isdigit():
                nums.append(int(stem))
        start_idx = (max(nums) + 1) if nums else len(existing)

    count = 0
    for i, src in enumerate(files, start=start_idx):
        dest_path = os.path.join(dest_dir, f"{i}.jpg")  # normalize to .jpg
        shutil.copy2(src, dest_path)
        count += 1
    return count


def main():
    # Validate sources
    any_source = False
    for s in SOURCES:
        if os.path.isdir(s):
            any_source = True
        else:
            print(f"[WARN] Source not found (skipping): {s}")
    if not any_source:
        print("[ERROR] None of the source folders exist. Nothing to merge.")
        sys.exit(1)

    prepare_dest()

    total = 0
    print(f"[INFO] Merging into: {DEST_ROOT}\n")
    for cls in CLASSES:
        files = collect_images_for_class(cls)
        if not files:
            print(f"[CLASS {cls}] 0 images (no sources had this class)")
            continue

        dest_dir = os.path.join(DEST_ROOT, cls)
        copied = renumber_and_copy(files, dest_dir)
        total += copied
        print(f"[CLASS {cls}] {copied} images")

    # Final tally (sanity check)
    final_count = 0
    for cls in CLASSES:
        d = os.path.join(DEST_ROOT, cls)
        if os.path.isdir(d):
            final_count += len([f for f in os.listdir(d)
                               if f.lower().endswith(".jpg")])

    print("\n[RESULT]")
    print(f"Total .jpg files in '{DEST_ROOT}': {final_count}")


if __name__ == "__main__":
    main()
