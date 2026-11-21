"""
prepare_data.py: verify the dataset, check duplicates, create leakage-safe splits.

Run:
    python src/prepare_data.py --root data/mvtec/metal_nut --out splits.json

What it does:
  1. Counts images per folder (so we VERIFY the dataset instead of assuming it).
  2. Computes a 256-bit difference-hash (dHash) for every image and reports duplicates.
  3. Creates ONE fixed split file (seed 42) used by every later script:
        train_good : good images only  -> PatchCore memory bank, VAE
        sup_train  : defective images  -> classifier, RetinaNet, transformer
        val        : good + defective  -> choose thresholds
        test       : good + defective  -> final reported numbers ONLY
  4. Converts each ground-truth mask into defect instances (connected components)
     with bounding boxes -> needed for RetinaNet.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

DEFECT_TYPES = ["bent", "color", "flip", "scratch"]


# ---------------------------------------------------------------- 1. counting
def list_images(folder: Path):
    return sorted(p for p in folder.glob("*.png"))


def count_dataset(root: Path):
    counts = {}
    counts["train/good"] = len(list_images(root / "train" / "good"))
    for sub in sorted((root / "test").iterdir()):
        if sub.is_dir():
            counts[f"test/{sub.name}"] = len(list_images(sub))
    gt = root / "ground_truth"
    if gt.exists():
        for sub in sorted(gt.iterdir()):
            if sub.is_dir():
                counts[f"ground_truth/{sub.name}"] = len(list_images(sub))
    return counts


# ---------------------------------------------------------------- 2. dHash
def dhash(path: Path, size: int = 8) -> int:
    """Difference hash: shrink to (size+1) x size grayscale, compare each pixel
    with its right neighbour -> size*size bits. Similar images -> similar bits."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def pixel_diff(a: Path, b: Path) -> float:
    """Mean absolute grayscale difference at 64x64 (0 = identical)."""
    x = np.asarray(Image.open(a).convert("L").resize((64, 64)), dtype=np.float32)
    y = np.asarray(Image.open(b).convert("L").resize((64, 64)), dtype=np.float32)
    return float(np.abs(x - y).mean())


def find_near_duplicates(paths, max_dist=0, confirm_diff=1.0):
    """Stage 1: 256-bit dHash, flag pairs with Hamming distance <= max_dist (cheap).
    Stage 2: CONFIRM each flagged pair by pixel difference, because metal nuts all look
    alike and a hash alone could wrongly flag two different (but similar) parts."""
    hashes = [(p, dhash(p, size=16)) for p in paths]
    pairs = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = hamming(hashes[i][1], hashes[j][1])
            if d <= max_dist and pixel_diff(hashes[i][0], hashes[j][0]) < confirm_diff:
                pairs.append((str(hashes[i][0]), str(hashes[j][0]), d))
    return pairs


# ---------------------------------------------------------------- 4. mask -> instances
def mask_to_instances(mask_path: Path, min_area: int = 30):
    """Label connected regions of the binary mask. Each region = one defect instance.
    Returns list of {bbox:[x1,y1,x2,y2], area} in ORIGINAL image pixel coordinates."""
    m = np.asarray(Image.open(mask_path).convert("L")) > 127
    labeled, n = ndimage.label(m)          # 4-connectivity by default
    instances = []
    for k, sl in enumerate(ndimage.find_objects(labeled), start=1):
        area = int((labeled[sl] == k).sum())
        if area < min_area:                # ignore tiny specks of annotation noise
            continue
        ys, xs = sl
        instances.append({"bbox": [xs.start, ys.start, xs.stop, ys.stop], "area": area})
    return instances


def split_list(items, fracs, rng):
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n1 = round(n * fracs[0])
    n2 = round(n * (fracs[0] + fracs[1]))
    return items[:n1], items[n1:n2], items[n2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/mvtec/metal_nut")
    ap.add_argument("--out", default="splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra_good", type=int, default=30,
                    help="train/good images moved into val/test so FRR is measured on more good parts")
    args = ap.parse_args()
    root = Path(args.root)
    rng = random.Random(args.seed)

    # 1. verify
    counts = count_dataset(root)
    print("== Folder counts ==")
    for k, v in counts.items():
        print(f"  {k:28s} {v}")
    sample = list_images(root / "train" / "good")[0]
    print("Sample image size:", Image.open(sample).size, "mode:", Image.open(sample).mode)

    # 2. duplicates
    all_paths = list_images(root / "train" / "good")
    for d in (root / "test").iterdir():
        if d.is_dir():
            all_paths += list_images(d)
    dups = find_near_duplicates(all_paths, max_dist=0)
    print(f"\n== Duplicate check: {len(all_paths)} images, {len(dups)} identical-hash pairs ==")
    for a, b, d in dups[:10]:
        print("  ", a, "<->", b)
    dup_paths = {b for _, b, _ in dups}   # drop the second image of every pair

    # 3. splits
    train_good = [p for p in list_images(root / "train" / "good") if str(p) not in dup_paths]
    test_good = [p for p in list_images(root / "test" / "good") if str(p) not in dup_paths]
    rng.shuffle(train_good)
    moved = train_good[: args.extra_good]
    train_good = train_good[args.extra_good:]
    pool_good = test_good + moved
    rng.shuffle(pool_good)
    half = len(pool_good) // 2
    val_good, test_good_final = pool_good[:half], pool_good[half:]

    def rec(p, label, mask=None):
        r = {"path": str(p), "label": label, "mask": None, "instances": []}
        if mask is not None:
            r["mask"] = str(mask)
            r["instances"] = mask_to_instances(mask)
        return r

    splits = defaultdict(list)
    splits["train_good"] = [rec(p, "good") for p in train_good]
    splits["val"] += [rec(p, "good") for p in val_good]
    splits["test"] += [rec(p, "good") for p in test_good_final]

    for t in DEFECT_TYPES:
        imgs = [p for p in list_images(root / "test" / t) if str(p) not in dup_paths]
        recs = []
        for p in imgs:
            mask = root / "ground_truth" / t / f"{p.stem}_mask.png"
            if not mask.exists():
                raise FileNotFoundError(f"Missing mask for {p}: expected {mask}")
            recs.append(rec(p, t, mask))
        tr, va, te = split_list(recs, (0.6, 0.2), rng)   # stratified: split inside each type
        splits["sup_train"] += tr
        splits["val"] += va
        splits["test"] += te

    # leakage check: no image path may appear in two splits
    seen = Counter(r["path"] for s in splits.values() for r in s)
    leaks = [p for p, c in seen.items() if c > 1]
    assert not leaks, f"LEAKAGE: {leaks[:5]}"

    print("\n== Split summary ==")
    for name, recs in splits.items():
        c = Counter(r["label"] for r in recs)
        n_inst = sum(len(r["instances"]) for r in recs)
        print(f"  {name:10s} n={len(recs):4d}  {dict(c)}  instances={n_inst}")
    print("Leakage check passed: every image is in exactly one split.")

    out = {"seed": args.seed, "root": str(root), "counts": counts,
           "duplicates_removed": sorted(dup_paths), "splits": splits}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()