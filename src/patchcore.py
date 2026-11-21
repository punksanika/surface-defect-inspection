"""
patchcore.py: PatchCore anomaly detection (learns from GOOD images only).

Run:
    python src/patchcore.py --splits splits.json --backbone resnet18 --out experiments/patchcore_r18

Pipeline (shapes for resnet18, 224x224 input):
  image (B,3,224,224)
    -> layer2 features (B,128,28,28)   mid-level: edges, textures
    -> layer3 features (B,256,14,14)   higher-level: parts  -> upsampled to 28x28
    -> 3x3 average pooling on each     (each patch also "sees" its neighbours)
    -> concat                          (B,384,28,28)
    -> reshape                         (B*784, 384)   one vector per image patch
  Memory bank  = coreset subsample of all patch vectors from good training images.
  Patch score  = distance to nearest memory-bank vector (kNN, k=1).
  Image score  = max patch score.          Anomaly map = 28x28 scores -> upsample -> blur.

Noise-robust variant (optional, added for the robustness fix):
    python src/patchcore.py --blur_sigma 1.0 --noisy_copies 1 --aug_noise 0.03 --out experiments/patchcore_r18_robust
  --blur_sigma   : light Gaussian blur on the input image, built INTO the feature extractor
                   (so it is also inside the ONNX file and the Java app gets it automatically).
  --noisy_copies : for every good training image, also put N noisy copies into the memory bank,
                   so "good nut + sensor noise" is remembered as normal.
  Defaults (0, 0) give exactly the original model.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import functional as TF

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet statistics: the backbone was trained with these
STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------- data
def load_image(path, noise_sigma=0.0, generator=None):
    """Load, resize to 224, (optionally add noise), normalise.
    noise_sigma > 0 adds Gaussian noise to pixel values in [0,1] and clips back to [0,1].
    This is EXACTLY what the Java robustness test does (Inspector.perturb), so both agree."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t = TF.to_tensor(img)                       # (3,H,W) in [0,1]
    if noise_sigma > 0:
        t = (t + noise_sigma * torch.randn(t.shape, generator=generator)).clamp(0, 1)
    return TF.normalize(t, MEAN, STD)


def load_mask(path):
    if path is None:
        return torch.zeros(IMG_SIZE, IMG_SIZE)
    m = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return (torch.from_numpy(np.asarray(m)) > 127).float()


# ---------------------------------------------------------------- model
class GaussianBlur(torch.nn.Module):
    """Fixed (not learned) Gaussian blur, applied to each colour channel separately.
    Why it helps against noise: pixel noise is random from pixel to pixel (high frequency);
    averaging each pixel with its neighbours cancels much of it, while the nut's shape
    and larger defects (bends, flips, colour patches) survive.
    Blur is linear and the kernel sums to 1, so blurring the normalised image is the same as
    blurring the raw image and then normalising."""
    def __init__(self, sigma):
        super().__init__()
        radius = max(1, int(round(3 * sigma)))           # cover +-3 sigma
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k1 = torch.exp(-x ** 2 / (2 * sigma ** 2)); k1 = k1 / k1.sum()
        k2 = torch.outer(k1, k1)                          # 2-D kernel = outer product of 1-D
        self.register_buffer("kernel", k2.expand(3, 1, -1, -1).clone())  # (3,1,k,k): one per channel
        self.radius = radius

    def forward(self, x):                                 # x: (B,3,H,W)
        x = F.pad(x, [self.radius] * 4, mode="reflect")   # mirror the border, no dark frame
        return F.conv2d(x, self.kernel, groups=3)         # groups=3 -> channels blurred separately


class PatchFeatureExtractor(torch.nn.Module):
    def __init__(self, backbone="resnet18", blur_sigma=0.0):
        super().__init__()
        # blur_sigma = 0 -> no blur (original model)
        self.blur = GaussianBlur(blur_sigma) if blur_sigma > 0 else torch.nn.Identity()
        if backbone == "resnet18":
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        elif backbone == "wide_resnet50_2":
            net = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")
        else:
            raise ValueError(backbone)
        self.body = create_feature_extractor(net, return_nodes={"layer2": "l2", "layer3": "l3"})
        self.eval()

    @torch.no_grad()
    def forward(self, x):                       # x: (B,3,224,224)
        f = self.body(self.blur(x))
        l2, l3 = f["l2"], f["l3"]               # (B,C2,28,28), (B,C3,14,14)
        l2 = F.avg_pool2d(l2, 3, stride=1, padding=1)
        l3 = F.avg_pool2d(l3, 3, stride=1, padding=1)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([l2, l3], dim=1)       # (B,C2+C3,28,28)
        return feat


def to_patches(feat):
    """(B,C,H,W) -> (B*H*W, C)"""
    B, C, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(-1, C)


def greedy_coreset(feats, ratio, proj_dim=128, seed=0):
    """Pick a small subset of vectors that 'covers' all others (k-center greedy).
    Each step adds the point FARTHEST from the current subset.
    Random projection to proj_dim makes distance computation cheap
    (Johnson-Lindenstrauss: distances are approximately preserved)."""
    N, C = feats.shape
    n_select = max(1, int(N * ratio))
    g = torch.Generator().manual_seed(seed)
    proj = (torch.randn(C, proj_dim, generator=g) / proj_dim ** 0.5).to(feats.device)
    z = feats @ proj                                         # (N, proj_dim)
    idx = torch.randint(N, (1,), generator=g).to(feats.device)
    selected = [idx]
    min_d = torch.linalg.norm(z - z[idx], dim=1)             # distance to subset, (N,)
    for _ in range(n_select - 1):
        idx = torch.argmax(min_d).unsqueeze(0)
        selected.append(idx)
        min_d = torch.minimum(min_d, torch.linalg.norm(z - z[idx], dim=1))
    return torch.cat(selected)


@torch.no_grad()
def build_memory_bank(extractor, records, device, coreset_ratio=0.1, noisy_copies=0, aug_noise=0.03, seed=0,
                      coreset_cpu=False):
    """Memory bank = coreset of patch features from GOOD training images.
    noisy_copies > 0: each image is also added `noisy_copies` times with Gaussian noise (sigma=aug_noise),
    so the bank also contains 'good + noise' patches.
    The coreset keeps the SAME number of vectors as the original model (ratio / (1+copies)), so the
    bank size and the inference speed do not change -> a fair comparison.
    coreset_cpu=True runs the greedy coreset on CPU (measured ~2x faster than MPS on this Mac,
    because the loop is thousands of tiny steps)."""
    g = torch.Generator().manual_seed(seed)       # fixed seed -> same noisy copies every run
    feats = []
    for r in records:
        feats.append(to_patches(extractor(load_image(r["path"]).unsqueeze(0).to(device))))
        for _ in range(noisy_copies):
            x = load_image(r["path"], noise_sigma=aug_noise, generator=g)
            feats.append(to_patches(extractor(x.unsqueeze(0).to(device))))
    feats = torch.cat(feats)                       # (N_img*(1+copies)*784, C)
    if coreset_cpu:
        feats = feats.cpu()
    idx = greedy_coreset(feats, coreset_ratio / (1 + noisy_copies))
    return feats[idx].contiguous().to(device), tuple(feats.shape)


@torch.no_grad()
def score_image(extractor, bank, x, device):
    """Returns image score (float) and anomaly map (224,224) on CPU."""
    feat = extractor(x.unsqueeze(0).to(device))             # (1,C,28,28)
    H, W = feat.shape[-2:]
    patches = to_patches(feat)                               # (784,C)
    d = torch.cdist(patches, bank)                           # (784, M)
    patch_scores = d.min(dim=1).values.reshape(1, 1, H, W)   # nearest neighbour distance
    amap = F.interpolate(patch_scores, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                         align_corners=False)
    amap = TF.gaussian_blur(amap.cpu(), kernel_size=33, sigma=4.0)[0, 0]
    return float(patch_scores.max()), amap


# ---------------------------------------------------------------- metrics
def choose_threshold(scores, labels):
    """Pick the threshold that maximises F1 on the VALIDATION set."""
    best = (-1, None)
    for t in np.unique(scores):
        pred = scores >= t
        tp = np.sum(pred & (labels == 1)); fp = np.sum(pred & (labels == 0))
        fn = np.sum(~pred & (labels == 1))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def binary_report(scores, labels, thr):
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"threshold": thr, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "recall_catch_rate": float(rec), "escape_rate": float(1 - rec),
            "false_reject_rate": float(fp / max(fp + tn, 1)), "precision": float(prec),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-9))}


def evaluate(extractor, bank, records, device, noise_sigma=0.0, seed=0):
    """noise_sigma > 0 -> evaluate on noisy copies of the images (robustness test)."""
    g = torch.Generator().manual_seed(seed)
    scores, labels, maps, masks, times = [], [], [], [], []
    for r in records:
        x = load_image(r["path"], noise_sigma=noise_sigma, generator=g)
        t0 = time.perf_counter()
        s, amap = score_image(extractor, bank, x, device)
        times.append(time.perf_counter() - t0)
        scores.append(s); labels.append(0 if r["label"] == "good" else 1)
        maps.append(amap); masks.append(load_mask(r["mask"]))
    return np.array(scores), np.array(labels), torch.stack(maps), torch.stack(masks), times


def pixel_iou(maps, masks, thr):
    """Mean IoU over DEFECTIVE images between thresholded map and ground-truth mask."""
    ious = []
    for m, g in zip(maps, masks):
        if g.sum() == 0:
            continue
        p = m >= thr
        inter = (p & (g > 0)).sum().item(); union = (p | (g > 0)).sum().item()
        ious.append(inter / max(union, 1))
    return float(np.mean(ious)) if ious else float("nan")


def save_figures(records, scores, maps, thr, out_dir, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    vmax = float(maps.max())
    for i, r in enumerate(records[:n] + records[-n:]):
        idx = records.index(r)
        img = Image.open(r["path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(img); ax[0].set_title(f"{r['label']}")
        ax[1].imshow(load_mask(r["mask"]), cmap="gray"); ax[1].set_title("ground truth")
        ax[2].imshow(img); ax[2].imshow(maps[idx], cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        verdict = "REJECT" if scores[idx] >= thr else "PASS"
        ax[2].set_title(f"score {scores[idx]:.2f} -> {verdict}")
        for a in ax: a.axis("off")
        fig.tight_layout(); fig.savefig(out_dir / f"{i:02d}_{Path(r['path']).stem}_{r['label']}.png", dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--coreset_ratio", type=float, default=0.1)
    ap.add_argument("--out", default="experiments/patchcore_r18")
    ap.add_argument("--blur_sigma", type=float, default=0.0, help="input Gaussian blur (0 = off)")
    ap.add_argument("--noisy_copies", type=int, default=0, help="noisy copies per good image in the bank")
    ap.add_argument("--aug_noise", type=float, default=0.03, help="noise sigma of those copies")
    ap.add_argument("--coreset_cpu", action="store_true", help="run the coreset on CPU (faster on Mac)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    device = get_device(); print("Device:", device)
    torch.manual_seed(0)
    S = json.loads(Path(args.splits).read_text())["splits"]
    extractor = PatchFeatureExtractor(args.backbone, args.blur_sigma).to(device)

    # --- 1. build memory bank from GOOD training images only
    t0 = time.time()
    bank, all_shape = build_memory_bank(extractor, S["train_good"], device, args.coreset_ratio,
                                        args.noisy_copies, args.aug_noise, coreset_cpu=args.coreset_cpu)
    print(f"All patch features: {all_shape}")
    print(f"Memory bank after coreset: {tuple(bank.shape)}  ({time.time()-t0:.1f}s)")
    # blur_sigma is saved with the bank so every loader rebuilds the SAME extractor
    torch.save({"bank": bank.cpu(), "backbone": args.backbone, "blur_sigma": args.blur_sigma},
               out / "memory_bank.pt")

    # --- 2. VALIDATION: choose image threshold and pixel threshold
    vs, vl, vmaps, vmasks, _ = evaluate(extractor, bank, S["val"], device)
    img_thr = choose_threshold(vs, vl)
    cand = torch.quantile(vmaps.flatten()[::97], torch.linspace(0.80, 0.995, 40))
    pix_thr = float(max(cand, key=lambda t: pixel_iou(vmaps, vmasks, float(t))))

    # --- 3. TEST: report only, never tune here
    ts, tl, tmaps, tmasks, times = evaluate(extractor, bank, S["test"], device)
    metrics = {
        "val_image_auroc": float(roc_auc_score(vl, vs)),
        "test_image_auroc": float(roc_auc_score(tl, ts)),
        "test_pixel_auroc": float(roc_auc_score(tmasks.flatten().numpy().astype(int),
                                                tmaps.flatten().numpy())),
        "test_mean_iou_defects": pixel_iou(tmaps, tmasks, pix_thr),
        "pixel_threshold": pix_thr,
        "test_at_val_threshold": binary_report(ts, tl, img_thr),
        "ms_per_image": 1000 * float(np.mean(times[1:])),   # skip first (warm-up)
        "fps": 1.0 / float(np.mean(times[1:])),
        "n_test_good": int((tl == 0).sum()), "n_test_defect": int((tl == 1).sum()),
    }
    # per-defect-type catch rate
    per_type = {}
    for t in ["bent", "color", "flip", "scratch"]:
        m = np.array([r["label"] == t for r in S["test"]])
        per_type[t] = f"{int((ts[m] >= img_thr).sum())}/{int(m.sum())} caught"
    metrics["per_type"] = per_type

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    json.dump({"image_threshold": img_thr, "pixel_threshold": pix_thr},
              open(out / "thresholds.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    save_figures(S["test"], ts, tmaps, img_thr, out / "figures")
    print("Saved results to", out)


if __name__ == "__main__":
    main()