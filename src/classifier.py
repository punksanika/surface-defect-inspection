"""
classifier.py: defect-TYPE classifier (bent / color / flip / scratch) + 2D conv visualizations.

Run:
    python src/classifier.py --arch resnet18  --out experiments/cls_resnet18
    python src/classifier.py --arch googlenet --out experiments/cls_googlenet

Why a separate classifier?  PatchCore only says "abnormal + where". It cannot name the defect.
This model is trained (supervised) on labelled DEFECTIVE images only, so it answers "which kind".
In the product: PatchCore decides PASS/REJECT, then this model names the defect of rejected parts.

Data: sup_train (56 images) to train, val defects (18) to pick the best epoch, test defects (19) reported once.
Transfer learning: start from ImageNet weights, replace the last layer with a 4-class layer, fine-tune everything.

Augmentation (train only):
  - random rotation: a nut can lie at any angle, the defect type does not change
  - NO horizontal/vertical flips: a mirror image of a nut looks like the "flip" defect (nut upside down),
    so flipping would silently turn a "bent" example into something that looks "flipped" -> wrong labels
  - NO hue/colour jitter: it would erase exactly what defines the "color" class
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torchvision import transforms as T

CLASSES = ["bent", "color", "flip", "scratch"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DefectDataset(torch.utils.data.Dataset):
    def __init__(self, records, train):
        self.recs = [r for r in records if r["label"] in CLASSES]   # defective images only
        if train:
            self.tf = T.Compose([
                T.Resize((256, 256)),
                T.RandomRotation(180),                       # any angle; background is black anyway
                T.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(1.0, 1.0)),
                T.ColorJitter(brightness=0.1),               # brightness only, no hue change
                T.ToTensor(), T.Normalize(MEAN, STD)])
        else:
            self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        x = self.tf(Image.open(r["path"]).convert("RGB"))
        return x, CLASSES.index(r["label"])


def build_model(arch):
    if arch == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))        # 512 -> 4
    elif arch == "googlenet":
        m = torchvision.models.googlenet(weights="IMAGENET1K_V1")   # auxiliary heads removed by default
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))        # 1024 -> 4
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, times = [], [], []
    for x, y in loader:
        t0 = time.perf_counter()
        logits = model(x.to(device))
        p = logits.argmax(1).cpu()
        times.append((time.perf_counter() - t0) / len(x))
        ys += y.tolist(); ps += p.tolist()
    return np.array(ys), np.array(ps), times


def save_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def save_conv_visuals(model, arch, sample_path, out_dir, device):
    """Req 7b: (1) the first-layer conv filters, (2) feature maps of one image after the first conv block."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conv1 = model.conv1 if arch == "resnet18" else model.conv1.conv
    w = conv1.weight.detach().cpu()                         # (64, 3, 7, 7) for both nets
    w = (w - w.min()) / (w.max() - w.min())
    fig, axes = plt.subplots(8, 8, figsize=(6, 6))
    for k, ax in enumerate(axes.flat):
        ax.imshow(w[k].permute(1, 2, 0).numpy()); ax.axis("off")
    fig.suptitle(f"{arch}: 64 first-layer filters (7x7x3)")
    fig.savefig(out_dir / "conv1_filters.png", dpi=120); plt.close(fig)

    feats = {}
    h = conv1.register_forward_hook(lambda m, i, o: feats.__setitem__("c1", o.detach().cpu()))
    x = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])(
        Image.open(sample_path).convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        model(x)
    h.remove()
    fmap = feats["c1"][0]                                   # (64, 112, 112)
    fig, axes = plt.subplots(4, 8, figsize=(10, 5.5))
    for k, ax in enumerate(axes.flat):
        ax.imshow(fmap[k].numpy(), cmap="viridis"); ax.axis("off")
    fig.suptitle(f"{arch}: first 32 feature maps after conv1 for {Path(sample_path).parent.name}/{Path(sample_path).name}")
    fig.savefig(out_dir / "conv1_feature_maps.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "googlenet"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="experiments/cls_resnet18")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    torch.manual_seed(0); np.random.seed(0)
    device = get_device(); print("Device:", device)
    S = json.loads(Path(args.splits).read_text())["splits"]
    tr, va, te = DefectDataset(S["sup_train"], True), DefectDataset(S["val"], False), DefectDataset(S["test"], False)
    print(f"Defect images  train {len(tr)}  val {len(va)}  test {len(te)}")
    g = torch.Generator().manual_seed(0)
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, generator=g)
    vl = torch.utils.data.DataLoader(va, batch_size=16)
    sl = torch.utils.data.DataLoader(te, batch_size=16)

    model = build_model(args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, log = -1.0, []
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        yv, pv, _ = predict(model, vl, device)
        f1v = f1_score(yv, pv, average="macro")
        log.append({"epoch": ep, "train_loss": tot / n, "val_acc": accuracy_score(yv, pv), "val_macro_f1": f1v})
        print(f"epoch {ep:2d}  train loss {tot / n:.4f}  val acc {log[-1]['val_acc']:.3f}  val macro-F1 {f1v:.3f}")
        if f1v > best_f1:                                   # keep the epoch that is best on VALIDATION
            best_f1 = f1v
            torch.save(model.state_dict(), out / "best.pt")
    train_time = time.time() - t_start

    # ---- TEST: load the best-on-val weights, report once
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    yt, pt, times = predict(model, sl, device)
    cm = confusion_matrix(yt, pt, labels=range(4))
    metrics = {
        "arch": args.arch, "params_millions": round(n_params / 1e6, 2),
        "best_val_macro_f1": float(best_f1),
        "test_accuracy": float(accuracy_score(yt, pt)),
        "test_macro_f1": float(f1_score(yt, pt, average="macro")),
        "test_confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "classes": CLASSES,
        "train_time_s": round(train_time, 1),
        "ms_per_image": round(1000 * float(np.mean(times)), 2),
    }
    print(classification_report(yt, pt, labels=range(4), target_names=CLASSES, zero_division=0))
    print(json.dumps(metrics, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    save_confusion(cm, out / "confusion_matrix.png", f"{args.arch} test (n={len(yt)})")
    sample = next(r["path"] for r in S["test"] if r["label"] == "scratch")
    save_conv_visuals(model, args.arch, sample, out, device)
    print("Saved results to", out)


if __name__ == "__main__":
    main()