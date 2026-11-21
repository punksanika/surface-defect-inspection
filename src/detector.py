"""
detector.py: defect INSTANCE detection with RetinaNet (ResNet-50-FPN) vs Faster R-CNN (ResNet-50-FPN).

Run:
    python src/detector.py --model retinanet   --out experiments/det_retinanet
    python src/detector.py --model fasterrcnn  --out experiments/det_fasterrcnn

Boxes come from the ground-truth masks (connected components, made in prepare_data.py).
Each box gets the defect type of its image: 1=bent 2=color 3=flip 4=scratch (0 = background).

RetinaNet  (one-stage): dense anchors on every FPN level -> class score + box offsets per anchor.
           Focal loss  FL(p) = -(1-p)^gamma log(p)  down-weights the many easy background anchors.
Faster R-CNN (two-stage): Region Proposal Network proposes boxes -> RoI Align -> classify + refine.

Metric: mAP@0.5, written by hand below (VOC-style, all-point interpolation).
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # run unsupported ops on CPU instead of crashing

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as TF

CLASSES = ["background", "bent", "color", "flip", "scratch"]
SIZE = 512             # images resized 700 -> 512 (boxes scaled the same way)


def get_device(name):
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


# ---------------------------------------------------------------- data
class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, records, train):
        self.recs = [r for r in records if r["label"] in CLASSES[1:] and r["instances"]]
        self.train = train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(r["path"]).convert("RGB")
        W, H = img.size
        s = SIZE / W
        img = img.resize((SIZE, SIZE), Image.BILINEAR)
        boxes = torch.tensor([[b * s for b in inst["bbox"]] for inst in r["instances"]], dtype=torch.float32)
        x = TF.to_tensor(img)                                     # (3,512,512) in [0,1]; model normalizes itself
        if self.train:                                            # rotate by k*90 deg (NO flips: mirror = "flip" defect)
            k = random.randint(0, 3)
            for _ in range(k):
                x = torch.rot90(x, 1, dims=(1, 2))                # 90 deg counter-clockwise
                x1, y1, x2, y2 = boxes.unbind(1)
                boxes = torch.stack([y1, SIZE - x2, y2, SIZE - x1], dim=1)
        labels = torch.full((len(boxes),), CLASSES.index(r["label"]), dtype=torch.int64)
        return x, {"boxes": boxes, "labels": labels}, r["path"]


def collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------- model
def build_model(name):
    kw = dict(min_size=SIZE, max_size=SIZE)
    if name == "retinanet":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        m = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT", **kw)   # COCO-pretrained
        na = m.head.classification_head.num_anchors
        m.head.classification_head = RetinaNetClassificationHead(256, na, len(CLASSES))   # new 5-class head
    elif name == "fasterrcnn":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", **kw)
        inf = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(inf, len(CLASSES))
    else:
        raise ValueError(name)
    return m


# ---------------------------------------------------------------- metrics (by hand)
def box_iou(a, b):
    """IoU between every box in a (N,4) and b (M,4) -> (N,M).  IoU = overlap area / union area."""
    if len(a) == 0 or len(b) == 0:
        return torch.zeros(len(a), len(b))
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def average_precision(preds, gts, cls, iou_thr=0.5):
    """preds: list per image of (boxes, scores, labels); gts: list per image of (boxes, labels).
    1. take every predicted box of this class from every image, sort by score (most confident first)
    2. walk down the list: TP if IoU >= thr with a not-yet-matched true box of this class, else FP
    3. precision = TP/(TP+FP), recall = TP/#true boxes after each step -> PR curve
    4. AP = area under the (monotone) PR curve"""
    n_gt = sum(int((gl == cls).sum()) for _, gl in gts)
    if n_gt == 0:
        return None
    dets = []
    for i, (pb, ps, pl) in enumerate(preds):
        for b, s in zip(pb[pl == cls], ps[pl == cls]):
            dets.append((float(s), i, b))
    dets.sort(key=lambda d: -d[0])
    used = [torch.zeros(int((gl == cls).sum()), dtype=torch.bool) for _, gl in gts]
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for k, (_, i, b) in enumerate(dets):
        g = gts[i][0][gts[i][1] == cls]
        if len(g) == 0:
            fp[k] = 1; continue
        ious = box_iou(b[None], g)[0]
        j = int(ious.argmax())
        if ious[j] >= iou_thr and not used[i][j]:
            tp[k] = 1; used[i][j] = True
        else:
            fp[k] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):                      # make precision monotonically decreasing
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    preds, gts, times = [], [], []
    for xs, ts, _ in loader:
        t0 = time.perf_counter()
        out = model([x.to(device) for x in xs])
        times.append((time.perf_counter() - t0) / len(xs))
        for o, t in zip(out, ts):
            preds.append((o["boxes"].cpu(), o["scores"].cpu(), o["labels"].cpu()))
            gts.append((t["boxes"], t["labels"]))
    aps = {CLASSES[c]: average_precision(preds, gts, c) for c in range(1, len(CLASSES))}
    valid = [v for v in aps.values() if v is not None]
    # class-agnostic AP: "did we box the defect at all?" (ignores the type)
    agn_p = [(b, s, torch.ones_like(l)) for b, s, l in preds]
    agn_g = [(b, torch.ones_like(l)) for b, l in gts]
    return {"mAP50": float(np.mean(valid)), "AP50_per_class": aps,
            "AP50_class_agnostic": average_precision(agn_p, agn_g, 1),
            "ms_per_image": 1000 * float(np.mean(times[1:] if len(times) > 1 else times))}, preds, gts


def save_examples(loader_ds, preds, out_dir, n=8, score_thr=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(loader_ds))):
        x, t, path = loader_ds[i]
        pb, ps, pl = preds[i]
        fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(x.permute(1, 2, 0).numpy()); ax.axis("off")
        for b in t["boxes"]:
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=2))
        for b, s, l in zip(pb, ps, pl):
            if s < score_thr:
                continue
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=2))
            ax.text(b[0], b[1] - 4, f"{CLASSES[l]} {s:.2f}", color="red", fontsize=8, backgroundcolor="white")
        ax.set_title(f"green = truth ({CLASSES[int(t['labels'][0])]}), red = prediction")
        fig.tight_layout(); fig.savefig(out_dir / f"{i:02d}_{Path(path).parent.name}_{Path(path).stem}.png", dpi=100)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--model", default="retinanet", choices=["retinanet", "fasterrcnn"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto | mps | cpu")
    ap.add_argument("--out", default="experiments/det_retinanet")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    device = get_device(args.device); print("Device:", device)
    S = json.loads(Path(args.splits).read_text())["splits"]
    tr, va, te = BoxDataset(S["sup_train"], True), BoxDataset(S["val"], False), BoxDataset(S["test"], False)
    print(f"Images  train {len(tr)}  val {len(va)}  test {len(te)}")
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate,
                                     generator=torch.Generator().manual_seed(0))
    vl = torch.utils.data.DataLoader(va, batch_size=4, collate_fn=collate)
    sl = torch.utils.data.DataLoader(te, batch_size=4, collate_fn=collate)

    model = build_model(args.model).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    best, log, t_start = -1.0, [], time.time()
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for xs, ts, _ in tl:
            xs = [x.to(device) for x in xs]
            ts = [{k: v.to(device) for k, v in t.items()} for t in ts]
            losses = model(xs, ts)                      # dict of losses (classification + box regression ...)
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        m, _, _ = run_eval(model, vl, device)
        log.append({"epoch": ep, "train_loss": tot / n, "val_mAP50": m["mAP50"]})
        print(f"epoch {ep:2d}  train loss {tot / n:.4f}  val mAP@0.5 {m['mAP50']:.3f}")
        if m["mAP50"] > best:
            best = m["mAP50"]; torch.save(model.state_dict(), out / "best.pt")
    train_time = time.time() - t_start

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    m, preds, gts = run_eval(model, sl, device)
    m.update({"model": args.model, "best_val_mAP50": best, "train_time_s": round(train_time, 1),
              "n_test_images": len(te), "n_test_boxes": int(sum(len(g[0]) for g in gts)),
              "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)})
    print(json.dumps(m, indent=2))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    save_examples(te, preds, out / "figures")
    print("Saved results to", out)


if __name__ == "__main__":
    main()