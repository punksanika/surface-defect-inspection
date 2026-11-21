"""
report_transformer.py: a small Transformer that reads the most anomalous patches found by PatchCore and
writes a structured defect report (type + confidence + location + affected area), with attention maps.

Run:
    python src/report_transformer.py

Idea:
  PatchCore gives 784 patch vectors (384-d) and an anomaly score per patch.
  We keep the K = 32 most anomalous patches  ->  a SEQUENCE of 32 tokens.
  token = Linear([feature(384), score(1)]) + Linear([row, col])        (content + position embedding)
  [CLS] + 32 tokens -> 2 Transformer encoder blocks -> CLS output -> 4 defect types.
  Self-attention:  Attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V
  The CLS row of the last attention matrix tells us WHICH patches the model looked at -> attention map.

Data: sup_train defects (train), val defects (pick best epoch), test defects (report once).
Outputs (experiments/report_transformer/): metrics.json, reports.txt, attention.png, train_curve.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patchcore as pc  # noqa: E402

CLASSES = ["bent", "color", "flip", "scratch"]
K, D, HEADS, LAYERS = 32, 64, 4, 2
OUT = Path("experiments/report_transformer")


class Block(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))

    def forward(self, x):
        h = self.n1(x)
        a, w = self.att(h, h, h, need_weights=True, average_attn_weights=True)   # w: (B, T, T)
        x = x + a                                                               # residual
        return x + self.ff(self.n2(x)), w


class ReportEncoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.inp = nn.Linear(feat_dim + 1, D)
        self.pos = nn.Linear(2, D)
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.blocks = nn.ModuleList([Block(D, HEADS) for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, len(CLASSES))

    def forward(self, feats, scores, coords):          # (B,K,C), (B,K,1), (B,K,2)
        x = self.inp(torch.cat([feats, scores], -1)) + self.pos(coords)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)       # (B, K+1, D)
        for b in self.blocks:
            x, w = b(x)
        return self.head(self.norm(x[:, 0])), w                        # logits, last attention


@torch.no_grad()
def tokens_for(records, ext, bank, dev):
    """For every defect image: top-K anomalous patches -> features, scores, (row, col)."""
    F_, S_, C_, extra = [], [], [], []
    for r in records:
        f = ext(pc.load_image(r["path"]).unsqueeze(0).to(dev))       # (1,C,28,28)
        H, W = f.shape[-2:]
        p = pc.to_patches(f)                                          # (784,C)
        s = torch.cdist(p, bank).min(1).values                        # (784,)
        idx = s.topk(K).indices
        rows, cols = (idx // W).float(), (idx % W).float()
        F_.append(p[idx].cpu()); S_.append(s[idx].cpu().unsqueeze(1))
        C_.append(torch.stack([rows / (H - 1), cols / (W - 1)], 1).cpu())
        extra.append({"rows": rows.cpu(), "cols": cols.cpu(), "max_score": float(s.max()),
                      "frac_above": float((s >= THR).float().mean())})
    return torch.stack(F_), torch.stack(S_), torch.stack(C_), extra


def location_words(r, c):
    v = "upper" if r < 9.3 else ("lower" if r > 18.6 else "middle")
    h = "left" if c < 9.3 else ("right" if c > 18.6 else "centre")
    return f"{v}-{h}"


def main():
    global THR
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    dev = pc.get_device(); print("Device:", dev)
    ck = torch.load("experiments/patchcore_r18/memory_bank.pt", map_location="cpu")
    THR = json.loads(Path("experiments/patchcore_r18/thresholds.json").read_text())["image_threshold"]
    ext = pc.PatchFeatureExtractor(ck["backbone"]).to(dev)
    bank = ck["bank"].to(dev)
    S = json.loads(Path("splits.json").read_text())["splits"]
    sets = {k: [r for r in S[k] if r["label"] in CLASSES] for k in ["sup_train", "val", "test"]}
    data = {k: tokens_for(v, ext, bank, dev) for k, v in sets.items()}
    ys = {k: torch.tensor([CLASSES.index(r["label"]) for r in v]) for k, v in sets.items()}
    print("Token sequences:", {k: tuple(v[0].shape) for k, v in data.items()}, "(images, K tokens, feature dim)")

    model = ReportEncoder(data["sup_train"][0].shape[2]).to(dev)
    print("Transformer parameters:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss()
    to = lambda t: [x.to(dev) for x in t[:3]]                         # noqa: E731
    tr, va, te = to(data["sup_train"]), to(data["val"]), to(data["test"])
    best, curve = -1, []
    for ep in range(200):
        model.train()
        logits, _ = model(*tr)
        loss = lossf(logits, ys["sup_train"].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(*va)[0].argmax(1).cpu()
        f1 = f1_score(ys["val"], pv, average="macro")
        curve.append((loss.item(), f1))
        if f1 > best:
            best = f1; torch.save(model.state_dict(), OUT / "best.pt")
        if ep % 20 == 0:
            print(f"epoch {ep:3d}  train loss {loss.item():.4f}  val macro-F1 {f1:.3f}")

    model.load_state_dict(torch.load(OUT / "best.pt", map_location=dev)); model.eval()
    with torch.no_grad():
        logits, att = model(*te)
    prob = torch.softmax(logits, 1).cpu(); pred = prob.argmax(1)
    yt = ys["test"]
    m = {"test_accuracy": float(accuracy_score(yt, pred)), "test_macro_f1": float(f1_score(yt, pred, average="macro")),
         "confusion_rows_true_cols_pred": confusion_matrix(yt, pred, labels=range(4)).tolist(),
         "best_val_macro_f1": float(best), "K_tokens": K, "d_model": D, "heads": HEADS, "layers": LAYERS}
    print(json.dumps(m, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2))

    # --- structured reports
    att_cls = att[:, 0, 1:].cpu()                                     # CLS -> each of the K patch tokens
    lines = []
    for i, r in enumerate(sets["test"]):
        ex = data["test"][3][i]
        top = int(att_cls[i].argmax())
        rr, cc = float(ex["rows"][top]), float(ex["cols"][top])
        lines.append(
            f"PART {Path(r['path']).parent.name}/{Path(r['path']).name} | VERDICT {'REJECT' if ex['max_score'] >= THR else 'PASS'} "
            f"(anomaly {ex['max_score']:.2f}, threshold {THR:.2f}) | DEFECT {CLASSES[int(pred[i])]} ({float(prob[i].max()):.2f}) | "
            f"LOCATION {location_words(rr, cc)} (~x={cc * 8 + 4:.0f}px, y={rr * 8 + 4:.0f}px of 224) | "
            f"AFFECTED {100 * ex['frac_above']:.1f}% of patches | TRUE {r['label']}")
    (OUT / "reports.txt").write_text("\n".join(lines) + "\n")
    print("\nExample reports:\n" + "\n".join(lines[:4]))

    # --- attention visualization: one test image per class
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, t in zip(axes, CLASSES):
        i = next(j for j, r in enumerate(sets["test"]) if r["label"] == t)
        ex = data["test"][3][i]
        ax.imshow(Image.open(sets["test"][i]["path"]).convert("RGB").resize((224, 224)))
        w = att_cls[i] / att_cls[i].max()
        ax.scatter(ex["cols"] * 8 + 4, ex["rows"] * 8 + 4, s=40 + 260 * w, c=w, cmap="autumn_r",
                   alpha=0.8, edgecolors="k")
        ax.set_title(f"true {t} -> pred {CLASSES[int(pred[i])]}", fontsize=9); ax.axis("off")
    fig.suptitle("CLS attention over the 32 most anomalous patches (bigger/yellower = more attention)")
    fig.tight_layout(); fig.savefig(OUT / "attention.png", dpi=110); plt.close(fig)

    c = np.array(curve)
    plt.figure(figsize=(6, 3.5)); plt.plot(c[:, 0], label="train loss"); plt.plot(c[:, 1], label="val macro-F1")
    plt.xlabel("epoch"); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "train_curve.png", dpi=110); plt.close()
    print("Saved results to", OUT)


THR = None
if __name__ == "__main__":
    main()