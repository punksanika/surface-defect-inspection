"""
evaluate_robustness.py: how do the ORIGINAL and the NOISE-ROBUST PatchCore models behave when the
camera image changes? Evaluated on the TEST set (45 images), never used for any choice.

Run:
    python src/evaluate_robustness.py
Writes experiments/robustness/results.json, results.csv and per_image.csv

Conditions (applied to pixel values in [0, 1] after resizing to 224x224, then clipped to [0, 1]):
  clean             no change
  brighter_20pct    x * 1.2
  darker_20pct      x * 0.8
  noise_sigma_0.03  x + 0.03 * N(0, 1)   (a fresh random generator with seed 0 for each condition,
                                          drawn image after image, so every run gives the same noise)

Nothing is re-trained or re-tuned: each model uses its saved memory bank and its saved
validation-chosen threshold (2.049 original, 2.136 noise-robust).

Reported per model and condition: AUROC, TP/FP/FN/TN, catch rate (recall), false-reject rate, F1,
ms per image, and peak memory (see measure_peak_memory below).
"""
import csv
import json
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.transforms import functional as TF

import patchcore as pc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "robustness"
MODELS = {"original": ROOT / "experiments" / "patchcore_r18",
          "noise_robust": ROOT / "experiments" / "patchcore_r18_robust"}
CONDITIONS = ["clean", "brighter_20pct", "darker_20pct", "noise_sigma_0.03"]
NOISE_SIGMA = 0.03
NOISE_SEED = 0


def load_raw(path):
    """Image -> RGB -> 224x224 (bilinear) -> tensor (3,224,224) with values in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((pc.IMG_SIZE, pc.IMG_SIZE), Image.BILINEAR)
    return TF.to_tensor(img)


def perturb(x, condition, generator):
    """Apply one test condition to a [0,1] image and clip back to [0, 1]."""
    if condition == "clean":
        y = x
    elif condition == "brighter_20pct":
        y = x * 1.2
    elif condition == "darker_20pct":
        y = x * 0.8
    elif condition == "noise_sigma_0.03":
        y = x + NOISE_SIGMA * torch.randn(x.shape, generator=generator)
    else:
        raise ValueError(condition)
    return y.clamp(0, 1)


def measure_peak_memory():
    """Peak resident memory (RAM) of this process so far, in MB. Each model runs in its own fresh
    process, so this is the peak for that model. On Apple GPUs (MPS) the memory the GPU driver
    holds is reported separately (gpu_driver_mb)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 1e6 if sys.platform == "darwin" else peak / 1e3     # macOS: bytes, Linux: KB
    gpu = None
    if torch.backends.mps.is_available():
        gpu = torch.mps.driver_allocated_memory() / 1e6
    elif torch.cuda.is_available():
        gpu = torch.cuda.max_memory_allocated() / 1e6
    return peak_mb, gpu


def evaluate_model(name):
    """Runs in a separate process: load one model, score the test set under every condition."""
    device = pc.get_device()
    model_dir = MODELS[name]
    saved = torch.load(model_dir / "memory_bank.pt", map_location="cpu")
    threshold = json.loads((model_dir / "thresholds.json").read_text())["image_threshold"]
    extractor = pc.PatchFeatureExtractor(saved["backbone"], saved.get("blur_sigma", 0.0)).to(device)
    bank = saved["bank"].to(device)

    test = json.loads((ROOT / "splits.json").read_text())["splits"]["test"]
    images = [load_raw(r["path"]) for r in test]
    labels = np.array([0 if r["label"] == "good" else 1 for r in test])
    pc.score_image(extractor, bank, TF.normalize(images[0], pc.MEAN, pc.STD), device)   # warm-up, not timed

    rows, per_image = [], []
    for cond in CONDITIONS:
        g = torch.Generator().manual_seed(NOISE_SEED)          # same noise for both models
        scores, times = [], []
        for r, x in zip(test, images):
            xn = TF.normalize(perturb(x, cond, g), pc.MEAN, pc.STD)
            t0 = time.perf_counter()
            s, _ = pc.score_image(extractor, bank, xn, device)   # full inference incl. the heatmap
            times.append(time.perf_counter() - t0)
            scores.append(s)
            per_image.append({"model": name, "condition": cond, "path": r["path"], "label": r["label"],
                              "score": round(s, 5), "decision": "REJECT" if s >= threshold else "PASS"})
        scores = np.array(scores)
        rep = pc.binary_report(scores, labels, threshold)
        peak_mb, gpu_mb = measure_peak_memory()
        rows.append({
            "model": name, "condition": cond, "threshold": round(threshold, 4),
            "auroc": round(float(roc_auc_score(labels, scores)), 4),
            "TP": rep["TP"], "FP": rep["FP"], "FN": rep["FN"], "TN": rep["TN"],
            "catch_rate": round(rep["recall_catch_rate"], 4),
            "false_reject_rate": round(rep["false_reject_rate"], 4),
            "f1": round(rep["f1"], 4),
            "ms_per_image": round(1000 * float(np.mean(times)), 1),
            "peak_ram_mb": round(peak_mb, 1),
            "gpu_driver_mb": None if gpu_mb is None else round(gpu_mb, 1),
            "device": str(device), "n_good": int((labels == 0).sum()), "n_defect": int((labels == 1).sum()),
        })
    return rows, per_image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_image = [], []
    # a fresh process per model, so the peak-memory number belongs to that model only
    ctx = mp.get_context("spawn")
    for name in MODELS:
        with ctx.Pool(1) as pool:
            r, p = pool.apply(evaluate_model, (name,))
        rows += r
        per_image += p

    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    for fname, data in [("results.csv", rows), ("per_image.csv", per_image)]:
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print(f"{'model':<13}{'condition':<18}{'AUROC':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} "
          f"{'catch':>7} {'FRR':>7} {'F1':>6} {'ms/img':>7} {'RAM MB':>7} {'GPU MB':>7}")
    for r in rows:
        gpu = "-" if r["gpu_driver_mb"] is None else f"{r['gpu_driver_mb']:.0f}"
        catch = 100 * r["TP"] / (r["TP"] + r["FN"])          # printed from the counts (no double rounding)
        frr = 100 * r["FP"] / (r["FP"] + r["TN"])
        print(f"{r['model']:<13}{r['condition']:<18}{r['auroc']:>6.3f} {r['TP']:>3} {r['FP']:>3} {r['FN']:>3} "
              f"{r['TN']:>3} {catch:>6.1f}% {frr:>6.1f}% {r['f1']:>6.3f} "
              f"{r['ms_per_image']:>7.1f} {r['peak_ram_mb']:>7.0f} {gpu:>7}")
    write_before_after(rows, per_image)
    print(f"\nDevice: {rows[0]['device']}.  Saved {OUT.relative_to(ROOT)}/results.json, results.csv, "
          "per_image.csv, before_after.md")


def write_before_after(rows, per_image):
    """Markdown summary of the noise fix, generated from the results (no number typed by hand)."""
    chosen = json.loads((ROOT / "experiments" / "noise_robustness" / "chosen.json").read_text())
    lines = ["# Noise fix: before / after (Python, TEST set: 26 good + 19 defects)", "",
             "Generated by src/evaluate_robustness.py. Setting chosen on VALIDATION only "
             "(experiments/noise_robustness/val_grid.csv, chosen.json): "
             f"blur_sigma = {chosen['blur_sigma']}, noisy_copies = {chosen['noisy_copies']}, "
             f"memory bank size {chosen['bank_size']}, threshold {chosen['threshold']}.", "",
             "| Condition | Model | AUROC | TP | FP | FN | TN | Catch | FRR | F1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        for r in rows:
            if r["condition"] == cond:
                lines.append(f"| {cond} | {r['model']} | {r['auroc']:.3f} | {r['TP']} | {r['FP']} | {r['FN']} | "
                             f"{r['TN']} | {100 * r['TP'] / (r['TP'] + r['FN']):.1f}% | "
                             f"{100 * r['FP'] / (r['FP'] + r['TN']):.1f}% | {r['f1']:.3f} |")
    lines.append("")
    for name in MODELS:
        missed = [Path(p["path"]).parent.name + "/" + Path(p["path"]).name for p in per_image
                  if p["model"] == name and p["condition"] == "noise_sigma_0.03"
                  and p["label"] != "good" and p["decision"] == "PASS"]
        lines.append(f"- Missed defects under noise, {name}: {', '.join(missed) if missed else 'none'}")
    (OUT / "before_after.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
