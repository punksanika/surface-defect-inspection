"""
noise_robustness.py: choose a noise fix for PatchCore using the VALIDATION set only.

Problem: in the robustness test, Gaussian noise (sigma 0.03 on pixels in [0,1]) made the
original model reject many GOOD nuts (final test-set numbers: experiments/robustness/, written by
src/evaluate_robustness.py). Noise adds tiny random changes to every pixel; the patch
features move away from the memory bank, so good patches look "unusual".

Two fixes are tried, alone and together:
  (a) blur_sigma   : light Gaussian blur on the input image (removes pixel-level noise)
  (b) noisy_copies : add noisy versions of the good training images to the memory bank
                     (so "good + noise" counts as normal)

Rule fixed BEFORE looking at any result (test set is never used here):
  1. for each setting, build the memory bank from train_good
  2. image threshold = max-F1 threshold on CLEAN val (same rule as the original model)
  3. at that threshold measure F1 on clean val and on noisy val (3 noise draws per image)
  4. pick the setting with the highest average of (clean val F1, noisy val F1);
     ties -> the simpler setting (less blur, fewer copies)

Run:
    python src/noise_robustness.py
Writes experiments/noise_robustness/val_grid.csv and chosen.json
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import patchcore as pc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "noise_robustness"
NOISE = 0.03                     # the noise level that failed in the robustness test
NOISE_DRAWS = [1, 2, 3]          # seeds: each val image is tested with 3 different noise patterns
BLURS = [0.0, 0.5, 1.0, 1.5, 2.0]
COPIES = [0, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = pc.get_device(); print("Device:", device)
    S = json.loads((ROOT / "splits.json").read_text())["splits"]
    val = S["val"]
    rows = []
    for blur in BLURS:
        ext = pc.PatchFeatureExtractor("resnet18", blur).to(device)
        for copies in COPIES:
            t0 = time.time()
            bank, _ = pc.build_memory_bank(ext, S["train_good"], device, 0.1, copies, NOISE, coreset_cpu=True)
            # clean validation -> threshold
            cs, cl, _, _, _ = pc.evaluate(ext, bank, val, device)
            thr = pc.choose_threshold(cs, cl)
            clean = pc.binary_report(cs, cl, thr)
            # noisy validation (3 draws), same threshold
            ns, nl = [], []
            for seed in NOISE_DRAWS:
                s, l, _, _, _ = pc.evaluate(ext, bank, val, device, noise_sigma=NOISE, seed=seed)
                ns.append(s); nl.append(l)
            ns, nl = np.concatenate(ns), np.concatenate(nl)
            noisy = pc.binary_report(ns, nl, thr)
            row = {
                "blur_sigma": blur, "noisy_copies": copies, "bank_size": bank.shape[0], "threshold": round(thr, 4),
                "val_clean_auroc": round(roc_auc_score(cl, cs), 4), "val_clean_f1": round(clean["f1"], 4),
                "val_clean_FRR": round(clean["false_reject_rate"], 4), "val_clean_catch": round(clean["recall_catch_rate"], 4),
                "val_noisy_auroc": round(roc_auc_score(nl, ns), 4), "val_noisy_f1": round(noisy["f1"], 4),
                "val_noisy_FRR": round(noisy["false_reject_rate"], 4), "val_noisy_catch": round(noisy["recall_catch_rate"], 4),
            }
            row["selection_score"] = round((clean["f1"] + noisy["f1"]) / 2, 4)
            rows.append(row)
            print(row, f"({time.time() - t0:.0f}s)", flush=True)

    with open(OUT / "val_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # best score; ties broken by the simpler setting (rows are already ordered simple -> complex,
    # and max() keeps the FIRST maximum)
    best = max(rows, key=lambda r: r["selection_score"])
    (OUT / "chosen.json").write_text(json.dumps(best, indent=2))
    print("\nCHOSEN (on validation only):", best)


if __name__ == "__main__":
    main()
