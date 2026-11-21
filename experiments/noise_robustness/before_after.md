# Noise fix: before / after (Java app, ONNX Runtime, TEST set: 26 good + 19 defects)

Setting chosen on VALIDATION only (see val_grid.csv, chosen.json): blur_sigma = 0, noisy_copies = 1 (aug noise 0.03).
Memory bank size kept the same (14896 x 384). Threshold re-chosen on clean validation: 2.0494 -> 2.1357.

| Condition        | Model  | AUROC | TP | FP | FN | TN | Catch | FRR   | F1    |
|------------------|--------|-------|----|----|----|----|-------|-------|-------|
| clean            | before | 0.960 | 18 | 1  | 1  | 25 | 94.7% | 3.8%  | 0.947 |
| clean            | after  | 0.966 | 18 | 1  | 1  | 25 | 94.7% | 3.8%  | 0.947 |
| brighter 20%     | before | 0.962 | 18 | 1  | 1  | 25 | 94.7% | 3.8%  | 0.947 |
| brighter 20%     | after  | 0.964 | 18 | 1  | 1  | 25 | 94.7% | 3.8%  | 0.947 |
| darker 20%       | before | 0.968 | 18 | 1  | 1  | 25 | 94.7% | 3.8%  | 0.947 |
| darker 20%       | after  | 0.966 | 18 | 0  | 1  | 26 | 94.7% | 0.0%  | 0.973 |
| noise sigma 0.03 | before | 0.931 | 18 | 14 | 1  | 12 | 94.7% | 53.8% | 0.706 |
| noise sigma 0.03 | after  | 0.974 | 15 | 0  | 4  | 26 | 78.9% | 0.0%  | 0.882 |

Trade-off under noise: false rejects 14 -> 0, but missed defects 1 -> 4
(after: color/004, color/015, scratch/020, scratch/021; before: color/004).

Python (PyTorch) test metrics, clean images:
- before (experiments/patchcore_r18):        image AUROC 0.974, pixel AUROC 0.993, 18/19 caught, 1/26 false reject
- after  (experiments/patchcore_r18_robust): image AUROC 0.962, pixel AUROC 0.992, 18/19 caught, 1/26 false reject

Java/Python parity on clean test: decisions agree 45/45 for both models
(max score difference 0.0535 before, 0.0653 after).
