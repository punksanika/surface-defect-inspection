"""Automated checks for the inspection pipeline.   Run:  python -m pytest

1. the data split has the documented sizes and no image in two splits (no leakage)
2. both saved models have their validation-chosen thresholds
3. PatchCore gives the same score twice for the same image (deterministic)
4. a known defective image is REJECTED and a known good image PASSES
5. the robustness-test perturbations keep pixel values in [0, 1]
6. the saved robustness results are consistent with their own TP/FP/FN/TN counts

Tests 3-4 need the dataset (data/) and the memory bank (.pt), which are not on GitHub;
they are skipped automatically when those files are missing.
"""
import json
from pathlib import Path

import pytest
import torch

import evaluate_robustness as er
import patchcore as pc

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
SCRATCH = ROOT / "data/mvtec/metal_nut/test/scratch/020.png"
GOOD = ROOT / "data/mvtec/metal_nut/test/good/006.png"


def splits():
    return json.loads((ROOT / "splits.json").read_text())["splits"]


# ---------------------------------------------------------------- 1. data split
def test_split_sizes():
    s = splits()
    count = lambda name, good: sum((r["label"] == "good") == good for r in s[name])
    assert len(s["train_good"]) == 190 and count("train_good", True) == 190
    assert len(s["sup_train"]) == 56 and count("sup_train", False) == 56
    assert count("val", True) == 26 and count("val", False) == 18
    assert count("test", True) == 26 and count("test", False) == 19


def test_split_has_no_leakage():
    seen = {}
    for name, records in splits().items():
        for r in records:
            assert r["path"] not in seen, f"{r['path']} is in both {seen[r['path']]} and {name}"
            seen[r["path"]] = name


# ---------------------------------------------------------------- 2. thresholds
@pytest.mark.parametrize("folder, expected", [("patchcore_r18", 2.049), ("patchcore_r18_robust", 2.136)])
def test_thresholds(folder, expected):
    thr = json.loads((EXP / folder / "thresholds.json").read_text())
    assert thr["image_threshold"] == pytest.approx(expected, abs=1e-3)
    assert thr["pixel_threshold"] < thr["image_threshold"]


# ---------------------------------------------------------------- 3-4. model behaviour
@pytest.fixture(scope="module")
def original_model():
    bank_file = EXP / "patchcore_r18" / "memory_bank.pt"
    if not bank_file.exists() or not SCRATCH.exists():
        pytest.skip("needs data/ and experiments/patchcore_r18/memory_bank.pt")
    device = pc.get_device()
    ck = torch.load(bank_file, map_location="cpu")
    ext = pc.PatchFeatureExtractor(ck["backbone"], ck.get("blur_sigma", 0.0)).to(device)
    thr = json.loads((EXP / "patchcore_r18" / "thresholds.json").read_text())["image_threshold"]
    score = lambda path: pc.score_image(ext, ck["bank"].to(device), pc.load_image(path), device)[0]
    return score, thr


def test_score_is_deterministic(original_model):
    score, _ = original_model
    assert score(SCRATCH) == pytest.approx(score(SCRATCH), abs=1e-5)


def test_defect_rejected_good_passes(original_model):
    score, thr = original_model
    assert score(SCRATCH) >= thr, "scratch/020 should be REJECTED"
    assert score(GOOD) < thr, "good/006 should PASS"


# ---------------------------------------------------------------- 5. perturbations
@pytest.mark.parametrize("condition", er.CONDITIONS)
def test_perturbation_keeps_pixels_in_range(condition):
    x = torch.rand(3, 224, 224, generator=torch.Generator().manual_seed(0))
    y = er.perturb(x, condition, torch.Generator().manual_seed(0))
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    if condition == "clean":
        assert torch.equal(y, x)
    if condition == "brighter_20pct":
        assert torch.allclose(y, (x * 1.2).clamp(0, 1))


# ---------------------------------------------------------------- 6. saved robustness results
def test_robustness_results_are_consistent():
    rows = json.loads((EXP / "robustness" / "results.json").read_text())
    assert len(rows) == 2 * len(er.CONDITIONS)                       # both models x all conditions
    for r in rows:
        assert r["TP"] + r["FN"] == 19 and r["FP"] + r["TN"] == 26     # every test image counted once
        assert r["catch_rate"] == pytest.approx(r["TP"] / 19, abs=1e-4)
        assert r["false_reject_rate"] == pytest.approx(r["FP"] / 26, abs=1e-4)
        precision = r["TP"] / max(r["TP"] + r["FP"], 1)
        recall = r["TP"] / 19
        assert r["f1"] == pytest.approx(2 * precision * recall / (precision + recall), abs=1e-3)
