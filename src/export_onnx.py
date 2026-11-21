"""
export_onnx.py: export the PatchCore ResNet-18 feature extractor to ONNX so the Java app can run it.

Run:
    python src/export_onnx.py                                   # original model  -> deploy/
    python src/export_onnx.py --pc_dir experiments/patchcore_r18_robust --out deploy_robust
                                                                # noise-robust model -> deploy_robust/

Writes into deploy/:
  patchcore_r18.onnx    image (1,3,224,224) -> features (1,384,28,28)   [the neural-network part]
  memory_bank.f32       raw float32, little-endian, shape (M,384)       [the "normal" patches]
  model.properties      sizes, threshold, normalization constants       [plain text, easy for Java]
  test_list.csv         path,label,python_score for every TEST image    [Java evaluates the same images
                                                                          and checks it agrees with Python]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patchcore as pc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc_dir", default="experiments/patchcore_r18", help="folder with memory_bank.pt")
    ap.add_argument("--out", default="deploy", help="output folder for the Java app")
    args = ap.parse_args()
    PC_DIR, OUT = ROOT / args.pc_dir, ROOT / args.out
    OUT.mkdir(exist_ok=True)
    ck = torch.load(PC_DIR / "memory_bank.pt", map_location="cpu")
    thr = json.loads((PC_DIR / "thresholds.json").read_text())
    blur = ck.get("blur_sigma", 0.0)       # old checkpoints have no blur -> 0
    # the blur layer is part of the network, so it is exported INTO the .onnx file
    ext = pc.PatchFeatureExtractor(ck["backbone"], blur).eval()    # CPU, float32
    print("Exporting", PC_DIR, "| input blur sigma =", blur)
    dummy = torch.randn(1, 3, pc.IMG_SIZE, pc.IMG_SIZE)
    onnx_path = OUT / "patchcore_r18.onnx"

    # 1. export the network (try the classic exporter first, then the newer one)
    try:
        torch.onnx.export(ext, dummy, str(onnx_path), input_names=["image"], output_names=["features"],
                          opset_version=17, dynamo=False)
    except Exception as e:
        print("Classic exporter failed (", type(e).__name__, ") -> trying dynamo exporter")
        torch.onnx.export(ext, dummy, str(onnx_path), input_names=["image"], output_names=["features"],
                          dynamo=True)
    print("Saved", onnx_path, f"({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # 2. check ONNX Runtime gives the same numbers as PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = ext(dummy).numpy()
    got = sess.run(None, {"image": dummy.numpy()})[0]
    print("ONNX output shape", got.shape, "| max |PyTorch - ONNX| =", float(np.abs(ref - got).max()))

    # 3. memory bank as raw little-endian float32
    bank = ck["bank"].float().numpy()
    bank.astype("<f4").tofile(OUT / "memory_bank.f32")
    print("Saved memory bank", bank.shape)

    # 4. settings for Java
    props = {
        "image_size": pc.IMG_SIZE, "feature_dim": bank.shape[1], "feature_h": got.shape[2], "feature_w": got.shape[3],
        "bank_size": bank.shape[0], "image_threshold": thr["image_threshold"], "blur_sigma": blur,
        "mean": ",".join(map(str, pc.MEAN)), "std": ",".join(map(str, pc.STD)),
    }
    (OUT / "model.properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()))

    # 5. Python scores of every test image (reference for the Java parity check)
    S = json.loads((ROOT / "splits.json").read_text())["splits"]
    bank_t = ck["bank"]
    lines = ["path,label,python_score"]
    for r in S["test"]:
        s, _ = pc.score_image(ext, bank_t, pc.load_image(ROOT / r["path"]), torch.device("cpu"))
        lines.append(f"{(ROOT / r['path']).resolve()},{r['label']},{s:.6f}")
    (OUT / "test_list.csv").write_text("\n".join(lines) + "\n")
    print("Saved", OUT / "test_list.csv", f"({len(lines) - 1} test images)")


if __name__ == "__main__":
    main()