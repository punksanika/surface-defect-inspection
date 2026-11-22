"""
streamlit_app.py: demo interface for the inspection pipeline.

Run (from the project folder):
    streamlit run app/streamlit_app.py

Pages (choose in the sidebar):
  1. Inspection  - inspect one image: PatchCore score + heatmap -> PASS / REJECT -> defect type (+ boxes)
  2. Noise demo  - add Gaussian noise with a slider; compare ORIGINAL vs NOISE-ROBUST PatchCore
  3. Results     - benchmark tables (read live from experiments/*/metrics.json) + figures
  4. Pipeline    - diagram of the whole system

Pipeline per image:
  1. PatchCore (ResNet-18 memory bank)  -> anomaly score + heatmap -> PASS / REJECT (threshold chosen on val)
  2. if REJECT: ResNet-18 classifier     -> defect type (bent / color / flip / scratch)
  3. optional:  Faster R-CNN or RetinaNet -> boxes around defect instances

Every number shown on the Results page is read from files written by the training/evaluation
scripts, nothing is typed in by hand.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import patchcore as pc                      # noqa: E402  (our own files in src/)
import classifier as cl                     # noqa: E402

DEVICE = pc.get_device()
EXP = ROOT / "experiments"
PC_MODELS = {                               # name shown in the app -> folder with memory_bank.pt
    "Original PatchCore": EXP / "patchcore_r18",
    "Noise-robust PatchCore": EXP / "patchcore_r18_robust",
}
CLS_DIR = EXP / "cls_resnet18"
DETECTORS = {                               # name shown in the app -> (build_model name, folder with best.pt)
    "Faster R-CNN (more accurate)": ("fasterrcnn", EXP / "det_fasterrcnn"),
    "RetinaNet (faster)": ("retinanet", EXP / "det_retinanet"),
}


# ================================================================ model loading (cached = loaded once)
@st.cache_resource
def load_patchcore(folder):
    ck = torch.load(Path(folder) / "memory_bank.pt", map_location="cpu")
    # blur_sigma is stored with the bank (0 for the original model / old checkpoints)
    ext = pc.PatchFeatureExtractor(ck["backbone"], ck.get("blur_sigma", 0.0)).to(DEVICE)
    thr = json.loads((Path(folder) / "thresholds.json").read_text())
    return ext, ck["bank"].to(DEVICE), thr


@st.cache_resource
def load_classifier():
    m = cl.build_model("resnet18")
    m.load_state_dict(torch.load(CLS_DIR / "best.pt", map_location="cpu"))
    return m.to(DEVICE).eval()


@st.cache_resource
def load_detector(name):
    import detector as dt
    arch, folder = DETECTORS[name]
    m = dt.build_model(arch)
    m.load_state_dict(torch.load(folder / "best.pt", map_location="cpu"))
    return m.to(DEVICE).eval(), dt


# ================================================================ helpers
def overlay(img224, amap, model_name):
    """Colour only the unusual pixels, using the model's validation-chosen thresholds:
       below the pixel threshold        -> no colour (the nut is shown as it is)
       pixel threshold .. image threshold -> yellow -> orange -> red (red = as unusual as a REJECT)
    So a good nut stays clean and a defect shows up as a clear red spot."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    thr = load_patchcore(str(PC_MODELS[model_name]))[2]
    lo, hi = thr["pixel_threshold"], thr["image_threshold"]
    a = amap.numpy()
    t = np.clip((a - lo) / (hi - lo), 0, 1)
    colour = cm.jet(0.6 + 0.4 * t)[..., :3] * 255          # 0.6 = yellow ... 1.0 = dark red on the jet scale
    alpha = np.where(a >= lo, 0.35 + 0.35 * t, 0.0)[..., None]   # unusual pixels only, stronger when redder
    base = np.asarray(img224).astype(np.float32)
    return Image.fromarray((base * (1 - alpha) + colour * alpha).astype(np.uint8))


def add_noise(img224, sigma, seed):
    """Same kind of noise as the robustness test: pixels in [0,1], + N(0, sigma), clip to [0,1]."""
    x = np.asarray(img224).astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    x = np.clip(x + sigma * rng.standard_normal(x.shape).astype(np.float32), 0, 1)
    return Image.fromarray((x * 255).round().astype(np.uint8))


def run_patchcore(model_name, img224):
    """Returns score, anomaly map, threshold for one 224x224 PIL image."""
    ext, bank, thr = load_patchcore(str(PC_MODELS[model_name]))
    x = pc.TF.normalize(pc.TF.to_tensor(img224), pc.MEAN, pc.STD)
    score, amap = pc.score_image(ext, bank, x, DEVICE)
    return score, amap, thr["image_threshold"]


def pick_image(key):
    """Sidebar image chooser shared by the Inspection and Noise pages."""
    test_recs = json.loads((ROOT / "splits.json").read_text())["splits"]["test"]
    src = st.sidebar.radio("Image source", ["Held-out test image", "Upload"], key=key + "_src")
    if src == "Upload":
        up = st.sidebar.file_uploader("PNG / JPG of a metal nut", type=["png", "jpg", "jpeg"], key=key + "_up")
        # PatchCore only learned what a GOOD nut looks like for THIS product and camera setup.
        # Any other nut / background / angle is "out-of-distribution" and will (correctly) be rejected.
        st.sidebar.warning(
            "The model only knows the MVTec AD **metal_nut** product: a 4-tab nut, photographed "
            "straight from above on a black background. Other nuts (e.g. hex nuts), backgrounds or "
            "angles are unknown to it and will be REJECTED. For a fair test, use images from "
            "data/mvtec/metal_nut/test/.")
        if up is None:
            st.info("Upload an image or switch to a held-out test image.")
            st.stop()
        return Image.open(up).convert("RGB"), "unknown"
    names = [f"{Path(r['path']).parent.name}/{Path(r['path']).name}" for r in test_recs]
    k = st.sidebar.selectbox("Test image", range(len(names)), format_func=lambda i: names[i], key=key + "_sel")
    return Image.open(ROOT / test_recs[k]["path"]).convert("RGB"), test_recs[k]["label"]


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


# ================================================================ page 1: inspection (original demo)
def page_inspection():
    st.header("Inspect one image")
    img, true_label = pick_image("insp")
    model_name = st.sidebar.radio("PatchCore model", list(PC_MODELS), key="insp_model")
    show_boxes = st.sidebar.checkbox("Also draw defect boxes (detector)", value=False)
    if show_boxes:
        det_name = st.sidebar.radio("Detector", list(DETECTORS), key="insp_det")
        min_score = st.sidebar.slider("Minimum box score", 0.05, 0.95, 0.5, 0.05,
                                      help="Boxes below this confidence are hidden. RetinaNet's scores are "
                                           "often lower than Faster R-CNN's.")

    img224 = img.resize((pc.IMG_SIZE, pc.IMG_SIZE))
    t0 = time.perf_counter()
    score, amap, thr = run_patchcore(model_name, img224)
    t_pc = (time.perf_counter() - t0) * 1000
    reject = score >= thr

    c1, c2, c3 = st.columns(3)
    c1.image(img224, caption=f"Input (true label: {true_label})", width="stretch")
    c2.image(overlay(img224, amap, model_name),
             caption="Anomaly heatmap: coloured = unusual, red = as unusual as a REJECT", width="stretch")
    with c3:
        if reject:
            st.error(f"REJECT: anomaly score {score:.2f} ≥ threshold {thr:.2f}")
            clf = load_classifier()
            xc = cl.T.Compose([cl.T.Resize((224, 224)), cl.T.ToTensor(), cl.T.Normalize(cl.MEAN, cl.STD)])(img)
            with torch.no_grad():
                prob = torch.softmax(clf(xc.unsqueeze(0).to(DEVICE)), 1)[0].cpu()
            st.subheader(f"Defect type: {cl.CLASSES[int(prob.argmax())]}")
            st.bar_chart({c: float(p) for c, p in zip(cl.CLASSES, prob)})
            st.caption("The type comes from a supervised classifier trained on 56 labelled images; "
                       "it can only choose among these 4 known types.")
        else:
            st.success(f"PASS: anomaly score {score:.2f} < threshold {thr:.2f}")
        st.caption(f"{model_name} time: {t_pc:.1f} ms")

    if show_boxes:
        det, dt = load_detector(det_name)
        xd = dt.TF.to_tensor(img.resize((dt.SIZE, dt.SIZE)))
        with torch.no_grad():
            o = det([xd.to(DEVICE)])[0]
        from PIL import ImageDraw
        boxes, scores, labels = o["boxes"].cpu(), o["scores"].cpu(), o["labels"].cpu()
        im = img.resize((dt.SIZE, dt.SIZE))
        d = ImageDraw.Draw(im)
        keep = [i for i in range(len(scores)) if scores[i] >= min_score]
        for i in keep:                                        # confident boxes: red
            b = boxes[i].tolist()
            d.rectangle(b, outline="red", width=4)
            d.text((b[0] + 4, b[1] + 4), f"{dt.CLASSES[int(labels[i])]} {float(scores[i]):.2f}", fill="red")
        if keep:
            st.success(f"{len(keep)} box(es) with score ≥ {min_score:.2f}")
        elif len(scores):
            # nothing passed the cut-off: still show the detector's single most confident guess, clearly marked
            i = int(scores.argmax())
            b = boxes[i].tolist()
            d.rectangle(b, outline="orange", width=4)
            d.text((b[0] + 4, b[1] + 4), f"{dt.CLASSES[int(labels[i])]} {float(scores[i]):.2f}", fill="orange")
            st.warning(f"No box reaches score {min_score:.2f}. Orange = the detector's most confident guess "
                       f"(score {float(scores[i]):.2f}, below the cut-off).")
        else:
            st.info("The detector found no box at all in this image.")
        st.image(im, caption=f"{det_name}: red = score ≥ {min_score:.2f}", width=420)


# ================================================================ page 2: live noise demo
def page_noise():
    st.header("Noise demo: original vs noise-robust PatchCore")
    st.write("Gaussian noise is added to the 224×224 image (pixel values 0–1), like the robustness "
             "test. The noise-robust model has noisy copies of the good training images in "
             "its memory bank (setting chosen on the validation set).")
    img, true_label = pick_image("noise")
    sigma = st.slider("Noise sigma (0 = clean, robustness test used 0.03)", 0.0, 0.10, 0.03, 0.005)
    seed = st.number_input("Noise seed (different random pattern)", 0, 999, 0)

    img224 = img.resize((pc.IMG_SIZE, pc.IMG_SIZE))
    noisy = add_noise(img224, sigma, seed) if sigma > 0 else img224
    st.caption(f"True label: **{true_label}**")
    cols = st.columns(3)
    cols[0].image(noisy, caption=f"Input with noise sigma = {sigma:.3f}", width="stretch")
    for col, name in zip(cols[1:], PC_MODELS):
        score, amap, thr = run_patchcore(name, noisy)
        clean_score, _, _ = run_patchcore(name, img224)
        col.image(overlay(noisy, amap, name), caption=name, width="stretch")
        verdict = "REJECT" if score >= thr else "PASS"
        (col.error if verdict == "REJECT" else col.success)(
            f"{verdict}: score {score:.3f} (threshold {thr:.3f})")
        col.caption(f"Score on the clean image: {clean_score:.3f}  →  change {score - clean_score:+.3f}")


# ================================================================ page 3: results
def robustness_table(model):
    """One model's rows from experiments/robustness/results.json (written by src/evaluate_robustness.py)."""
    rows = read_json(EXP / "robustness" / "results.json") or []
    out = []
    for r in rows:
        if r["model"] != model:
            continue
        out.append({"condition": r["condition"], "AUROC": r["auroc"],
                    "defects caught": f"{r['TP']}/{r['TP'] + r['FN']}",
                    "good falsely rejected": f"{r['FP']}/{r['FP'] + r['TN']}",
                    "FRR %": round(100 * r["FP"] / (r["FP"] + r["TN"]), 1), "F1": r["f1"],
                    "ms / image": r["ms_per_image"], "peak RAM (MB)": r["peak_ram_mb"]})
    return pd.DataFrame(out)


def show_figs(paths, ncols=4, width=None):
    paths = [p for p in paths if Path(p).exists()]
    for i in range(0, len(paths), ncols):
        cols = st.columns(ncols)
        for c, p in zip(cols, paths[i:i + ncols]):
            c.image(str(p), caption=Path(p).stem, width="stretch")


def page_results():
    st.header("Results (test set: 26 good + 19 defective nuts)")
    st.caption("All values are read from experiments/*/metrics.json and experiments/robustness/; thresholds were "
               "chosen on the validation set, the test set was only used for these final numbers.")

    # ---- anomaly detection
    st.subheader("1. Anomaly detection (PatchCore, trained on good images only)")
    rows = []
    for name, folder in [("PatchCore ResNet-18", "patchcore_r18"),
                         ("PatchCore ResNet-18 noise-robust", "patchcore_r18_robust"),
                         ("PatchCore WideResNet-50", "patchcore_wrn50")]:
        m = read_json(EXP / folder / "metrics.json")
        if m is None:
            continue
        b = m["test_at_val_threshold"]
        rows.append({"model": name, "image AUROC": round(m["test_image_auroc"], 3),
                     "pixel AUROC": round(m["test_pixel_auroc"], 3),
                     "defects caught": f"{b['TP']}/{b['TP'] + b['FN']}",
                     "good falsely rejected": f"{b['FP']}/{b['FP'] + b['TN']}",
                     "F1": round(b["f1"], 3), "ms / image": round(m["ms_per_image"], 1)})
    m = read_json(EXP / "vae" / "metrics.json")
    if m:
        rows.append({"model": "VAE reconstruction error", "image AUROC": round(m["vae_recon_error_test_auroc"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # ---- classification
    st.subheader("2. Defect type classification (4 classes, supervised)")
    rows = []
    for name, folder in [("ResNet-18", "cls_resnet18"), ("GoogLeNet", "cls_googlenet")]:
        m = read_json(EXP / folder / "metrics.json")
        if m:
            rows.append({"model": name, "params (M)": m["params_millions"], "test accuracy": round(m["test_accuracy"], 3),
                         "test macro-F1": round(m["test_macro_f1"], 3), "ms / image": m["ms_per_image"]})
    m = read_json(EXP / "report_transformer" / "metrics.json")
    if m:
        rows.append({"model": "Report Transformer", "test accuracy": round(m["test_accuracy"], 3),
                     "test macro-F1": round(m["test_macro_f1"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    show_figs([EXP / "cls_resnet18" / "confusion_matrix.png", EXP / "cls_googlenet" / "confusion_matrix.png"], 2)

    # ---- detection
    st.subheader("3. Defect localisation with boxes (detectors)")
    rows = []
    for name, folder in [("Faster R-CNN", "det_fasterrcnn"), ("RetinaNet", "det_retinanet")]:
        m = read_json(EXP / folder / "metrics.json")
        if m:
            rows.append({"model": name, "mAP@50": round(m["mAP50"], 3),
                         **{f"AP50 {k}": round(v, 3) for k, v in m["AP50_per_class"].items()},
                         "params (M)": m["params_millions"], "ms / image": round(m["ms_per_image"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # ---- robustness (src/evaluate_robustness.py)
    st.subheader("4. Robustness: brightness and noise (test set)")
    for title, model in [("Before: original model (threshold 2.049)", "original"),
                         ("After: noise-robust model (threshold 2.136)", "noise_robust")]:
        st.markdown(f"**{title}**")
        st.dataframe(robustness_table(model), hide_index=True, width="stretch")
    grid = EXP / "noise_robustness" / "val_grid.csv"
    if grid.exists():
        with st.expander("How the noise fix was chosen (validation set only)"):
            st.dataframe(pd.read_csv(grid), hide_index=True, width="stretch")
            st.caption("Rule: highest average of clean-val F1 and noisy-val F1 (selection_score).")

    # ---- figures
    st.subheader("5. Figures")
    tabs = st.tabs(["PatchCore heatmaps", "Detector boxes", "CNN filters", "VAE", "Transformer"])
    with tabs[0]:
        show_figs(sorted((EXP / "patchcore_r18" / "figures").glob("*.png")), 2)
    with tabs[1]:
        d = st.radio("Detector", ["det_fasterrcnn", "det_retinanet"], horizontal=True)
        show_figs(sorted((EXP / d / "figures").glob("*.png")), 2)
    with tabs[2]:
        show_figs([EXP / "cls_resnet18" / "conv1_filters.png", EXP / "cls_resnet18" / "conv1_feature_maps.png"], 2)
    with tabs[3]:
        show_figs(sorted((EXP / "vae").glob("*.png")), 2)
    with tabs[4]:
        show_figs([EXP / "report_transformer" / "attention.png", EXP / "report_transformer" / "train_curve.png"], 2)
        rep = EXP / "report_transformer" / "reports.txt"
        if rep.exists():
            st.code("\n".join(rep.read_text().splitlines()[:8]), language=None)


# ================================================================ page 4: pipeline diagram
PIPELINE_DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
  img   [label="Camera image\\n(metal nut)", fillcolor="#ffffff"];
  pre   [label="Resize 224×224\\n+ ImageNet normalise"];
  feat  [label="ResNet-18\\n(layer2 + layer3 features)\\n784 patch vectors"];
  bank  [label="Memory bank\\n14,896 'good' patches\\n(good + noisy-good images)", shape=cylinder, fillcolor="#fff6dd"];
  knn   [label="Nearest-neighbour distance\\nper patch → heatmap"];
  dec   [label="Image score = max patch\\nscore ≥ threshold?", shape=diamond, fillcolor="#ffffff"];
  pass  [label="PASS", fillcolor="#d8f3dc"];
  cls   [label="ResNet-18 classifier\\nbent / color / flip / scratch", fillcolor="#fde2e2"];
  det   [label="Faster R-CNN / RetinaNet\\n(optional) boxes", fillcolor="#fde2e2"];
  img -> pre -> feat -> knn -> dec;
  bank -> knn;
  dec -> pass [label="no"];
  dec -> cls  [label="yes (REJECT)"];
  cls -> det;
}
"""


def page_pipeline():
    st.header("Pipeline")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    st.markdown("""
**Training (offline, once)**
1. Data split (seed 42, no leakage): 190 good images for the memory bank, 56 defects for supervised models,
   validation (26 good + 18 defects) to choose thresholds/settings, test (26 good + 19 defects) for final numbers only.
2. PatchCore: pass every good training image (and one noisy copy, for the robust model) through ResNet-18,
   keep the patch features, shrink them with greedy coreset selection → memory bank.
3. Threshold = value that gives the best F1 on the validation set.

**Inspection (every new nut)**
1. Extract patch features → distance of each patch to its nearest 'good' patch → heatmap.
2. Highest patch distance = anomaly score. Score ≥ threshold → REJECT.
3. Rejected nuts go to the classifier (defect type) and optionally the detector (boxes).
""")


# ================================================================ main
st.set_page_config(page_title="Surface Defect Inspection", layout="wide")
st.title("Industrial Surface Defect Inspection: metal_nut")
st.caption("PatchCore (anomaly + location) → ResNet-18 (defect type) → Faster R-CNN / RetinaNet (boxes). "
           "Research prototype on MVTec AD (CC BY-NC-SA 4.0); not validated for production use.")
PAGES = {"Inspection": page_inspection, "Noise demo": page_noise, "Results": page_results, "Pipeline": page_pipeline}
page = st.sidebar.radio("Page", list(PAGES))
st.sidebar.divider()
PAGES[page]()
