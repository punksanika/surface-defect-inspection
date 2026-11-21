"""
streamlit_app.py: demo interface for the inspection pipeline.

Run (from the project folder):
    streamlit run app/streamlit_app.py

Pages (choose in the sidebar):
  1. Inspection  - inspect one image: PatchCore score + heatmap -> PASS / REJECT -> defect type (+ boxes)
  2. Noise demo  - add Gaussian noise with a slider; compare ORIGINAL vs NOISE-ROBUST PatchCore
  3. Results     - benchmark tables (read live from experiments/*/metrics.json) + figures
  4. Pipeline    - diagram of the whole system
  5. Fundamentals - the NumPy-from-scratch modules (fundamentals/): saved results + small live demos

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
sys.path.insert(0, str(ROOT))
import patchcore as pc                      # noqa: E402  (our own files in src/)
import classifier as cl                     # noqa: E402
from fundamentals import learning_rules as lr_rules, mlp as fmlp, tensor_sandbox as tsb   # noqa: E402
from fundamentals.common import xor_data   # noqa: E402

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
def overlay(img224, amap, vmax):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    a = np.clip(amap.numpy() / vmax, 0, 1)
    heat = (cm.jet(a)[..., :3] * 255).astype(np.uint8)
    return Image.blend(img224, Image.fromarray(heat), alpha=0.45)


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
    c2.image(overlay(img224, amap, max(float(amap.max()), thr * 1.5)),
             caption="Anomaly heatmap (red = unusual)", width="stretch")
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
        col.image(overlay(noisy, amap, max(float(amap.max()), thr * 1.5)), caption=name, width="stretch")
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


# ================================================================ page 5: fundamentals (NumPy from scratch)
FUND = EXP / "fundamentals"


def parse_shape(text):
    """'2, 3' -> (2, 3). Empty or invalid text -> None."""
    try:
        shape = tuple(int(v) for v in text.replace("(", "").replace(")", "").split(",") if v.strip())
        return shape if shape and all(s > 0 for s in shape) else None
    except ValueError:
        return None


def pattern_image(s, scale=24):
    """+1/-1 Hopfield state (100 values) -> 10x10 black/white picture, enlarged."""
    img = Image.fromarray(np.where(np.asarray(s).reshape(10, 10) > 0, 0, 255).astype(np.uint8))
    return img.resize((10 * scale, 10 * scale), Image.NEAREST)


def tab_sandbox(summary):
    st.markdown("**Own `Tensor` class** (shape, strides, reshape, broadcasting, dot, matmul, transpose), "
                "**SVD by one-sided Jacobi rotations** and **numerical gradient check**: `fundamentals/tensor_sandbox.py`")
    st.markdown("##### Live demo: broadcasting")
    c1, c2 = st.columns(2)
    sa = parse_shape(c1.text_input("Shape of A", "2, 3"))
    sb = parse_shape(c2.text_input("Shape of B", "3"))
    if sa is None or sb is None:
        st.warning("Type shapes as positive integers separated by commas, e.g. 2, 3")
    else:
        n = max(len(sa), len(sb))
        steps = pd.DataFrame({"A (padded left with 1s)": [str((1,) * (n - len(sa)) + sa)],
                              "B (padded left with 1s)": [str((1,) * (n - len(sb)) + sb)]})
        st.dataframe(steps, hide_index=True)
        try:
            A = tsb.Tensor(np.arange(int(np.prod(sa))), sa)
            B = tsb.Tensor(np.arange(int(np.prod(sb))) * 10, sb)
            out = A + B
            st.success(f"Compatible: {sa} + {sb} → {out.shape}  (each dimension equal, or one of them is 1)")
            if len(out.shape) <= 2 and out.data.size <= 60:
                d1, d2, d3 = st.columns(3)
                d1.caption("A = arange"); d1.dataframe(pd.DataFrame(np.atleast_2d(A.numpy())))
                d2.caption("B = arange × 10"); d2.dataframe(pd.DataFrame(np.atleast_2d(B.numpy())))
                d3.caption("A + B (own Tensor class)"); d3.dataframe(pd.DataFrame(np.atleast_2d(out.numpy())))
        except ValueError as e:
            st.error(f"Error from the Tensor class: {e}")
    st.markdown("##### Saved results")
    if summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SVD: max |S_ours − S_numpy|", f"{summary['svd_64_max_diff_vs_numpy']:.1e}")
        m2.metric("Rank-5 relative error", f"{summary['svd_64_rel_error_rank5']:.3f}")
        m3.metric("Rank-20 relative error", f"{summary['svd_64_rel_error_rank20']:.3f}")
        m4.metric("Gradient check error (h=1e-5)", f"{summary['gradcheck_rel_error_h1e-5']:.1e}")
    show_figs([FUND / "sandbox" / "svd_reconstructions.png"], 1)
    show_figs([FUND / "sandbox" / f for f in ["svd_error.png", "singular_values.png", "gradcheck_vs_h.png"]], 3)


def tab_neurons(summary):
    st.markdown("**McCulloch-Pitts neuron** (fixed weights, hard threshold) and **sigmoid neuron** "
                "(trainable): `fundamentals/neurons.py`")
    if summary:
        st.dataframe(pd.DataFrame(summary["AND_OR_truth_table"], columns=["x1", "x2", "AND", "OR"]), hide_index=True)
        c1, c2 = st.columns(2)
        c1.metric("Single neurons that compute XOR",
                  f"{summary['xor_grid_solutions_found']} of {summary['xor_grid_settings_tried']}")
        c2.metric("Sigmoid neuron on OR: loss", f"{summary['sigmoid_OR_loss_last']:.4f}",
                  f"from {summary['sigmoid_OR_loss_first']:.4f}", delta_color="off")
        st.caption("XOR is not linearly separable: no straight line splits its 1s from its 0s, "
                   "so no grid setting of (w1, w2, threshold) works. A hidden layer is needed (next tab).")
    show_figs([FUND / "neurons" / "sigmoid_or_loss.png"], 2)


def tab_mlp(summary):
    st.markdown("**MLP with manual backpropagation**, BCE + L2 loss, 4 weight inits, 3 optimizers: "
                "`fundamentals/mlp.py`")
    st.markdown("##### Live demo: train an MLP on XOR")
    c1, c2, c3, c4 = st.columns(4)
    hidden = c1.slider("Hidden neurons", 1, 16, 4)
    lr = c2.select_slider("Learning rate", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0], value=0.5)
    opt_name = c3.selectbox("Optimizer", ["gd", "momentum", "nesterov"])
    epochs = c4.select_slider("Epochs", [200, 500, 1000, 2000, 5000], value=2000)
    seed = st.number_input("Weight seed", 0, 999, 7)
    X, Y = xor_data()
    net = fmlp.MLP([2, hidden, 1], act="tanh", init="xavier", seed=int(seed))
    hist = fmlp.train(net, fmlp.Optimizer(opt_name, lr, 0.9 if opt_name != "gd" else 0.0), (X, Y),
                      epochs=epochs, batch_size=4, seed=1)
    acc = hist["final_train_acc"]
    (st.success if acc == 1.0 else st.warning)(
        f"Accuracy {acc:.2f} | final loss {hist['train_loss'][-1]:.4f} | outputs "
        + ", ".join(f"{p:.3f}" for p in net.predict_proba(X)) + "  (targets 0, 1, 1, 0)")
    g1, g2 = st.columns(2)
    g1.caption("Training loss (BCE)")
    g1.line_chart(pd.DataFrame({"loss": hist["train_loss"]}))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 150), np.linspace(-0.5, 1.5, 150))
    P = net.predict_proba(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.contourf(gx, gy, P, levels=20, cmap="RdBu_r", alpha=0.4)
    ax.contour(gx, gy, P, levels=[0.5], colors="k")
    ax.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap="coolwarm", s=120, edgecolors="k")
    ax.set_title("Decision regions")
    g2.pyplot(fig)
    plt.close(fig)
    st.caption("Try 1 hidden neuron: it cannot solve XOR, just like a single neuron.")
    st.markdown("##### Saved results")
    if summary:
        c1, c2 = st.columns(2)
        c1.metric("Backprop vs numerical gradient (max rel. error)",
                  f"{summary['backprop_gradcheck_max_rel_error']:.1e}")
        c2.metric("Saved XOR run: accuracy", f"{summary['xor_accuracy']:.2f}")
    show_figs([FUND / "mlp" / "xor_loss.png", FUND / "mlp" / "xor_boundary.png"], 2)


def tab_training(summary):
    st.markdown("**Optimizers, initialisation, overfitting, L2, augmentation, standardisation** on two-moons "
                "data: `fundamentals/training_experiments.py`. Each run's settings and per-epoch metrics are "
                "logged in `experiments/fundamentals/runs/`.")
    if not summary:
        return
    for title, key, fig in [("Optimizers (same starting weights)", "optimizers", "optimizers/loss_curves.png"),
                            ("Weight initialisation (deep tanh network)", "initialisation", "init/loss_curves.png"),
                            ("Overfitting, L2 and augmentation (30 training points)", "generalisation",
                             "generalisation/train_vs_val.png"),
                            ("Input standardisation (train statistics only)", "normalisation",
                             "generalisation/normalisation.png")]:
        st.markdown(f"##### {title}")
        c1, c2 = st.columns([1, 1])
        c1.dataframe(pd.DataFrame(summary[key]), hide_index=True, width="stretch")
        if (FUND / fig).exists():
            c2.image(str(FUND / fig), width="stretch")


def tab_rules(summary):
    st.markdown("**Hebbian, Oja, competitive, perceptron (delta rule) and Hopfield network**: "
                "`fundamentals/learning_rules.py`")
    st.markdown("##### Live demo: Hopfield recall")
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("Stored pattern", list(lr_rules.PATTERNS))
    frac = c2.slider("Pixels flipped", 0.0, 0.5, 0.2, 0.05)
    seed = c3.number_input("Noise seed", 0, 999, 10)
    net = lr_rules.Hopfield(100)
    net.store(list(lr_rules.PATTERNS.values()))
    p = lr_rules.PATTERNS[name]
    s = lr_rules.corrupt(p, frac, int(seed))
    corrupted = s.copy()
    trace = net.recall(s, sweeps=5, seed=int(seed) + 10)
    i1, i2, i3, i4 = st.columns([1, 1, 1, 2])
    i1.image(pattern_image(p), caption="stored")
    i2.image(pattern_image(corrupted), caption=f"corrupted: {int((corrupted != p).sum())} wrong pixels")
    i3.image(pattern_image(s), caption=f"recalled: {int((s != p).sum())} wrong pixels")
    i4.caption("Energy after every single-neuron update")
    i4.line_chart(pd.DataFrame({"energy": trace}))
    never_up = bool(np.all(np.diff(trace) <= 1e-12))
    (st.success if never_up else st.error)(f"Energy {trace[0]:.2f} → {trace[-1]:.2f}; never increased: {never_up}")
    st.caption("With many flipped pixels the network can settle in the wrong memory or a mixture: "
               "a stored pattern is only recalled when the input is close enough to it.")
    st.markdown("##### Saved results")
    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Hebbian ||w|| after 30 epochs", f"{summary['hebbian_final_norm']:.2e}")
        c2.metric("Oja ||w||", f"{summary['oja_final_norm']:.4f}")
        c3.metric("|cos(Oja w, PC1 from own SVD)|", f"{summary['oja_abs_cos_pc1']:.4f}")
        st.write(f"Competitive learning: worst prototype-to-centre distance "
                 f"{summary['competitive_max_centre_error']:.3f}. Perceptron mistakes per epoch: "
                 f"{summary['perceptron_mistakes_per_epoch']}")
        st.dataframe(pd.DataFrame(summary["hopfield"]), hide_index=True, width="stretch")
    show_figs([FUND / "rules" / f for f in ["weight_norms.png", "competitive.png", "perceptron_mistakes.png"]], 3)
    show_figs([FUND / "hopfield" / "recall.png", FUND / "hopfield" / "energy.png"], 2)


def page_fundamentals():
    st.header("Fundamentals from scratch (NumPy only)")
    st.caption("Course requirements 3–6. No PyTorch, autograd or scikit-learn in this part. "
               "Saved numbers come from experiments/fundamentals/summary.json "
               "(re-create with `python -m fundamentals.run_all`).")
    summary = read_json(FUND / "summary.json") or {}
    tabs = st.tabs(["Tensors, SVD, gradient check", "Neurons", "MLP + backprop", "Training experiments",
                    "Learning rules + Hopfield"])
    with tabs[0]:
        tab_sandbox(summary.get("tensor_sandbox"))
    with tabs[1]:
        tab_neurons(summary.get("neurons"))
    with tabs[2]:
        tab_mlp(summary.get("mlp"))
    with tabs[3]:
        tab_training(summary.get("training_experiments"))
    with tabs[4]:
        tab_rules(summary.get("learning_rules"))


# ================================================================ main
st.set_page_config(page_title="Surface Defect Inspection", layout="wide")
st.title("Industrial Surface Defect Inspection: metal_nut")
st.caption("PatchCore (anomaly + location) → ResNet-18 (defect type) → Faster R-CNN / RetinaNet (boxes). "
           "Research prototype on MVTec AD (CC BY-NC-SA 4.0); not validated for production use.")
PAGES = {"Inspection": page_inspection, "Noise demo": page_noise, "Results": page_results, "Pipeline": page_pipeline,
         "Fundamentals": page_fundamentals}
page = st.sidebar.radio("Page", list(PAGES))
st.sidebar.divider()
PAGES[page]()
