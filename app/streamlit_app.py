"""
streamlit_app.py: demo interface for the inspection pipeline.

Run (from the project folder):
    streamlit run app/streamlit_app.py

Pipeline per image:
  1. PatchCore (ResNet-18 memory bank)  -> anomaly score + heatmap -> PASS / REJECT (threshold chosen on val)
  2. if REJECT: ResNet-18 classifier     -> defect type (bent / color / flip / scratch)
  3. optional:  RetinaNet                -> boxes around defect instances
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import patchcore as pc                      # noqa: E402  (our own files in src/)
import classifier as cl                     # noqa: E402

DEVICE = pc.get_device()
PC_DIR = ROOT / "experiments" / "patchcore_r18"
CLS_DIR = ROOT / "experiments" / "cls_resnet18"
DET_DIR = ROOT / "experiments" / "det_retinanet"


@st.cache_resource
def load_patchcore():
    ck = torch.load(PC_DIR / "memory_bank.pt", map_location="cpu")
    ext = pc.PatchFeatureExtractor(ck["backbone"]).to(DEVICE)
    thr = json.loads((PC_DIR / "thresholds.json").read_text())
    return ext, ck["bank"].to(DEVICE), thr


@st.cache_resource
def load_classifier():
    m = cl.build_model("resnet18")
    m.load_state_dict(torch.load(CLS_DIR / "best.pt", map_location="cpu"))
    return m.to(DEVICE).eval()


@st.cache_resource
def load_detector():
    import detector as dt
    m = dt.build_model("retinanet")
    m.load_state_dict(torch.load(DET_DIR / "best.pt", map_location="cpu"))
    return m.to(DEVICE).eval(), dt


def overlay(img224, amap, vmax):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    a = np.clip(amap.numpy() / vmax, 0, 1)
    heat = (cm.jet(a)[..., :3] * 255).astype(np.uint8)
    return Image.blend(img224, Image.fromarray(heat), alpha=0.45)


st.set_page_config(page_title="Surface Defect Inspection", layout="wide")
st.title("Industrial Surface Defect Inspection: metal_nut")
st.caption("PatchCore (anomaly + location) → ResNet-18 (defect type) → RetinaNet (boxes). "
           "Research prototype on MVTec AD (CC BY-NC-SA 4.0); not validated for production use.")

splits = json.loads((ROOT / "splits.json").read_text())["splits"]
test_recs = splits["test"]
src = st.sidebar.radio("Image source", ["Held-out test image", "Upload"])
if src == "Upload":
    up = st.sidebar.file_uploader("PNG / JPG of a metal nut", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("Upload an image or switch to a held-out test image.")
        st.stop()
    img = Image.open(up).convert("RGB"); true_label = "unknown"
else:
    names = [f"{Path(r['path']).parent.name}/{Path(r['path']).name}" for r in test_recs]
    k = st.sidebar.selectbox("Test image", range(len(names)), format_func=lambda i: names[i])
    img = Image.open(test_recs[k]["path"]).convert("RGB"); true_label = test_recs[k]["label"]
show_boxes = st.sidebar.checkbox("Also run RetinaNet boxes", value=False)

ext, bank, thr = load_patchcore()
x = pc.TF.normalize(pc.TF.to_tensor(img.resize((pc.IMG_SIZE, pc.IMG_SIZE))), pc.MEAN, pc.STD)
t0 = time.perf_counter()
score, amap = pc.score_image(ext, bank, x, DEVICE)
t_pc = (time.perf_counter() - t0) * 1000
reject = score >= thr["image_threshold"]

c1, c2, c3 = st.columns(3)
img224 = img.resize((pc.IMG_SIZE, pc.IMG_SIZE))
c1.image(img224, caption=f"Input (true label: {true_label})", use_container_width=True)
c2.image(overlay(img224, amap, max(float(amap.max()), thr["image_threshold"] * 1.5)),
         caption="Anomaly heatmap (red = unusual)", use_container_width=True)
with c3:
    if reject:
        st.error(f"REJECT: anomaly score {score:.2f} ≥ threshold {thr['image_threshold']:.2f}")
        clf = load_classifier()
        xc = cl.T.Compose([cl.T.Resize((224, 224)), cl.T.ToTensor(), cl.T.Normalize(cl.MEAN, cl.STD)])(img)
        with torch.no_grad():
            prob = torch.softmax(clf(xc.unsqueeze(0).to(DEVICE)), 1)[0].cpu()
        st.subheader(f"Defect type: {cl.CLASSES[int(prob.argmax())]}")
        st.bar_chart({c: float(p) for c, p in zip(cl.CLASSES, prob)})
        st.caption("The type comes from a supervised classifier trained on 56 labelled images; "
                   "it can only choose among these 4 known types.")
    else:
        st.success(f"PASS: anomaly score {score:.2f} < threshold {thr['image_threshold']:.2f}")
    st.caption(f"PatchCore time: {t_pc:.1f} ms")

if show_boxes:
    det, dt = load_detector()
    xd = dt.TF.to_tensor(img.resize((dt.SIZE, dt.SIZE)))
    with torch.no_grad():
        o = det([xd.to(DEVICE)])[0]
    from PIL import ImageDraw
    im = img.resize((dt.SIZE, dt.SIZE)); d = ImageDraw.Draw(im)
    for b, s, l in zip(o["boxes"].cpu(), o["scores"].cpu(), o["labels"].cpu()):
        if s >= 0.5:
            d.rectangle(b.tolist(), outline="red", width=3)
            d.text((float(b[0]) + 3, float(b[1]) + 3), f"{dt.CLASSES[int(l)]} {float(s):.2f}", fill="red")
    st.image(im, caption="RetinaNet detections (score ≥ 0.5)", width=420)