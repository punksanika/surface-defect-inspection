# Demo script (about 10 minutes)

## Before the panel (10 minutes early)

Terminal 1: start the app and leave it running
```bash
cd ~/surface-defect-inspection
source .venv/bin/activate
streamlit run app/streamlit_app.py
```
The browser opens at http://localhost:8501. Click every page once so the models are already loaded.

Terminal 2: for the Java part (open a second terminal tab with Cmd+T)
```bash
cd ~/surface-defect-inspection
```
Run the Java command from step 5 once, so it's fast during the demo.

---

## Step 1: Problem and data (1 min, just talk)
SAY: "Factories need to find defective metal nuts automatically. I used the MVTec AD metal_nut dataset:
4 defect types (bent, color, flip, scratch). My split is fixed and leakage-safe: 190 good images for
training, validation to choose thresholds, and a test set of 26 good + 19 defective nuts used only for final numbers."

## Step 2: Streamlit, Pipeline page (1 min)
DO: sidebar -> Page -> **Pipeline**
SAY: "ResNet-18 turns the image into 784 patch features. The memory bank holds features of GOOD nuts only.
For each patch I measure the distance to the nearest good patch, which gives the heatmap. The highest distance is
the anomaly score; above the threshold means REJECT. Rejected nuts then go to a classifier for the
defect type and a detector for boxes."

## Step 3: Streamlit, Inspection page (3 min)
DO: sidebar -> Page -> **Inspection**, Image source -> **Held-out test image**, PatchCore model -> **Original PatchCore**

| # | Test image (sidebar) | Expected | What to point at |
|---|---|---|---|
| a | `good/006.png` | PASS (green) | "a good nut passes" |
| b | `scratch/004.png` | REJECT (red) + type | red spot on the heatmap = the scratch |
| c | `flip/019.png` | REJECT, high score | "flip is easy, very high score" |
| d | tick **Also run RetinaNet boxes** | box drawn | "a detector can also draw boxes" (then untick) |
| e | `color/004.png` | PASS (a miss!) | "my one missed defect: colour is the hardest type" |

## Step 4: Streamlit, Results page (2 min)
DO: sidebar -> Page -> **Results**, scroll slowly
SAY:
- Table 1: "PatchCore ResNet-18: image AUROC 0.974, catches 18 of 19 defects, 1 of 26 false rejects.
  WideResNet-50 has higher AUROC but is 2x slower."
- Table 2: "For defect type, ResNet-18 and GoogLeNet both reach 0.947 accuracy."
- Table 3: "For boxes, Faster R-CNN (mAP 0.820) beat RetinaNet (0.549)."
- At the bottom, open the **PatchCore heatmaps** tab and show 1-2 pictures.

## Step 5: Java deployment, in Terminal 2 (1-2 min)
TYPE:
```bash
java --enable-native-access=ALL-UNNAMED -cp target/app:lib/onnxruntime-1.23.0.jar Inspector inspect data/mvtec/metal_nut/test/scratch/020.png && open deploy/heatmap_020.png
```
Expected: `020.png -> REJECT (score 2.212, threshold 2.049, ...)`, then the heatmap opens in Preview.
SAY: "This is the deployed version: no Python, only Java plus the exported ONNX model. Java and Python
give the same decision on all 45 test images."

## Step 6: Streamlit, Noise demo page: my improvement (2 min)
DO: sidebar -> Page -> **Noise demo**, Test image -> `good/006.png`
1. Move the slider to **0.0**: both models PASS.
2. Move the slider to **0.03**: the Original model REJECTS a good nut, the Noise-robust model still PASSES.
SAY: "In Java testing, noise made the original model reject 14 of 26 good nuts. I added noisy copies of
the good training images to the memory bank, choosing the setting on validation only. Now 0 of 26 are falsely
rejected under noise. The trade-off: under noise, missed defects went from 1 to 4."
DO (optional): Page -> **Results** -> section 4: before/after tables side by side.

## Step 7: Limitations (30 sec)
DO: Page -> **Inspection**, Image source -> **Upload**: point at the yellow box.
SAY: "The model knows only this product and camera setup; a different nut is rejected. The test set is
small (45 images), and the noise fix was tested at one noise level only."

---

## If something goes wrong
- App frozen or showing an error: press Ctrl+C in Terminal 1, run `streamlit run app/streamlit_app.py` again.
- Java fails: show `experiments/java_app/java_eval_log.txt` and `experiments/java_app_robust/java_eval_log.txt`.
- Everything fails: open `README.md` (all tables and figures are there).
