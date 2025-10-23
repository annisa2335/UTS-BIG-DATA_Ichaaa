# faces_detector.py
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from io import BytesIO
from pathlib import Path
import base64
import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Face Detector (Real / Sketch / Synthetic)", layout="wide")
MODEL_PATH = "model/Annisa Humaira_Laporan 4.pt"  # ganti jika path berbeda
SHOW_BG = False  # set True kalau pakai bg.jpeg

def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

if SHOW_BG:
    bg = get_base64_image("bg.jpeg")
    if bg:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{bg}");
                background-size: cover; background-position: center; background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

st.markdown(
    "<h1 style='text-align:center;'>Face Detection: Real / Sketch / Synthetic</h1>",
    unsafe_allow_html=True
)

# =========================
# MODEL (cache)
# =========================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return YOLO(path)

def get_class_names(model) -> dict:
    """
    Ambil nama kelas asli dari model.names lalu map ke label kanonik:
    - Real Face
    - Sketch Face
    - Synthetic Face
    Menghindari salah mapping saat urutan/penamaan di training berbeda.
    """
    raw = model.names if hasattr(model, "names") else {}
    mapped = {}
    for cid, name in raw.items():
        n = str(name).lower().replace("_", " ").strip()
        if any(k in n for k in ["real", "photo", "natural"]):
            mapped[cid] = "Real Face"
        elif any(k in n for k in ["sketch", "draw", "hand", "pencil", "line"]):
            mapped[cid] = "Sketch Face"
        elif any(k in n for k in ["synt", "fake", "gen", "ai", "cg", "render"]):
            mapped[cid] = "Synthetic Face"
        else:
            # fallback: rapikan nama asli
            mapped[cid] = name.replace("_", " ").title()
    return mapped

def draw_and_get_image(result):
    """Ultralytics result.plot() -> ndarray BGR. Convert ke PIL RGB."""
    bgr = result.plot()
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)

def summarize_counts(result, names: dict):
    if result.boxes is None or len(result.boxes) == 0:
        return {}
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    counts = {}
    for cid in cls_ids:
        label = names.get(cid, str(cid))
        counts[label] = counts.get(label, 0) + 1
    return counts

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.01)
    iou  = st.slider("IoU threshold (NMS)", 0.10, 0.90, 0.50, 0.01)
    imgsz = st.select_slider("Image size (inference)", options=[320, 416, 512, 640, 768, 960], value=640)
    show_labels = st.checkbox("Show labels", value=True)
    show_conf   = st.checkbox("Show confidences", value=True)

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# =========================
# INFERENCE
# =========================
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    c1, c2 = st.columns([1.25, 1.0], gap="large")
    with c1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Detection", use_container_width=True):
            with st.spinner("Running YOLO detection..."):
                model = load_model(MODEL_PATH)
                names = get_class_names(model)

                results = model(
                    img,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    verbose=False,
                    show_labels=show_labels,
                    show_conf=show_conf
                )
                result = results[0]
                out_img = draw_and_get_image(result)
                counts = summarize_counts(result, names)

                # siapkan tabel boxes
                df = pd.DataFrame()
                if result.boxes is not None and len(result.boxes) > 0:
                    cls = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    rows = []
                    for i in range(len(cls)):
                        rows.append({
                            "label": names.get(cls[i], str(cls[i])),
                            "conf": float(confs[i]),
                            "x1": float(xyxy[i][0]),
                            "y1": float(xyxy[i][1]),
                            "x2": float(xyxy[i][2]),
                            "y2": float(xyxy[i][3]),
                        })
                    df = pd.DataFrame(rows)

            st.session_state["det"] = {
                "annotated": out_img,
                "counts": counts,
                "boxes_df": df
            }

# =========================
# HASIL
# =========================
det = st.session_state.get("det")
if det:
    a1, a2 = st.columns([1.25, 1.0], gap="large")

    with a1:
        st.image(det["annotated"], caption="Detections", use_container_width=True)

        # Download annotated image
        buf = BytesIO()
        det["annotated"].save(buf, format="PNG")
        st.download_button(
            "⬇️ Download annotated image",
            data=buf.getvalue(),
            file_name=f"faces_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )

    with a2:
        st.markdown("## Detection Summary")
        if det["counts"]:
            for k, v in det["counts"].items():
                st.write(f"- **{k}**: {v}")
        else:
            st.info("No faces detected.")

        if not det["boxes_df"].empty:
            st.markdown("## Boxes (preview)")
            st.dataframe(det["boxes_df"].head(20), use_container_width=True)
