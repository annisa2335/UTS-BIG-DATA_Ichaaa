# faces_detector_simple.py
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from io import BytesIO
from pathlib import Path
import base64
import datetime

# =========================
# KONFIGURASI
# =========================
st.set_page_config(page_title="Face Detector (Real / Sketch / Synthetic)", layout="wide")
MODEL_PATH = "model/Annisa Humaira_Laporan 4.pt"

# =========================
# BACKGROUND (opsional)
# =========================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

bg_img = get_base64_image("bg2.jpg")
if bg_img:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .title {{
            text-align: center;
            font-size: 48px;
            font-weight: 800;
            color: #f0f0f0;
            text-shadow: 0 2px 6px rgba(0,0,0,.35);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown('<div class="title">Face Detection: Real / Sketch / Synthetic</div>', unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return YOLO(path)

def get_class_names(model) -> dict:
    """Map nama asli model ke label Real/Sketch/Synthetic"""
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
            mapped[cid] = name.replace("_", " ").title()
    return mapped

def draw_and_get_image(result):
    """Convert hasil YOLO (BGR) ke PIL RGB"""
    bgr = result.plot()
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)

def summarize_counts(result, names: dict):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    dets = []
    for cid, conf in zip(cls_ids, confs):
        dets.append((names.get(cid, str(cid)), float(conf)))
    return dets

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# =========================
# INFERENCE
# =========================
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1.25, 1.0], gap="large")

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Detection", use_container_width=True):
            with st.spinner("Detecting faces..."):
                model = load_model(MODEL_PATH)
                names = get_class_names(model)

                results = model(img, conf=0.5, iou=0.5, imgsz=640, verbose=False)
                result = results[0]
                annotated = draw_and_get_image(result)
                detections = summarize_counts(result, names)

            st.session_state["output"] = {
                "annotated": annotated,
                "detections": detections
            }

# =========================
# HASIL DETEKSI
# =========================
out = st.session_state.get("output")
if out:
    colA, colB = st.columns([1.25, 1.0], gap="large")

    with colA:
        st.image(out["annotated"], caption="Detections", use_container_width=True)

        buf = BytesIO()
        out["annotated"].save(buf, format="PNG")
        filename = f"faces_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        st.download_button(
            label="⬇️ Download Detection Result",
            data=buf.getvalue(),
            file_name=filename,
            mime="image/png",
            use_container_width=True
        )

    with colB:
        st.markdown("## Detection Result")
        if out["detections"]:
            for label, conf in out["detections"]:
                st.markdown(f"- **{label}** — Confidence: `{conf:.2f}`")
        else:
            st.info("No faces detected.")
