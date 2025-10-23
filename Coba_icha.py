# faces_detector_layout_like_classifier.py
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from io import BytesIO
from pathlib import Path
import base64
import datetime

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Face Detection (Real / Sketch / Synthetic)", layout="wide")

MODEL_PATH = "model/Annisa Humaira_Laporan 4.pt"
CONF_THRESH = 0.50
IOU_THRESH  = 0.50
IMGSZ       = 640

# =========================
# BACKGROUND (opsional)
# =========================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        import base64 as _b64
        return _b64.b64encode(f.read()).decode("utf-8")

bg_img = get_base64_image("bg2.jpg") or get_base64_image("bg.jpeg")
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
# MODEL (cache)
# =========================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return YOLO(path)

def map_class_names(model) -> dict:
    """
    Samakan nama kelas berdasarkan kata kunci agar tidak kebalik:
    Real Face / Sketch Face / Synthetic Face
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
            mapped[cid] = name.replace("_", " ").title()
    return mapped

def annotate_image(result):
    """Kembalikan PIL image beranotasi dari result YOLO."""
    bgr = result.plot()            # ndarray BGR
    rgb = bgr[:, :, ::-1]         # ke RGB
    return Image.fromarray(rgb)

def top_detection(result, names: dict):
    """
    Ambil 1 deteksi dengan confidence tertinggi.
    Return (label, conf) atau (None, None) jika tidak ada.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None, None
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)
    idx = int(np.argmax(confs))
    label = names.get(clses[idx], str(clses[idx]))
    conf  = float(confs[idx])
    return label, conf

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# =========================
# INFERENCE + LAYOUT 3 KOLOM
# =========================
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns([1.2, 0.9, 1.2], gap="large")

    # Kiri: gambar upload (seperti classifier)
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # Tengah: tombol Run (centered look)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Run Classification", use_container_width=True):
            with st.spinner("Detecting..."):
                model = load_model(MODEL_PATH)
                names = map_class_names(model)
                results = model(img, conf=CONF_THRESH, iou=IOU_THRESH, imgsz=IMGSZ, verbose=False)
                result = results[0]

                # Ambil hasil utama
                pred_label, pred_conf = top_detection(result, names)
                annotated = annotate_image(result)

            st.session_state["pred"] = {
                "label": pred_label,
                "conf": pred_conf,
                "annotated": annotated
            }

    # Kanan: hasil (hanya label + confidence) — sama gaya dengan classifier
    with col3:
        pred = st.session_state.get("pred")
        st.markdown("<br><br>", unsafe_allow_html=True)
        if pred:
            if pred["label"] is not None:
                st.markdown(
                    f"""
                    <h2>Prediction: <span style="background:#e6f4ea;border-radius:8px;padding:4px 10px;">
                    {pred['label']}</span></h2>
                    <h3>Confidence: <span style="background:#e6f4ea;border-radius:8px;padding:2px 8px;">
                    {pred['conf']:.2f}</span></h3>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("No faces detected.")

            # (opsional) tombol download hasil anotasi
            if pred.get("annotated") is not None:
                buf = BytesIO()
                pred["annotated"].save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Detection Result",
                    data=buf.getvalue(),
                    file_name=f"faces_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
