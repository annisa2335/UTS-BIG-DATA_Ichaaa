import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from io import BytesIO
import base64
import datetime
import importlib

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="CV Dashboard: Car/Truck & Face Detection", layout="wide")

# =========================
# KONSTAN / PATH MODEL
# =========================
CT_MODEL_PATH = "model/Annisa Humaira_Laporan 2.h5"   # TensorFlow classifier
FD_MODEL_PATH = "model/Annisa Humaira_Laporan 4.pt"   # YOLO face detector
CT_IMG_SIZE = (128, 128)
FD_CONF_THRESH = 0.50
FD_IOU_THRESH  = 0.50
FD_IMGSZ       = 640

# =========================
# UTIL BACKGROUND
# =========================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def apply_background(img_candidates):
    for p in img_candidates:
        b64 = get_base64_image(p)
        if b64:
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/jpeg;base64,{b64}");
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
            break

# =========================
# UTIL ENV CHECK
# =========================
def _can_import(modname):
    try:
        importlib.import_module(modname)
        return True, ""
    except Exception as e:
        return False, str(e)

def _cv2_is_headless():
    """Return True jika OpenCV dapat di-import (headless wheel)."""
    try:
        importlib.import_module("cv2")
        return True
    except Exception as e:
        st.error(f"OpenCV belum siap (kemungkinan non-headless): {e}")
        return False

with st.sidebar.expander("🔧 Environment check"):
    ok_tf, err_tf     = _can_import("tensorflow")
    ok_cv2, err_cv2   = _can_import("cv2")
    ok_torch, err_trc = _can_import("torch")
    st.write("tensorflow:", "✅ OK" if ok_tf else f"❌ {err_tf[:100]}...")
    st.write("cv2 (OpenCV):", "✅ OK" if ok_cv2 else f"❌ {err_cv2[:100]}...")
    st.write("torch:", "✅ OK" if ok_torch else f"❌ {err_trc[:100]}...")

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Car vs Truck", "Face Detection (Real / Sketch / Synthetic)"],
    index=0
)

# ====================================================
# =============== HALAMAN 1: CAR/TRUCK ===============
# ====================================================
if page == "Car vs Truck":
    apply_background(["bg.jpg"])
    st.markdown('<div class="title">Car or Truck Classification</div>', unsafe_allow_html=True)

    # Lazy import + cache TF model
    @st.cache_resource(show_spinner=False)
    def load_ct_model(path: str):
        try:
            import tensorflow as tf
        except Exception as e:
            st.error(f"Gagal import TensorFlow: {e}")
            return None
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            st.error(f"Gagal load model .h5: {e}")
            return None

    def ct_preprocess(img: Image.Image):
        img = img.convert("RGB").resize(CT_IMG_SIZE)
        x = np.asarray(img).astype("float32") / 255.0
        return np.expand_dims(x, 0)

    def ct_predict(img: Image.Image, model):
        x = ct_preprocess(img)
        preds = model.predict(x, verbose=0)
        p_car = float(preds.ravel()[0])
        label = "Car" if p_car >= 0.5 else "Truck"
        conf = p_car if p_car >= 0.5 else 1 - p_car
        return label, conf, p_car

    uploaded_ct = st.file_uploader(
        "Upload image (JPG/PNG) untuk klasifikasi Car/Truck",
        type=["jpg", "jpeg", "png"],
        key="ct_upload",
    )

    if uploaded_ct:
        img_ct = Image.open(uploaded_ct)
        col1, col2, col3 = st.columns([1.2, 0.8, 1.2])

        with col1:
            st.image(img_ct, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("Run Classification", use_container_width=True, key="ct_run"):
                with st.spinner("Classifying..."):
                    model = load_ct_model(CT_MODEL_PATH)
                    if not model:
                        st.stop()
                    label, conf, raw = ct_predict(img_ct, model)
                    st.session_state["ct_pred"] = {"label": label, "conf": conf, "raw": raw}

        with col3:
            pred = st.session_state.get("ct_pred")
            if pred:
                st.markdown(
                    f"""
                    <br><br>
                    <h3>Prediction: <code>{pred['label']}</code></h3>
                    <h4>Confidence: <code>{pred['conf']:.2f}</code></h4>
                    <p style="font-size:14px;color:gray;">(Raw probability = {pred['raw']:.2f})</p>
                    """,
                    unsafe_allow_html=True
                )

# ====================================================
# ============ HALAMAN 2: FACE DETECTION ============
# ====================================================
else:
    apply_background(["bg2.jpg", "bg.jpeg"])
    st.markdown('<div class="title">Face Detection: Real / Sketch / Synthetic</div>', unsafe_allow_html=True)

    # Lazy import + cache YOLO model
    @st.cache_resource(show_spinner=False)
    def load_fd_model(path: str):
        try:
            from ultralytics import YOLO
        except Exception as e:
            st.error(f"Gagal import Ultralytics/YOLO: {e}")
            return None
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"Gagal load model .pt: {e}")
            return None

    def fd_map_names(model) -> dict:
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

    def fd_annotate(res):
        bgr = res.plot()
        rgb = bgr[:, :, ::-1]
        return Image.fromarray(rgb)

    def fd_top_det(res, names):
        if res.boxes is None or len(res.boxes) == 0:
            return None, None
        confs = res.boxes.conf.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)
        i = int(np.argmax(confs))
        return names.get(clses[i], str(clses[i])), float(confs[i])

    uploaded_fd = st.file_uploader(
        "Upload image (JPG/PNG) untuk Face Detection",
        type=["jpg", "jpeg", "png"],
        key="fd_upload",
    )

    if uploaded_fd:
        img_fd = Image.open(uploaded_fd).convert("RGB")
        col1, col2, col3 = st.columns([1.2, 0.8, 1.2])

        with col1:
            st.image(img_fd, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("Run Classification", use_container_width=True, key="fd_run"):
                # cek OpenCV headless saat tombol ditekan (BUKAN di top-level)
                if not _cv2_is_headless():
                    st.stop()
                with st.spinner("Detecting..."):
                    model = load_fd_model(FD_MODEL_PATH)
                    if not model:
                        st.stop()
                    names = fd_map_names(model)
                    results = model(img_fd, conf=FD_CONF_THRESH, iou=FD_IOU_THRESH, imgsz=FD_IMGSZ, verbose=False)
                    res = results[0]
                    label, conf = fd_top_det(res, names)
                    annotated = fd_annotate(res)
                    st.session_state["fd_pred"] = {"label": label, "conf": conf, "annotated": annotated}

        with col3:
            pred = st.session_state.get("fd_pred")
            st.markdown("<br><br>", unsafe_allow_html=True)
            if pred:
                if pred["label"]:
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
                if pred.get("annotated"):
                    buf = BytesIO()
                    pred["annotated"].save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Detection Result",
                        data=buf.getvalue(),
                        file_name=f"faces_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
