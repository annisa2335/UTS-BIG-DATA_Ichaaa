import io
import base64
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

# ------------------ Optional imports with safe fallback ------------------
_HAS_ULTRA = True
try:
    from ultralytics import YOLO
except Exception as e:
    _HAS_ULTRA = False
    _ULTRA_ERR = e

_HAS_TF = True
try:
    import tensorflow as tf
except Exception as e:
    _HAS_TF = False
    _TF_ERR = e

# =========================
# KONFIG & STATE
# =========================
st.set_page_config(page_title="Dashboard ...", layout="wide")
if "page" not in st.session_state:
    st.session_state.page = "home"  # home | detect | classify
if "det_output" not in st.session_state:
    st.session_state.det_output = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# =========================
# MODEL PATH & PARAM
# =========================
YOLO_MODEL_PATH = "model/Annisa Humaira_Laporan 4.pt"   # Face Detection (Real/Sketch/Synthetic)
KERAS_MODEL_PATH = "model/Annisa Humaira_Laporan 2.h5"  # Car vs Truck
IMG_SIZE = (128, 128)  # classifier input

# =========================
# BACKGROUND & THEME CSS
# =========================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

bg_img = ""
for cand in ["bg.jpg"]:
    bg_img = get_base64_image(cand)
    if bg_img:
        break

st.markdown(
    f"""
    <style>
    .stApp {{
        {"background-image: url('data:image/jpeg;base64," + bg_img + "');" if bg_img else ""}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .hero {{
        background: rgba(0,0,0,0.55);
        border-radius: 24px;
        padding: 32px;
        color: #f7f7f7;
        box-shadow: 0 10px 30px rgba(0,0,0,.25);
    }}
    .hero h1 {{
        font-size: 44px; margin: 0 0 8px 0; font-weight: 800;
    }}
    .hero p {{
        font-size: 16px; opacity: .95; margin: 0;
    }}
    .card {{
        background: rgba(255,255,255,.92);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
        transition: transform .2s ease, box-shadow .2s ease;
        height: 100%;
    }}
    .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(0,0,0,.18);
    }}
    .card h3 {{ margin-top: 0; }}
    .pill {{
        display:inline-block; padding:6px 12px; border-radius:999px;
        background:#EDF2FF; color:#4C6EF5; font-weight:600; font-size:12px;
        margin-bottom:10px;
    }}
    .btn-primary {{
        background:#4C6EF5; color:white; padding:10px 16px; border-radius:12px;
        text-decoration:none; font-weight:700; display:inline-block; border:0;
    }}
    .btn-ghost {{
        background:transparent; color:#4C6EF5; padding:10px 16px; border-radius:12px;
        border:1.5px solid #4C6EF5; text-decoration:none; font-weight:700; display:inline-block;
    }}
    .muted {{ color:#6b7280; font-size:14px; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HELPERS: Detection (YOLO)
# =========================
@st.cache_resource(show_spinner=False)
def load_yolo_model(path: str):
    if not _HAS_ULTRA:
        raise RuntimeError(f"Ultralytics/YOLO belum tersedia: {_ULTRA_ERR}")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model YOLO tidak ditemukan: {path}")
    return YOLO(path)

def get_class_names(model) -> dict:
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
    bgr = result.plot()
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)

def summarize_counts(result, names: dict):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    return [(names.get(cid, str(cid)), float(conf)) for cid, conf in zip(cls_ids, confs)]

# =========================
# HELPERS: Classification (Keras)
# =========================
@st.cache_resource
def load_keras_model():
    if not _HAS_TF:
        raise RuntimeError(f"TensorFlow/Keras belum tersedia: {_TF_ERR}")
    p = Path(KERAS_MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Model Keras tidak ditemukan: {KERAS_MODEL_PATH}")
    return tf.keras.models.load_model(KERAS_MODEL_PATH)

def preprocess_image(img: Image.Image, size=(128, 128)):
    img = img.convert("RGB").resize(size)
    x = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(x, 0)

def predict_car_truck(img: Image.Image, model):
    x = preprocess_image(img, size=IMG_SIZE)
    preds = model.predict(x, verbose=0)
    p_car = float(preds.ravel()[0])
    if p_car >= 0.5:
        label, conf = "Car", p_car
    else:
        label, conf = "Truck", 1.0 - p_car
    return label, conf, p_car

# =========================
# ROUTING (Home / Detect / Classify)
# =========================
def go(page_name: str):
    st.session_state.page = page_name
    # reset output saat berpindah halaman
    if page_name != "detect":
        st.session_state.det_output = None
    if page_name != "classify":
        st.session_state.prediction = None

# ========== HOME ==========
def page_home():
    st.markdown(
        """
        <div class="hero">
            <h1>Dashboard ...</h1>
            <p>Dashboard untuk <b>Deteksi Objek</b> dan <b>Klasifikasi Gambar</b>.
               Aplikasi ini dirancang ringan, cepat, dan mudah dipakai. Pilih mode yang kamu butuhkan </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="card">
                <h3>Face Detection</h3>
                <p class="muted">Untuk mendeteksi wajah (real, sketch, synthetic) dengan hasil anotasi siap unduh.</p>
                """,
            unsafe_allow_html=True
        )
        if st.button("→ Mulai Deteksi Objek", use_container_width=True):
            go("detect")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            """
            <div class="card">
                <h3>Car vs Truck Classification</h3>
                <p class="muted">Untuk mengklasifikasi gambar kendaraan menjadi
                <i>Car</i> atau <i>Truck</i>.</p>
            """,
            unsafe_allow_html=True
        )
        if st.button("→ Mulai Klasifikasi Gambar", use_container_width=True):
            go("classify")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

# ========== DETECTION ==========
def page_detect():
    st.markdown("### 🔎 Face Detection — Real / Sketch / Synthetic")

    uploaded = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="up_det")
    if st.button("← Kembali ke Dashboard", type="secondary"):
        go("home")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1.25, 1.0], gap="large")
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Run Detection", use_container_width=True):
                if not _HAS_ULTRA:
                    st.error(f"Ultralytics/YOLO belum tersedia. Detail: `{_ULTRA_ERR}`")
                else:
                    try:
                        with st.spinner("Detecting faces..."):
                            model = load_yolo_model(YOLO_MODEL_PATH)
                            names = get_class_names(model)
                            results = model(img, conf=st.session_state.conf_det, iou=st.session_state.iou_det,
                                            imgsz=st.session_state.imgsz_det, verbose=False)
                            result = results[0]
                            annotated = draw_and_get_image(result)
                            detections = summarize_counts(result, names)
                        st.session_state.det_output = {"annotated": annotated, "detections": detections}
                    except Exception as e:
                        st.error(f"Gagal menjalankan detection: {e}")

    out = st.session_state.det_output
    if out:
        colA, colB = st.columns([1.25, 1.0], gap="large")
        with colA:
            st.image(out["annotated"], caption="Detections", use_container_width=True)
            if st.session_state.show_dl:
                buf = io.BytesIO()
                out["annotated"].save(buf, format="PNG")
                filename = f"faces_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                st.download_button("⬇️ Download annotated image", buf.getvalue(), file_name=filename, mime="image/png", use_container_width=True)
        with colB:
            st.markdown("#### Detection Result")
            if out["detections"]:
                for label, conf in out["detections"]:
                    st.markdown(f"- **{label}** — Confidence: `{conf:.2f}`")
            else:
                st.info("No faces detected.")

# ========== CLASSIFICATION ==========
def page_classify():
    st.markdown("### 🏷️ Car vs Truck Classification")

    uploaded = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="up_cls")
    if st.button("← Kembali ke Dashboard", type="secondary"):
        go("home")

    if uploaded:
        img = Image.open(uploaded)
        col1, col2, col3 = st.columns([1.2, 0.8, 1.2], gap="large")
        with col1:
            st.image(img, use_container_width=True, caption="Uploaded Image")
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("Run Classification", use_container_width=True):
                if not _HAS_TF:
                    st.error(f"TensorFlow/Keras belum tersedia. Detail: `{_TF_ERR}`")
                else:
                    try:
                        with st.spinner("Classifying..."):
                            model = load_keras_model()
                            label, conf, raw_car = predict_car_truck(img, model)
                        st.session_state.prediction = {"label": label, "conf": conf, "raw_car": raw_car}
                    except Exception as e:
                        st.error(f"Gagal menjalankan klasifikasi: {e}")
        with col3:
            pred = st.session_state.prediction
            if pred:
                st.markdown(
                    f"""
                    <br><br>
                    <div class="card">
                      <h3>Prediction: <code>{pred['label']}</code></h3>
                      <h4>Confidence: <code>{pred['conf']:.2f}</code></h4>
                      <p class="muted">(Raw probability for <b>Car</b> = {pred['raw_car']:.2f})</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================
# RENDER
# =========================
st.empty()  # anchor
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "detect":
    page_detect()
else:
    page_classify()
