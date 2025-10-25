# app_ui_minimal.py (dengan Nama, NPM, dan Logo USK)
# =========================================================
# Streamlit App (UI minimal, no Settings panel)
# Upload → Preview (ukuran sedang) → Jalankan Deteksi/Klasifikasi
# =========================================================
import io
import time
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
st.set_page_config(page_title="Dashboard_Annisa", layout="wide", page_icon="🪄")
if "page" not in st.session_state:
    st.session_state.page = "home"  # home | detect | classify | about | help
if "det_output" not in st.session_state:
    st.session_state.det_output = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# =========================
# IDENTITAS & MODEL PARAM
# =========================
AUTHOR_NAME = "Annisa Humaira"
AUTHOR_NPM  = "2208108010070"
LOGO_PATH   = "logo_usk.png"   # ← simpan file logo USK sebagai logo_usk.png di folder yang sama

YOLO_MODEL_PATH = "model/Annisa Humaira_Laporan 4.pt"   # Face Detection (Real/Sketch/Synthetic)
KERAS_MODEL_PATH = "model/Annisa Humaira_Laporan 2.h5"  # Car vs Truck
IMG_SIZE = (128, 128)                                   # classifier input

# --- UI sizes (px) ---
PREVIEW_WIDTH = 480   # ukuran tampilan pratinjau gambar
OUTPUT_WIDTH  = 640   # ukuran tampilan hasil anotasi/deteksi

# Default parameter (tanpa panel pengaturan)
YOLO_DEFAULT_CONF = 0.5
YOLO_DEFAULT_IOU  = 0.5
YOLO_INFER_SIZE   = 640
SHOW_DOWNLOAD_BTN = True

# =========================
# THEME & BACKGROUND
# =========================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

bg_img = ""
for cand in ["bg.jpg", "bg.jpeg"]:
    bg_img = get_base64_image(cand)
    if bg_img:
        break

PRIMARY = "#7C3AED"     # ungu-vivid
PRIMARY_DARK = "#5B21B6"
ACCENT = "#10B981"      # hijau mint
TEXT_MUTED = "#6B7280"

st.markdown(
    f"""
    <style>
    .stApp {{
        {"background-image: url('data:image/jpeg;base64," + bg_img + "');" if bg_img else ""}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .glass {{
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,.12);
        padding: 16px 22px;
    }}
    .hero {{
        background: linear-gradient(135deg, rgba(124,58,237,.85), rgba(16,185,129,.85));
        border-radius: 24px;
        padding: 36px;
        color: #fff;
        box-shadow: 0 18px 40px rgba(0,0,0,.25);
    }}
    .hero h1 {{ margin: 0 0 8px 0; font-size: 46px; font-weight: 800; }}
    .hero p  {{ margin: 0; font-size: 16px; opacity: .98; }}
    .card {{
        background: rgba(255,255,255,.96);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 10px 26px rgba(0,0,0,.12);
        transition: transform .15s ease, box-shadow .15s ease;
        height: 100%;
    }}
    .muted {{ color:{TEXT_MUTED}; font-size:14px; }}
    .footer {{ color:{TEXT_MUTED}; font-size:12px; text-align:center; margin-top:36px; }}
    /* Topbar */
    .topbar {{
        background: rgba(255,255,255,.95);
        border-radius: 14px;
        padding: 10px 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,.10);
        margin-bottom: 8px;
        display:flex; align-items:center; gap:12px;
    }}
    .topbar .title {{
        font-weight: 800; font-size: 18px; line-height:1.2;
    }}
    .topbar .sub {{
        color:{TEXT_MUTED}; font-size: 12px;
    }}
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
# TOP BAR (Logo + Nama + NPM)
# =========================
def topbar():
    logo_b64 = get_base64_image(LOGO_PATH)
    left, mid, right = st.columns([0.08, 0.62, 0.30])
    with left:
        if logo_b64:
            st.markdown(
                f"<div class='topbar' style='justify-content:center;'><img src='data:image/png;base64,{logo_b64}' height='48' /></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("<div class='topbar' style='justify-content:center;'>🏫</div>", unsafe_allow_html=True)
    with mid:
        st.markdown(
            f"""
            <div class='topbar'>
              <div>
                <div class='title'>Universitas Syiah Kuala</div>
                <div class='sub'>Fakultas Teknik — Informatika</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        st.markdown(
            f"""
            <div class='topbar' style='justify-content:flex-end;'>
              <div style='text-align:right'>
                <div class='title'>{AUTHOR_NAME}</div>
                <div class='sub'>NPM: {AUTHOR_NPM}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# NAVBAR
# =========================
def navbar():
    tabs = ["🏠 Home", "🧭 Detect", "🏷️ Classify", "ℹ️ About", "❓ Help"]
    ids  = ["home", "detect", "classify", "about", "help"]
    idx_default = ids.index(st.session_state.page) if st.session_state.page in ids else 0
    choice = st.radio("Navigation", tabs, horizontal=True, index=idx_default, label_visibility="collapsed")
    mapping = dict(zip(tabs, ids))
    st.session_state.page = mapping[choice]

# =========================
# PAGES
# =========================
def page_home():
    st.markdown(
        """
        <div class="hero">
          <h1>Dual Vision Dashboard</h1>
          <p>Deteksi wajah & klasifikasi kendaraan — cepat, ringan, dan mudah.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><h3>🧭 Face Detection</h3><p class='muted'>Deteksi wajah (Real/Sketch/Synthetic).</p></div>", unsafe_allow_html=True)
        if st.button("Mulai Deteksi", use_container_width=True):
            st.session_state.page = "detect"
    with col2:
        st.markdown("<div class='card'><h3>🏷️ Car vs Truck</h3><p class='muted'>Klasifikasi kendaraan (Car atau Truck).</p></div>", unsafe_allow_html=True)
        if st.button("Mulai Klasifikasi", use_container_width=True):
            st.session_state.page = "classify"

def page_detect():
    st.markdown("### 🧭 Face Detection — Real / Sketch / Synthetic")
    uploaded = st.file_uploader("📤 Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="up_det")
    if uploaded:
        img_preview = Image.open(uploaded).convert("RGB")
        st.image(img_preview, caption="Pratinjau Gambar", width=PREVIEW_WIDTH, use_container_width=False)
    else:
        st.info("Upload gambar untuk memulai deteksi.")
        return

    run = st.button("🔎 Jalankan Deteksi", use_container_width=True)
    if run:
        try:
            start = time.time()
            with st.spinner("Detecting faces..."):
                model = load_yolo_model(YOLO_MODEL_PATH)
                names = get_class_names(model)
                results = model(img_preview, conf=YOLO_DEFAULT_CONF, iou=YOLO_DEFAULT_IOU, imgsz=YOLO_INFER_SIZE)
                result = results[0]
                annotated = draw_and_get_image(result)
                detections = summarize_counts(result, names)
            elapsed = time.time() - start
            st.session_state.det_output = {"annotated": annotated, "detections": detections, "elapsed": elapsed}
        except Exception as e:
            st.error(f"Error: {e}")

    out = st.session_state.det_output
    if out:
        st.image(out["annotated"], caption="🖼️ Detections", width=OUTPUT_WIDTH, use_container_width=False)
        if SHOW_DOWNLOAD_BTN:
            buf = io.BytesIO()
            out["annotated"].save(buf, format="PNG")
            filename = f"faces_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            st.download_button("⬇️ Download annotated image", buf.getvalue(), file_name=filename, mime="image/png")
        if out["detections"]:
            st.markdown("#### 📊 Hasil Deteksi")
            for label, conf in out["detections"]:
                st.markdown(f"- **{label}** — `{conf:.2f}`")
        else:
            st.info("Tidak ada wajah terdeteksi.")

def page_classify():
    st.markdown("### 🏷️ Car vs Truck Classification")
    uploaded = st.file_uploader("📤 Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="up_cls")
    if uploaded:
        img_preview = Image.open(uploaded)
        st.image(img_preview, caption="Pratinjau Gambar", width=PREVIEW_WIDTH, use_container_width=False)
    else:
        st.info("Upload gambar untuk memulai klasifikasi.")
        return

    run = st.button("🧠 Jalankan Klasifikasi", use_container_width=True)
    if run:
        try:
            start = time.time()
            with st.spinner("Classifying..."):
                model = load_keras_model()
                label, conf, raw_car = predict_car_truck(img_preview, model)
            elapsed = time.time() - start
            st.session_state.prediction = {"label": label, "conf": conf, "raw_car": raw_car, "elapsed": elapsed}
        except Exception as e:
            st.error(f"Error: {e}")

    pred = st.session_state.prediction
    if pred:
        st.image(img_preview, caption=f"Predicted: {pred['label']} ({pred['conf']:.2f})", width=OUTPUT_WIDTH, use_container_width=False)
        st.markdown(f"**Confidence:** {pred['conf']:.2f}")
        st.markdown(f"**Raw prob (Car):** {pred['raw_car']:.2f}")
        st.caption(f"Latency: {pred['elapsed']:.2f}s")

def page_about():
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.info("Aplikasi sederhana untuk deteksi wajah (YOLOv8) dan klasifikasi kendaraan (Keras).")
    st.markdown(f"**Disusun oleh:** {AUTHOR_NAME}  \n**NPM:** {AUTHOR_NPM}  \n**Institusi:** Universitas Syiah Kuala")

def page_help():
    st.markdown("### ❓ Panduan Penggunaan")
    st.markdown("1️⃣ Masuk ke halaman **Detect** atau **Classify**.\n\n2️⃣ Upload gambar (JPG/PNG).\n\n3️⃣ Klik tombol proses.\n\n4️⃣ Lihat hasil & confidence.")

# =========================
# RENDER
# =========================
topbar()          # ← bar dengan logo USK + nama + NPM
navbar()

page = st.session_state.page
if page == "home":
    page_home()
elif page == "detect":
    page_detect()
elif page == "classify":
    page_classify()
elif page == "about":
    page_about()
else:
    page_help()

st.markdown(
    f"<div class='footer'>© {datetime.datetime.now().year} — {AUTHOR_NAME} • NPM {AUTHOR_NPM} • Universitas Syiah Kuala</div>",
    unsafe_allow_html=True
)
