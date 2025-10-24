# app_ui_minimal.py
# =========================================================
# Streamlit App (UI minimal, no Settings panel)
# - Landing Page -> (Face Detection | Car/Truck Classification) + About & Help
# - Gambar selalu ditampilkan sebelum proses
# =========================================================
# pip install streamlit ultralytics tensorflow pillow numpy opencv-python
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
# MODEL PATH & PARAM
# =========================
YOLO_MODEL_PATH = "model/Annisa Humaira_Laporan 4.pt"   # Face Detection (Real/Sketch/Synthetic)
KERAS_MODEL_PATH = "model/Annisa Humaira_Laporan 2.h5"  # Car vs Truck
IMG_SIZE = (128, 128)                                   # classifier input

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
    .chip {{
        display:inline-block; padding:7px 12px; border-radius:999px;
        background:#F3F4F6; color:#374151; font-weight:600; font-size:12px;
        margin-right:6px;
    }}
    .card {{
        background: rgba(255,255,255,.96);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 10px 26px rgba(0,0,0,.12);
        transition: transform .15s ease, box-shadow .15s ease;
        height: 100%;
    }}
    .card:hover {{ transform: translateY(-2px); box-shadow: 0 14px 30px rgba(0,0,0,.16); }}
    .muted {{ color:{TEXT_MUTED}; font-size:14px; }}
    .footer {{ color:{TEXT_MUTED}; font-size:12px; text-align:center; margin-top:36px; }}
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
# NAVBAR (Streamlit native)
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
          <div class="chip">🚀 Dual Vision Dashboard</div>
          <h1>Deteksi Objek & Klasifikasi Gambar</h1>
          <p>Tanpa panel pengaturan. Upload ➜ lihat pratinjau ➜ jalankan.</p>
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
              <h3>🧭 Face Detection</h3>
              <p class="muted">Deteksi wajah (Real / Sketch / Synthetic) dengan YOLOv8 (.pt). 
              Hasil anotasi siap diunduh.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Mulai Deteksi", use_container_width=True, key="home_detect"):
            st.session_state.page = "detect"
    with c2:
        st.markdown(
            """
            <div class="card">
              <h3>🏷️ Car vs Truck</h3>
              <p class="muted">Klasifikasi kendaraan (Keras .h5). Tampilkan label & confidence.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Mulai Klasifikasi", use_container_width=True, key="home_classify"):
            st.session_state.page = "classify"

def page_detect():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 🧭 Face Detection — Real / Sketch / Synthetic")
    st.caption(f"Model: `{YOLO_MODEL_PATH}`")

    uploaded = st.file_uploader("📤 Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="up_det")
    st.markdown('</div>', unsafe_allow_html=True)

    # Selalu tampilkan pratinjau gambar sebelum deteksi
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    if uploaded:
        img_preview = Image.open(uploaded).convert("RGB")
        st.image(img_preview, caption="Pratinjau Gambar", use_container_width=True)
    else:
        st.info("Upload gambar untuk memulai deteksi.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    colGo, colBack = st.columns([0.7, 0.3])
    with colGo:
        run = st.button("🔎 Jalankan Deteksi", use_container_width=True, type="primary", disabled=uploaded is None)
    with colBack:
        back = st.button("🏠 Kembali ke Home", use_container_width=True)
    if back:
        st.session_state.page = "home"

    if uploaded and run:
        if not _HAS_ULTRA:
            st.error(f"Ultralytics/YOLO belum tersedia. Detail: `{_ULTRA_ERR}`")
        else:
            try:
                start = time.time()
                with st.spinner("Detecting faces..."):
                    model = load_yolo_model(YOLO_MODEL_PATH)
                    names = get_class_names(model)
                    results = model(
                        img_preview,
                        conf=YOLO_DEFAULT_CONF,
                        iou=YOLO_DEFAULT_IOU,
                        imgsz=YOLO_INFER_SIZE,
                        verbose=False
                    )
                    result = results[0]
                    annotated = draw_and_get_image(result)
                    detections = summarize_counts(result, names)
                elapsed = time.time() - start
                st.session_state.det_output = {"annotated": annotated, "detections": detections, "elapsed": elapsed}
            except Exception as e:
                st.error(f"Gagal menjalankan detection: {e}")

    out = st.session_state.det_output
    if out:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### 📊 Hasil Deteksi")
        c1, c2 = st.columns([1.25, 1.0], gap="large")
        with c1:
            st.image(out["annotated"], caption="🖼️ Detections", use_container_width=True)
            if SHOW_DOWNLOAD_BTN:
                buf = io.BytesIO()
                out["annotated"].save(buf, format="PNG")
                filename = f"faces_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                st.download_button("⬇️ Download annotated image", buf.getvalue(), file_name=filename, mime="image/png", use_container_width=True)
        with c2:
            count = len(out["detections"]) if out["detections"] else 0
            mcol = st.columns(3)
            mcol[0].metric("Detections", count)
            mcol[1].metric("ImgSize", YOLO_INFER_SIZE)
            mcol[2].metric("Latency (s)", f"{out['elapsed']:.2f}")
            st.write("")
            st.markdown("#### 🔖 Detail")
            if out["detections"]:
                for i, (label, conf) in enumerate(out["detections"], start=1):
                    st.markdown(f"{i}. **{label}** — `{conf:.2f}`")
            else:
                st.info("Tidak ada wajah terdeteksi.")
        st.markdown('</div>', unsafe_allow_html=True)

def page_classify():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 🏷️ Car vs Truck Classification")
    st.caption(f"Model: `{KERAS_MODEL_PATH}`")
    uploaded = st.file_uploader("📤 Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="up_cls")
    st.markdown('</div>', unsafe_allow_html=True)

    # Selalu tampilkan pratinjau gambar sebelum klasifikasi
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    if uploaded:
        img_preview = Image.open(uploaded)
        st.image(img_preview, caption="Pratinjau Gambar", use_container_width=True)
    else:
        st.info("Upload gambar untuk memulai klasifikasi.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    colGo, colBack = st.columns([0.7, 0.3])
    with colGo:
        run = st.button("🧠 Jalankan Klasifikasi", use_container_width=True, type="primary", disabled=uploaded is None)
    with colBack:
        back = st.button("🏠 Kembali ke Home", use_container_width=True)
    if back:
        st.session_state.page = "home"

    if uploaded and run:
        if not _HAS_TF:
            st.error(f"TensorFlow/Keras belum tersedia. Detail: `{_TF_ERR}`")
        else:
            try:
                start = time.time()
                with st.spinner("Classifying..."):
                    model = load_keras_model()
                    label, conf, raw_car = predict_car_truck(img_preview, model)
                elapsed = time.time() - start
                st.session_state.prediction = {"label": label, "conf": conf, "raw_car": raw_car, "elapsed": elapsed}
            except Exception as e:
                st.error(f"Gagal menjalankan klasifikasi: {e}")

    pred = st.session_state.prediction
    if pred:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### 📊 Hasil Klasifikasi")
        c1, c2 = st.columns([1.2, 1.0], gap="large")
        with c1:
            st.markdown(
                f"""
                <div class="card">
                  <h3>🎯 Prediction: <code>{pred['label']}</code></h3>
                  <h4>Confidence: <code>{pred['conf']:.2f}</code></h4>
                  <p class="muted">Raw probability for <b>Car</b> = {pred['raw_car']:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            mcol = st.columns(3)
            mcol[0].metric("Is Car?", f"{pred['raw_car']:.2f}")
            mcol[1].metric("Is Truck?", f"{1.0 - pred['raw_car']:.2f}")
            mcol[2].metric("Latency (s)", f"{pred['elapsed']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

def page_about():
    st.markdown(
        """
        <div class="hero">
          <div class="chip">ℹ️ About</div>
          <h1>Tentang Aplikasi</h1>
          <p>Aplikasi ini terdiri dari dua fitur utama: <b>Face Detection</b> dan
             <b>Car vs Truck Classification</b>. Dirancang untuk demo cepat, praktikum, dan eksplorasi.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown(
        f"""
        <div class="card">
          <h3>🔧 Teknologi</h3>
          <ul>
            <li>Ultralytics YOLOv8 (.pt) — Object/Face Detection</li>
            <li>TensorFlow/Keras (.h5) — Image Classification</li>
            <li>Streamlit — UI cepat dan ringan</li>
          </ul>
          <h3>📁 Struktur Model</h3>
          <ul>
            <li>YOLO path: <code>{YOLO_MODEL_PATH}</code></li>
            <li>Keras path: <code>{KERAS_MODEL_PATH}</code></li>
          </ul>
          <h3>🧪 Saran Pengujian</h3>
          <ul>
            <li>Pakai gambar dengan objek jelas & resolusi memadai.</li>
            <li>Hindari kompresi berlebihan atau noise tinggi.</li>
            <li>Uji variasi pose/pencahayaan untuk deteksi wajah.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

def page_help():
    st.markdown(
        """
        <div class="hero">
          <div class="chip">❓ Help</div>
          <h1>Panduan Singkat</h1>
          <p>Ikuti langkah-langkah untuk memakai fitur yang kamu butuhkan.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown(
        """
        <div class="card">
          <h3>🧭 Deteksi Wajah</h3>
          <ol>
            <li>Masuk ke halaman <b>Detect</b>.</li>
            <li>Upload gambar (JPG/PNG) ➜ pratinjau muncul.</li>
            <li>Klik <b>Jalankan Deteksi</b> untuk melihat hasil.</li>
          </ol>
          <h3>🏷️ Klasifikasi Kendaraan</h3>
          <ol>
            <li>Masuk ke halaman <b>Classify</b>.</li>
            <li>Upload gambar (JPG/PNG) ➜ pratinjau muncul.</li>
            <li>Klik <b>Jalankan Klasifikasi</b> untuk melihat hasil.</li>
          </ol>
          <h3>🛠️ Troubleshooting</h3>
          <ul>
            <li><b>Model tidak ditemukan:</b> pastikan file ada di path yang benar.</li>
            <li><b>ModuleNotFoundError:</b> jalankan <code>pip install streamlit ultralytics tensorflow pillow numpy opencv-python</code>.</li>
            <li><b>Gambar tidak muncul:</b> pastikan format JPG/PNG dan file tidak corrupt.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# RENDER
# =========================
def navbar():
    tabs = ["🏠 Home", "🧭 Detect", "🏷️ Classify", "ℹ️ About", "❓ Help"]
    ids  = ["home", "detect", "classify", "about", "help"]
    idx_default = ids.index(st.session_state.page) if st.session_state.page in ids else 0
    choice = st.radio("Navigation", tabs, horizontal=True, index=idx_default, label_visibility="collapsed")
    mapping = dict(zip(tabs, ids))
    st.session_state.page = mapping[choice]

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

st.markdown(f'<div class="footer">© {datetime.datetime.now().year} Dual Vision — built with ❤️ using Streamlit</div>', unsafe_allow_html=True)
