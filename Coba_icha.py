# streamlit_app.py
import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# =========================================
# KONFIGURASI APLIKASI
# =========================================
st.set_page_config(page_title="Vision Dashboard — Icha", layout="wide")

# ==== PATH MODEL (GANTI JIKA PERLU) ====
YOLO_PATH = "model/Annisa Humaira_Laporan 4.pt"     # YOLO detector
H5_PATH   = "model/Annisa Humaira_Laporan 2.h5"     # Classifier Car vs Truck (sigmoid 1-neuron)

# Ukuran input classifier (kode awalmu)
IMG_SIZE = (128, 128)

# =========================================
# UTIL
# =========================================
def _file_exists(p: str) -> bool:
    return Path(p).exists()

@st.cache_resource(show_spinner=False)
def load_models():
    """Load YOLO & Keras classifier (cached)."""
    yolo_model = None
    classifier = None

    # YOLO
    try:
        if not _file_exists(YOLO_PATH):
            raise FileNotFoundError(f"File YOLO tidak ditemukan: {YOLO_PATH}")
        yolo_model = YOLO(YOLO_PATH)
    except Exception as e:
        st.error(f"❌ Gagal memuat YOLO: {e}")

    # Classifier
    try:
        if not _file_exists(H5_PATH):
            raise FileNotFoundError(f"File classifier (.h5) tidak ditemukan: {H5_PATH}")
        classifier = tf.keras.models.load_model(H5_PATH)
    except Exception as e:
        st.error(f"❌ Gagal memuat classifier: {e}")

    return yolo_model, classifier

yolo_model, classifier = load_models()

def preprocess_img_for_classifier(img: Image.Image) -> np.ndarray:
    """RGB -> resize -> scale 0..1 -> add batch dim."""
    img = img.convert("RGB").resize(IMG_SIZE)
    x = image.img_to_array(img)  # float32
    x = x / 255.0
    return np.expand_dims(x, axis=0)

def predict_car_truck(img: Image.Image):
    """Return (label, confidence, raw_car_prob). Output diasumsikan sigmoid p(Car)."""
    if classifier is None:
        raise RuntimeError("Classifier belum termuat.")
    X = preprocess_img_for_classifier(img)
    preds = classifier.predict(X, verbose=0)
    p_car = float(np.ravel(preds)[0])  # sigmoid
    if p_car >= 0.5:
        label = "Car"
        conf = p_car
    else:
        label = "Truck"
        conf = 1.0 - p_car
    return label, conf, p_car

def yolo_detect(pil_img: Image.Image):
    """Jalankan deteksi YOLO pada PIL image dan kembalikan (img_berbox_RGB, rows_table)."""
    if yolo_model is None:
        raise RuntimeError("YOLO belum termuat.")
    results = yolo_model(pil_img)
    r = results[0]

    # Gambar hasil deteksi
    plotted = r.plot()                          # BGR (numpy)
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    # Tabel ringkas
    rows = []
    boxes = r.boxes
    names = r.names if hasattr(r, "names") else {}
    if boxes is not None and boxes.data is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss  = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(confs)):
            label = names.get(clss[i], str(clss[i])) if isinstance(names, dict) else str(clss[i])
            rows.append({
                "label": label,
                "conf": float(confs[i]),
                "x1": float(xyxy[i][0]),
                "y1": float(xyxy[i][1]),
                "x2": float(xyxy[i][2]),
                "y2": float(xyxy[i][3]),
            })
    return plotted_rgb, rows

# =========================================
# SIDEBAR — MENU
# =========================================
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Classification", "Detection"],
        icons=["house", "car-front", "bounding-box"],
        menu_icon="ui-checks-grid",
        default_index=0,
    )
    st.markdown("---")
    if st.button("🔄 Reset"):
        for k in ["pred", "det_image", "det_rows"]:
            st.session_state.pop(k, None)
        st.rerun()

# =========================================
# HALAMAN: HOME
# =========================================
if selected == "Home":
    st.image(
        "https://tse2.mm.bing.net/th/id/OIP.kA2kMOzGD95g9evKDh5JsgAAAA?cb=12&rs=1&pid=ImgDetMain&o=7&rm=3",
        width=150,
    )
    st.title("Universitas Syiah Kuala")
    st.subheader("Praktikum Big Data — Vision Dashboard")

    st.write("**Nama:** Annisa Humaira (Icha)")
    st.write("**Topik:** Image Classification (Car vs Truck) & Object Detection (YOLO)")

    st.markdown("""
    Selamat datang 👋  
    - 🚗 **Car vs Truck Classifier** (model `.h5`)  
    - 📦 **Object Detection** (YOLO `.pt`)  
    """)

    st.markdown("""
    ---
    ### 🧭 Cara Menggunakan:
    1. Pilih menu di **sidebar**:
       - **Classification** → klasifikasi Car/Truck.
       - **Detection** → deteksi objek & tampilkan bounding box.
    2. **Unggah gambar** (JPG/PNG).
    3. Klik tombol untuk menjalankan model.
    4. Lihat hasil & confidence.
    ---
    """)
    st.info("💡 Tips: Gambar jelas & tajam membantu model memberi hasil lebih akurat.")

# =========================================
# HALAMAN: CLASSIFICATION
# =========================================
elif selected == "Classification":
    st.markdown("<h2 style='text-align:center;'>🚗 Car vs Truck Classifier</h2>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Unggah gambar kendaraan", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar diunggah", use_column_width=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            run_cls = st.button("🚀 Jalankan Klasifikasi", use_container_width=True)
        with col2:
            back_btn = st.button("⬅️ Kembali ke Beranda", use_container_width=True)

        if back_btn:
            st.experimental_rerun()

        if run_cls:
            st.write("🔍 Mengklasifikasi...")
            try:
                label, conf, p_car = predict_car_truck(img)
                st.session_state["pred"] = {"label": label, "conf": conf, "p_car": p_car}
            except Exception as e:
                st.error(f"❌ Klasifikasi gagal: {e}")

    # Panel hasil
    pred = st.session_state.get("pred")
    if pred:
        st.progress(int(pred["conf"] * 100))
        st.write(f"**Prediksi:** {pred['label']}")
        st.write(f"**Confidence:** {pred['conf']:.2%}")
        st.caption(f"(Raw probability Car = {pred['p_car']:.4f})")

        if pred["label"] == "Car":
            st.markdown("""
                <div style='background-color:#e9f7ef; padding:16px; border-radius:12px;'>
                🚘 <b>Car terdeteksi</b> — akses jalur hijau.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color:#fef5e7; padding:16px; border-radius:12px;'>
                🚚 <b>Truck terdeteksi</b> — arahkan ke jalur logistik.
                </div>
            """, unsafe_allow_html=True)

# =========================================
# HALAMAN: DETECTION
# =========================================
elif selected == "Detection":
    st.markdown("<h2 style='text-align:center;'>📦 Object Detection (YOLO)</h2>", unsafe_allow_html=True)
    uploaded_det = st.file_uploader("Unggah gambar untuk deteksi objek", type=["jpg", "jpeg", "png"])
    if uploaded_det:
        img_det = Image.open(uploaded_det)
        st.image(img_det, caption="Gambar diunggah", use_column_width=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            run_det = st.button("🧭 Jalankan Deteksi (YOLO)", use_container_width=True)
        with col2:
            back_btn = st.button("⬅️ Kembali ke Beranda", use_container_width=True)

        if back_btn:
            st.experimental_rerun()

        if run_det:
            st.write("🔎 Mendeteksi...")
            try:
                det_image, det_rows = yolo_detect(img_det)
                st.session_state["det_image"] = det_image
                st.session_state["det_rows"] = det_rows
            except Exception as e:
                st.error(f"❌ Deteksi gagal: {e}")

    # Panel hasil
    if st.session_state.get("det_image") is not None:
        st.image(st.session_state["det_image"], use_container_width=True, caption="Hasil Deteksi (with bounding box)")
    det_rows = st.session_state.get("det_rows")
    if det_rows is not None:
        if len(det_rows) == 0:
            st.warning("⚠️ Tidak ada objek yang terdeteksi.")
        else:
            st.dataframe(det_rows, use_container_width=True)
