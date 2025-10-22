import streamlit as st
import numpy as np
import base64
from pathlib import Path
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import cv2

# ============================
# KONFIGURASI
# ============================
st.set_page_config(page_title="Vision Dashboard: Classification & Detection", layout="wide")

# --- path & konstanta model ---
CLASSIFIER_PATH = "model/Annisa Humaira_Laporan 2.h5"   # Binary classifier: Car vs Truck
DETECTOR_PATH   = "model/Annisa Humaira_Laporan 4.pt"   # YOLO detector
IMG_SIZE = (128, 128)                                     # ukuran input classifier
CLASS_NAMES = ["Truck", "Car"]                           # Disesuaikan dengan logika output
# Catatan: model .h5 milikmu menghasilkan probabilitas "Car" tunggal (sigmoid)
# sehingga kita akan memetakan >= 0.5 => Car, < 0.5 => Truck

# ============================
# UTIL: Background Opsional
# ============================
def _get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

bg_img = _get_base64_image("bg.jpg")
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
            font-size: 44px;
            font-weight: 800;
            color: #f0f0f0;
            text-shadow: 0 2px 6px rgba(0,0,0,.35);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="title">Image Classification & Object Detection</div>', unsafe_allow_html=True)

# ============================
# LOAD MODELS (cached)
# ============================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load YOLO detector & TF classifier. Dibungkus try/except agar lebih aman."""
    yolo_model = None
    classifier = None

    # Load YOLO
    try:
        yolo_model = YOLO(DETECTOR_PATH)
    except Exception as e:
        st.error(f"Gagal memuat YOLO model: {e}")

    # Load Classifier
    try:
        classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
    except Exception as e:
        st.error(f"Gagal memuat classifier (.h5): {e}")

    return yolo_model, classifier


yolo_model, classifier = load_models()

# ============================
# PREPROCESS & PREDICT (Classifier)
# ============================
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize -> RGB -> scale 0..1 -> add batch dim."""
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(x, 0)


def predict_car_truck(img: Image.Image):
    """Kembalikan (label, confidence, raw_car_prob)."""
    if classifier is None:
        raise RuntimeError("Classifier belum termuat.")

    x = preprocess_image(img)
    preds = classifier.predict(x, verbose=0)
    # asumsi output sigmoid [p_car]
    p_car = float(np.ravel(preds)[0])
    if p_car >= 0.5:
        label = "Car"
        conf = p_car
    else:
        label = "Truck"
        conf = 1.0 - p_car
    return label, conf, p_car


# ============================
# SIDEBAR & UPLOADER
# ============================
mode = st.sidebar.selectbox(
    "Pilih Mode:",
    [
        "Deteksi Objek (YOLO)",
        "Klasifikasi Gambar (Car vs Truck)",
    ],
)

uploaded_file = st.file_uploader("Unggah gambar (jpg / jpeg / png)", type=["jpg", "jpeg", "png"])

# Tombol reset sederhana
if st.sidebar.button("🔄 Reset"):
    st.session_state.pop("prediction", None)
    st.rerun()

# ============================
# KONTEN UTAMA
# ============================
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1.1, 1.3], gap="large")

    with col1:
        st.subheader("Gambar Diupload")
        st.image(img, use_container_width=True)

        # Aksi khusus per mode
        if mode == "Klasifikasi Gambar (Car vs Truck)":
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Jalankan Klasifikasi", use_container_width=True):
                with st.spinner("Mengklasifikasi..."):
                    try:
                        label, conf, p_car = predict_car_truck(img)
                        st.session_state["prediction"] = {
                            "label": label,
                            "conf": conf,
                            "raw_car": p_car,
                        }
                    except Exception as e:
                        st.error(f"Klasifikasi gagal: {e}")

        else:  # Deteksi Objek (YOLO)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🧭 Jalankan Deteksi (YOLO)", use_container_width=True):
                if yolo_model is None:
                    st.error("Model YOLO belum termuat.")
                else:
                    with st.spinner("Mendeteksi objek..."):
                        try:
                            results = yolo_model(img)  # langsung dari PIL Image
                            r = results[0]

                            # Gambar hasil deteksi (BGR -> RGB)
                            plotted = r.plot()
                            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                            st.session_state["yolo_image"] = plotted_rgb

                            # Tabel ringkasan deteksi
                            boxes = r.boxes
                            if boxes is not None and boxes.data is not None and len(boxes) > 0:
                                xyxy = boxes.xyxy.cpu().numpy()
                                confs = boxes.conf.cpu().numpy()
                                clss = boxes.cls.cpu().numpy().astype(int)
                                names = r.names if hasattr(r, "names") else {}
                                rows = []
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
                                st.session_state["yolo_table"] = rows
                            else:
                                st.session_state["yolo_table"] = []
                        except Exception as e:
                            st.error(f"Deteksi gagal: {e}")

    with col2:
        if mode == "Klasifikasi Gambar (Car vs Truck)":
            st.subheader("Hasil Klasifikasi")
            pred = st.session_state.get("prediction")
            if pred:
                st.markdown(
                    f"""
                    <h3>Prediction: <code>{pred['label']}</code></h3>
                    <h4>Confidence: <code>{pred['conf']:.2f}</code></h4>
                    <p style="font-size:14px;color:gray;">(Raw probability Car = {pred['raw_car']:.2f})</p>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Klik tombol \"Jalankan Klasifikasi\" di kolom kiri untuk melihat hasil.")

        else:  # Deteksi Objek (YOLO)
            st.subheader("Hasil Deteksi YOLO")
            yimg = st.session_state.get("yolo_image")
            if yimg is not None:
                st.image(yimg, use_container_width=True, caption="Deteksi dengan bounding box")
            else:
                st.info("Klik tombol \"Jalankan Deteksi (YOLO)\" di kolom kiri untuk melihat hasil.")

            ytable = st.session_state.get("yolo_table")
            if ytable is not None:
                if len(ytable) == 0:
                    st.warning("Tidak ada objek yang terdeteksi.")
                else:
                    st.dataframe(ytable, use_container_width=True)

else:
    st.info("Silakan unggah gambar terlebih dahulu untuk mulai menggunakan dashboard.")
