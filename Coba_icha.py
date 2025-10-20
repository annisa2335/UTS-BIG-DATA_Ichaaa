import os
import numpy as np
import streamlit as st
from PIL import Image

# Keras (wajah)
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# YOLOv8 classification (kendaraan)
from ultralytics import YOLO

# ==========================
# Konfigurasi & utilitas
# ==========================
FACE_MODEL_PATH = "/mnt/data/Annisa Humaira_Laporan 2.h5"
VEHICLE_MODEL_PATH = "/mnt/data/Annisa Humaira_Laporan 4.pt"
FACE_INPUT_SIZE = (224, 224)  # sesuaikan dgn arsitektur wajahmu

# Mapping label (ubah sesuai urutan output model kamu)
FACE_CLASS_NAMES = ["Real Faces", "Sketch Faces", "Synthetic Faces"]  # contoh
# Untuk kendaraan, YOLO akan punya model.names; jika None, pakai fallback:
VEHICLE_FALLBACK_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle"]  # contoh

def assert_exists(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"File model {label} tidak ditemukan di: {path}")
        st.stop()

def topk_from_vector(vec, names, k=5):
    idxs = np.argsort(-vec)[:k]
    rows = []
    for i in idxs:
        name = names[i] if (names is not None and i < len(names)) else f"class_{i}"
        rows.append((int(i), name, float(vec[i])))
    return rows

# ==========================
# Cache load model
# ==========================
@st.cache_resource
def load_face_model():
    assert_exists(FACE_MODEL_PATH, "wajah (.h5)")
    # Tambahkan custom_objects jika modelmu punya layer/metric kustom
    model = tf.keras.models.load_model(FACE_MODEL_PATH)
    return model

@st.cache_resource
def load_vehicle_model():
    assert_exists(VEHICLE_MODEL_PATH, "kendaraan (.pt)")
    model = YOLO(VEHICLE_MODEL_PATH)  # diasumsikan YOLOv8 classification
    # Jika ini deteksi, results[0].probs kemungkinan None → ditangani saat inferensi
    return model

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification App — Wajah & Kendaraan")

mode = st.sidebar.selectbox(
    "Pilih Mode Klasifikasi:",
    ["Klasifikasi Wajah (Keras .h5)", "Klasifikasi Kendaraan (YOLOv8 .pt)"]
)

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Paksa RGB 3 kanal agar aman untuk kedua model
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if mode == "Klasifikasi Wajah (Keras .h5)":
        face_model = load_face_model()

        # Preprocess
        img_resized = img.resize(FACE_INPUT_SIZE)
        x = image.img_to_array(img_resized)        # (H, W, 3)
        x = np.expand_dims(x, axis=0)              # (1, H, W, 3)
        x = x / 255.0

        # Predict
        preds = face_model.predict(x)
        if preds.ndim == 2 and preds.shape[0] == 1:
            probs = preds[0]
        else:
            # Antisipasi output yang aneh (mis. sigmoid 1-unit, dll.)
            probs = np.ravel(preds)
        top_idx = int(np.argmax(probs))
        top_prob = float(np.max(probs))
        top_label = (
            FACE_CLASS_NAMES[top_idx]
            if 0 <= top_idx < len(FACE_CLASS_NAMES)
            else f"Class {top_idx}"
        )

        st.subheader("Hasil Prediksi (Top-1)")
        st.write(f"**Label:** {top_label}")
        st.write(f"**Probabilitas:** {top_prob:.4f}")

        # Tampilkan Top-5 (jika kelas > 1)
        if probs.size > 1:
            st.markdown("**Top-5 Probabilities**")
            rows = topk_from_vector(probs, FACE_CLASS_NAMES, k=min(5, probs.size))
            st.table({"Index": [r[0] for r in rows],
                      "Label": [r[1] for r in rows],
                      "Prob": [f"{r[2]:.4f}" for r in rows]})

    elif mode == "Klasifikasi Kendaraan (YOLOv8 .pt)":
        vehicle_model = load_vehicle_model()

        # YOLOv8 cls bisa langsung terima array/Path. Kita pakai ndarray
        img_np = np.array(img)  # RGB
        results = vehicle_model(img_np, verbose=False)
        if not results:
            st.error("Tidak ada output dari model. Periksa file model kendaraan.")
            st.stop()

        r0 = results[0]
        # Jika ini BUKAN model klasifikasi (mis. deteksi), probs akan None
        if getattr(r0, "probs", None) is None:
            st.error(
                "Model .pt yang dimuat tampaknya bukan model klasifikasi (YOLOv8-cls). "
                "Gunakan file YOLOv8-classification (.pt) agar fitur probabilitas tersedia."
            )
            st.stop()

        probs_vec = r0.probs.data.cpu().numpy() if hasattr(r0.probs, "data") else np.asarray(r0.probs)
        names = getattr(vehicle_model, "names", None)
        if not names:
            names = VEHICLE_FALLBACK_NAMES  # fallback contoh

        top_idx = int(np.argmax(probs_vec))
        top_prob = float(np.max(probs_vec))
        top_label = names[top_idx] if top_idx < len(names) else f"class_{top_idx}"

        st.subheader("Hasil Prediksi (Top-1)")
        st.write(f"**Label:** {top_label}")
        st.write(f"**Probabilitas:** {top_prob:.4f}")

        # Top-5
        if probs_vec.size > 1:
            st.markdown("**Top-5 Probabilities**")
            rows = topk_from_vector(probs_vec, names, k=min(5, probs_vec.size))
            st.table({"Index": [r[0] for r in rows],
                      "Label": [r[1] for r in rows],
                      "Prob": [f"{r[2]:.4f}" for r in rows]})
