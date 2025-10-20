import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

# ==========================
# Konfigurasi opsional (hindari error GPU)
# ==========================
# tf.config.set_visible_devices([], 'GPU')  # uncomment jika perlu paksa CPU

# ==========================
# Helper
# ==========================
def assert_exists(path: str):
    if not os.path.exists(path):
        st.error(f"File tidak ditemukan: {path}. Pastikan path dan nama file benar.")
        st.stop()

# (Opsional) mapping kelas
CLASS_NAMES = ["Real Faces", "Sketch Faces", "Synthetic Faces"]  # ganti sesuai modelmu

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/Annisa Humaira_Laporan 4.pt"
    cls_path  = "model/Annisa Humaira_Laporan 2.h5"
    assert_exists(yolo_path)
    assert_exists(cls_path)

    yolo_model = YOLO(yolo_path)  # Model deteksi objek

    # Jika model punya custom objects, tambahkan di custom_objects
    classifier = tf.keras.models.load_model(cls_path)  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Pastikan RGB 3 kanal
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Ultralytics lebih aman menerima ndarray
        img_np = np.array(img)  # RGB
        results = yolo_model(img_np)
        result_img = results[0].plot()  # BGR ndarray
        # Konversi BGR -> RGB untuk tampilan yang benar
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing klasifikasi
        img_resized = img.resize((224, 224))  # sesuaikan dengan arsitektur model
        img_array = image.img_to_array(img_resized)  # (224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = int(np.argmax(prediction, axis=1)[0])
        prob = float(np.max(prediction))

        # Tampilkan hasil dengan label (jika ada)
        label = CLASS_NAMES[class_index] if 0 <= class_index < len(CLASS_NAMES) else f"Class {class_index}"
        st.write("### Hasil Prediksi:", label)
        st.write("Probabilitas:", f"{prob:.4f}")
