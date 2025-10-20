# app.py
import os
import io
import glob
import tempfile
import numpy as np
import streamlit as st
from PIL import Image

# --- Keras (wajah)
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- YOLOv8 (kendaraan - classification)
from ultralytics import YOLO


# ==========================
# Konfigurasi & Label
# ==========================
# Kandidat lokasi file model (termasuk path upload-mu)
FACE_CANDIDATES = [
    "/mnt/data/Annisa Humaira_Laporan 2.h5",
    "model/Annisa Humaira_Laporan 2.h5",
    "models/Annisa Humaira_Laporan 2.h5",
    "Annisa Humaira_Laporan 2.h5",
]
VEHICLE_CANDIDATES = [
    "/mnt/data/Annisa Humaira_Laporan 4.pt",
    "model/Annisa Humaira_Laporan 4.pt",
    "models/Annisa Humaira_Laporan 4.pt",
    "Annisa Humaira_Laporan 4.pt",
]

FACE_INPUT_SIZE = (224, 224)  # sesuaikan jika arsitektur wajahmu berbeda

# Ganti ini sesuai urutan output model wajahmu
FACE_CLASS_NAMES = ["Real Faces", "Sketch Faces", "Synthetic Faces"]  # placeholder
# Untuk kendaraan, YOLOv8-cls biasanya menyertakan model.names; fallback di bawah hanya contoh
VEHICLE_FALLBACK_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle"]


# ==========================
# Utilitas
# ==========================
def debug_env():
    with st.expander("🔎 Debug path (klik jika butuh)"):
        st.write("**cwd**:", os.getcwd())
        try:
            st.write("**Isi direktori saat ini:**", os.listdir("."))
        except Exception as e:
            st.write("Tidak bisa membaca isi direktori:", e)
        candidates = glob.glob("**/*.pt", recursive=True) + glob.glob("**/*.h5", recursive=True)
        st.write("**Model yang terdeteksi (.pt/.h5):**", candidates[:100])

def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def topk_from_vector(vec, names, k=5):
    vec = np.asarray(vec).astype(float)
    idxs = np.argsort(-vec)[: min(k, vec.size)]
    rows = []
    for i in idxs:
        label = names[i] if (names is not None and i < len(names)) else f"class_{i}"
        rows.append((int(i), label, float(vec[i])))
    return rows


# ==========================
# Loader (cache)
# ==========================
@st.cache_resource
def load_face_model_from_path(path: str):
    # Jika ada custom_objects, tambahkan di sini
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_vehicle_model_from_path(path: str):
    return YOLO(path)  # diasumsikan YOLOv8 **classification** (.pt)


def get_face_model():
    """Coba muat model wajah dari repo/paths; jika tidak ada, minta upload .h5."""
    path = find_first_existing(FACE_CANDIDATES)
    if path:
        return load_face_model_from_path(path), f"✅ Model wajah dimuat dari: `{path}`"

    st.warning("Model **wajah** (.h5) belum ditemukan. Silakan upload file model di bawah.", icon="⚠️")
    up = st.file_uploader("Upload model wajah (.h5)", type=["h5"], key="upload_h5")
    if up is None:
        debug_env()
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(up.read())
        tmp_path = tmp.name
    return load_face_model_from_path(tmp_path), f"✅ Model wajah dimuat dari upload: `{os.path.basename(tmp_path)}`"


def get_vehicle_model():
    """Coba muat model kendaraan dari repo/paths; jika tidak ada, minta upload .pt."""
    path = find_first_existing(VEHICLE_CANDIDATES)
    if path:
        return load_vehicle_model_from_path(path), f"✅ Model kendaraan dimuat dari: `{path}`"

    st.warning("Model **kendaraan** (.pt) belum ditemukan. Silakan upload file model YOLOv8 **classification** di bawah.", icon="⚠️")
    up = st.file_uploader("Upload model kendaraan (.pt)", type=["pt"], key="upload_pt")
    if up is None:
        debug_env()
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(up.read())
        tmp_path = tmp.name
    return load_vehicle_model_from_path(tmp_path), f"✅ Model kendaraan dimuat dari upload: `{os.path.basename(tmp_path)}`"


# ==========================
# UI
# ==========================
st.set_page_config(page_title="Klasifikasi Wajah & Kendaraan", page_icon="🧠", layout="centered")
st.title("🧠 Image Classification — Wajah & Kendaraan")

mode = st.sidebar.selectbox(
    "Pilih Mode Klasifikasi:",
    ["Klasifikasi Wajah (Keras .h5)", "Klasifikasi Kendaraan (YOLOv8 .pt)"]
)

uploaded_img = st.file_uploader("Unggah Gambar (*.jpg, *.jpeg, *.png)", type=["jpg", "jpeg", "png"], key="img")
if uploaded_img is not None:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if mode == "Klasifikasi Wajah (Keras .h5)":
        model, src_msg = get_face_model()
        st.caption(src_msg)

        # Preprocess
        img_resized = img.resize(FACE_INPUT_SIZE)
        x = image.img_to_array(img_resized)           # (H, W, 3)
        x = np.expand_dims(x, axis=0)                 # (1, H, W, 3)
        x = x / 255.0

        # Predict
        preds = model.predict(x)
        probs = preds[0] if preds.ndim == 2 and preds.shape[0] == 1 else np.ravel(preds)
        top_idx = int(np.argmax(probs))
        top_prob = float(np.max(probs))
        label = FACE_CLASS_NAMES[top_idx] if 0 <= top_idx < len(FACE_CLASS_NAMES) else f"Class {top_idx}"

        st.subheader("Hasil Prediksi (Top-1)")
        st.write(f"**Label:** {label}")
        st.write(f"**Probabilitas:** {top_prob:.4f}")

        if probs.size > 1:
            st.markdown("**Top-5 Probabilities**")
            rows = topk_from_vector(probs, FACE_CLASS_NAMES, k=5)
            st.table({"Index": [r[0] for r in rows],
                      "Label": [r[1] for r in rows],
                      "Prob": [f"{r[2]:.4f}" for r in rows]})

    elif mode == "Klasifikasi Kendaraan (YOLOv8 .pt)":
        model, src_msg = get_vehicle_model()
        st.caption(src_msg)

        img_np = np.array(img)  # RGB ndarray
        results = model(img_np, verbose=False)
        if not results:
            st.error("Model tidak mengembalikan hasil. Periksa file model kendaraan.")
            st.stop()

        r0 = results[0]
        # Untuk YOLOv8 **classification**, r0.probs harus ada.
        if getattr(r0, "probs", None) is None:
            st.error("File .pt yang dimuat tampaknya **bukan** model klasifikasi (mungkin deteksi). "
                     "Gunakan YOLOv8 **classification** agar probabilitas kelas tersedia.")
            st.stop()

        # Ambil vektor probabilitas
        probs_vec = (
            r0.probs.data.cpu().numpy()
            if hasattr(r0.probs, "data")
            else np.asarray(r0.probs)
        )

        # Ambil nama kelas dari model; jika tidak ada, pakai fallback
        names = getattr(model, "names", None)
        if not names or len(names) < probs_vec.size:
            names = VEHICLE_FALLBACK_NAMES

        top_idx = int(np.argmax(probs_vec))
        top_prob = float(np.max(probs_vec))
        label = names[top_idx] if top_idx < len(names) else f"class_{top_idx}"

        st.subheader("Hasil Prediksi (Top-1)")
        st.write(f"**Label:** {label}")
        st.write(f"**Probabilitas:** {top_prob:.4f}")

        if probs_vec.size > 1:
            st.markdown("**Top-5 Probabilities**")
            rows = topk_from_vector(probs_vec, names, k=5)
            st.table({"Index": [r[0] for r in rows],
                      "Label": [r[1] for r in rows],
                      "Prob": [f"{r[2]:.4f}" for r in rows]})
else:
    st.info("Unggah gambar terlebih dahulu untuk melakukan klasifikasi.")
