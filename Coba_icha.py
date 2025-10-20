# app.py
import os, glob, tempfile
import numpy as np
import streamlit as st
from PIL import Image

# Wajah (Keras)
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Kendaraan (YOLOv8)
from ultralytics import YOLO


# ==========================
# Konfigurasi model & label
# ==========================
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

FACE_INPUT_SIZE = (224, 224)
# Ganti sesuai urutan output model wajahmu
FACE_CLASS_NAMES = ["Real Faces", "Sketch Faces", "Synthetic Faces"]


# ==========================
# Util umum
# ==========================
def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def debug_env():
    with st.expander("🔎 Debug (klik jika butuh)"):
        st.write("cwd:", os.getcwd())
        try:
            st.write("Isi direktori:", os.listdir("."))
        except Exception as e:
            st.write("Gagal list dir:", e)
        cands = glob.glob("**/*.pt", recursive=True) + glob.glob("**/*.h5", recursive=True)
        st.write("Model terdeteksi:", cands[:100])

def topk_from_vector(vec, names, k=5):
    vec = np.asarray(vec, dtype=float)
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
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_vehicle_model_from_path(path: str):
    return YOLO(path)

def get_face_model():
    path = find_first_existing(FACE_CANDIDATES)
    if path:
        return load_face_model_from_path(path), f"✅ Model wajah dimuat dari: `{path}`"

    st.warning("Model **wajah** (.h5) belum ditemukan. Silakan upload.", icon="⚠️")
    up = st.file_uploader("Upload model wajah (.h5)", type=["h5"], key="upload_h5")
    if up is None:
        debug_env(); st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(up.read()); tmp_path = tmp.name
    return load_face_model_from_path(tmp_path), f"✅ Model wajah dimuat dari upload: `{os.path.basename(tmp_path)}`"

def get_vehicle_model():
    path = find_first_existing(VEHICLE_CANDIDATES)
    if path:
        return load_vehicle_model_from_path(path), f"✅ Model kendaraan dimuat dari: `{path}`"

    st.warning("Model **kendaraan** (.pt) belum ditemukan. Upload YOLOv8 (cls/det) di sini.", icon="⚠️")
    up = st.file_uploader("Upload model kendaraan (.pt)", type=["pt"], key="upload_pt")
    if up is None:
        debug_env(); st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(up.read()); tmp_path = tmp.name
    return load_vehicle_model_from_path(tmp_path), f"✅ Model kendaraan dimuat dari upload: `{os.path.basename(tmp_path)}`"


# ==========================
# Normalisasi label → {car, truck}
# ==========================
CAR_ALIASES   = {"car","automobile","sedan","coupe","hatchback","convertible","sportscar","sports car","saloon"}
TRUCK_ALIASES = {"truck","pickup","pick-up","lorry","box truck","pickup truck"}

def normalize_to_binary(name: str):
    if not name:
        return None
    n = name.strip().lower()
    if n in CAR_ALIASES: return "car"
    if n in TRUCK_ALIASES: return "truck"
    if "truck" in n or "pickup" in n or "lorry" in n: return "truck"
    if "car" in n or "sedan" in n or "coupe" in n or "hatchback" in n: return "car"
    return None

def softmax2(a, b, eps=1e-9):
    x = np.array([a, b], dtype=float)
    x = x - x.max()
    ex = np.exp(x)
    p = ex / (ex.sum() + eps)
    return float(p[0]), float(p[1])


# ==========================
# UI
# ==========================
st.set_page_config(page_title="Klasifikasi Wajah & Kendaraan", page_icon="🧠", layout="centered")
st.title("🧠 Image Classification — Wajah & Kendaraan")

mode = st.sidebar.selectbox("Pilih Mode:", ["Klasifikasi Wajah (Keras .h5)", "Klasifikasi Kendaraan (YOLOv8 .pt)"])
uploaded_img = st.file_uploader("Unggah Gambar (*.jpg, *.jpeg, *.png)", type=["jpg","jpeg","png"], key="img")

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # =============== Wajah (Keras) ===============
    if mode == "Klasifikasi Wajah (Keras .h5)":
        face_model, src_msg = get_face_model()
        st.caption(src_msg)

        x = image.img_to_array(img.resize(FACE_INPUT_SIZE))
        x = np.expand_dims(x, 0) / 255.0
        preds = face_model.predict(x)
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
            st.table({"Index":[r[0] for r in rows], "Label":[r[1] for r in rows], "Prob":[f"{r[2]:.4f}" for r in rows]})

    # =============== Kendaraan (YOLOv8) ===============
    else:
        import cv2, pandas as pd

        veh_model, src_msg = get_vehicle_model()
        st.caption(src_msg)

        img_np = np.array(img)  # RGB
        results = veh_model(img_np, verbose=False)
        if not results:
            st.error("Model tidak mengembalikan hasil."); st.stop()
        r0 = results[0]

        names = getattr(veh_model, "names", None) or {}

        # ---- CASE 1: Model KLASIFIKASI (probs tersedia)
        if getattr(r0, "probs", None) is not None:
            vec = r0.probs.data.cpu().numpy() if hasattr(r0.probs, "data") else np.asarray(r0.probs)

            if len(vec) == 2:
                # coba baca nama kelas index 0/1
                n0 = names.get(0, "class_0").lower() if isinstance(names, dict) else str(names[0]).lower()
                n1 = names.get(1, "class_1").lower() if isinstance(names, dict) else str(names[1]).lower()
                b0 = normalize_to_binary(n0)
                b1 = normalize_to_binary(n1)
                if b0 and b1 and b0 != b1:
                    p_car, p_truck = (float(vec[0]), float(vec[1])) if b0=="car" else (float(vec[1]), float(vec[0]))
                    p_car, p_truck = softmax2(p_car, p_truck)
                    pred_label = "car" if p_car >= p_truck else "truck"
                    st.subheader("Hasil Prediksi (YOLOv8-cls → biner)")
                    st.write(f"**Label:** {pred_label}")
                    st.write(f"**Prob(car):** {p_car:.4f} | **Prob(truck):** {p_truck:.4f}")
                else:
                    idx = int(np.argmax(vec)); prob = float(np.max(vec))
                    raw_name = names[idx] if isinstance(names, dict) else names[idx]
                    mapped = normalize_to_binary(str(raw_name))
                    st.subheader("Hasil Prediksi (YOLOv8-cls)")
                    st.write(f"**Label:** {mapped or str(raw_name)}")
                    st.write(f"**Probabilitas:** {prob:.4f}")
            else:
                # >2 kelas → map ke biner dengan agregasi probabilitas
                p_car = p_truck = 0.0
                for i, p in enumerate(vec):
                    nm = names[i] if isinstance(names, dict) else names[i]
                    b = normalize_to_binary(str(nm))
                    if b == "car":   p_car   += float(p)
                    if b == "truck": p_truck += float(p)
                if p_car==0 and p_truck==0:
                    idx = int(np.argmax(vec)); prob = float(np.max(vec))
                    raw_name = names[idx] if isinstance(names, dict) else names[idx]
                    st.subheader("Hasil Prediksi (YOLOv8-cls, fallback)")
                    st.write(f"**Label (mentah):** {raw_name}")
                    st.write(f"**Probabilitas:** {prob:.4f}")
                else:
                    p_car, p_truck = softmax2(p_car, p_truck)
                    pred_label = "car" if p_car >= p_truck else "truck"
                    st.subheader("Hasil Prediksi (YOLOv8-cls → biner)")
                    st.write(f"**Label:** {pred_label}")
                    st.write(f"**Prob(car):** {p_car:.4f} | **Prob(truck):** {p_truck:.4f}")

        # ---- CASE 2: Model DETEKSI (boxes, tidak ada probs vektor)
        else:
            boxes = getattr(r0, "boxes", None)
            if boxes is None or len(boxes) == 0:
                st.warning("Tidak ada objek terdeteksi pada gambar.")
            else:
                cls = boxes.cls.cpu().numpy().astype(int)     # kelas per bbox
                conf = boxes.conf.cpu().numpy().astype(float) # confidence per bbox

                import pandas as pd
                df = pd.DataFrame({"cls": cls, "conf": conf})
                df["raw_name"] = df["cls"].map(lambda i: names.get(int(i), f"class_{int(i)}") if isinstance(names, dict) else names[int(i)] if names else f"class_{int(i)}")
                df["bin"] = df["raw_name"].map(lambda s: normalize_to_binary(str(s)))

                p_car   = float(df.loc[df["bin"]=="car",   "conf"].sum())
                p_truck = float(df.loc[df["bin"]=="truck", "conf"].sum())

                if p_car==0 and p_truck==0:
                    agg = df.groupby("raw_name", as_index=False)["conf"].sum().sort_values("conf", ascending=False)
                    top_raw = str(agg.iloc[0]["raw_name"]); top_score = float(agg.iloc[0]["conf"])
                    st.subheader("Hasil (Deteksi → fallback)")
                    st.write(f"**Label dominan (mentah):** {top_raw}")
                    st.write(f"**Skor Σconfidence:** {top_score:.4f}")
                else:
                    p_car, p_truck = softmax2(p_car, p_truck)
                    pred_label = "car" if p_car >= p_truck else "truck"
                    st.subheader("Hasil Prediksi (Deteksi → biner)")
                    st.write(f"**Label:** {pred_label}")
                    st.write(f"**Prob(car):** {p_car:.4f} | **Prob(truck):** {p_truck:.4f}")

                # tampilkan gambar hasil deteksi
                import cv2
                result_img = r0.plot()  # BGR
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img, caption="Deteksi Kendaraan (bbox)", use_container_width=True)
else:
    st.info("Unggah gambar terlebih dahulu.")
