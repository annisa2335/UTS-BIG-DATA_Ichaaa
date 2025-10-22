# app_multiclass.py
import streamlit as st
import numpy as np
import base64
from pathlib import Path
from PIL import Image
import tensorflow as tf

# ========= CONFIG =========
WEIGHTS_PATH = "model/Annisa Humaira_Laporan 2.h5"

# Urutan label SESUAI OUTPUT MODEL saat training!
# Jika modelmu output-nya mis. [Car, Truck, Other] biarkan seperti ini.
# Kalau ternyata urutan modelmu [Truck, Car, Other], ubah urutannya di list ini.
CLASS_NAMES_IN_MODEL = ["Car", "Truck", "Other"]

IMG_SIZE = (128, 128)   # ganti jika training pakai ukuran lain
SHOW_BG = True          # set False kalau tak mau background

# ========= PAGE =========
st.set_page_config(page_title="Vehicle Multiclass Classification", layout="wide")

def get_base64_image(pth: str) -> str:
    p = Path(pth)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

if SHOW_BG:
    bg = get_base64_image("bg.jpeg")
    if bg:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{bg}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            .title {{
                text-align: center; font-size: 48px; font-weight: 800;
                color: #f0f0f0; text-shadow: 0 2px 6px rgba(0,0,0,.35);
                margin-top: -50px;
            }}
            </style>
            """, unsafe_allow_html=True
        )

st.markdown('<div class="title">Vehicle Image Classification (Car / Truck / Other)</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {weights_path}")
    model = tf.keras.models.load_model(str(p))
    return model

def preprocess(img: Image.Image, size=(128,128)) -> np.ndarray:
    img = img.convert("RGB").resize(size)
    x = np.asarray(img).astype("float32") / 255.0
    x = np.expand_dims(x, 0)   # (1,H,W,C)
    return x

def predict_multiclass(pil_img: Image.Image, model, class_names):
    x = preprocess(pil_img, size=IMG_SIZE)
    y = model.predict(x, verbose=0)

    # Normalisasi bentuk output → (num_classes,)
    if y.ndim == 2:
        y = y[0]
    else:
        y = np.asarray(y).ravel()

    # Jika model biner (1 unit), naikkan jadi 2-kelas; jika 2 unit, pakai apa adanya.
    if y.shape[0] == 1:
        y = np.array([1.0 - y[0], y[0]])  # [neg, pos]
        # pad jadi 3 kelas: treat lainnya sebagai very small prob
        if len(class_names) == 3:
            y = np.array([y[0], y[1], 1e-6])
    elif y.shape[0] == 2 and len(class_names) == 3:
        y = np.append(y, 1e-6)

    probs = y / (y.sum() + 1e-12)
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs

# ========== UI ==========
uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])

# (opsional) “task view”: hanya untuk menampilkan highlight kelas tertentu
task = st.selectbox("View focus (optional)", ["All", "Car", "Truck", "Other"])

if uploaded:
    img = Image.open(uploaded)

    col1, col2, col3 = st.columns([1.2, 0.8, 1.2], gap="large")
    with col1:
        st.image(img, use_container_width=True, caption="Uploaded Image")

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Run Classification", use_container_width=True):
            with st.spinner("Classifying..."):
                model = load_model(WEIGHTS_PATH)
                label, score, probs = predict_multiclass(img, model, CLASS_NAMES_IN_MODEL)
            st.session_state["pred"] = dict(label=label, score=score, probs=probs)

    with col3:
        pred = st.session_state.get("pred")
        if pred:
            st.markdown(
                f"""
                <br><br>
                <h3>Prediction: <code>{pred['label']}</code></h3>
                <h4>Confidence: <code>{pred['score']:.2f}</code></h4>
                """,
                unsafe_allow_html=True
            )
            # tampilkan probabilitas per kelas (ringkas)
            prob_lines = []
            for i, cname in enumerate(CLASS_NAMES_IN_MODEL):
                p = float(pred["probs"][i])
                if task == "All" or task == cname:
                    prob_lines.append(f"- **{cname}**: {p:.2f}")
            st.markdown("\n".join(prob_lines))

