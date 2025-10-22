# Coba2icha.py
import streamlit as st
import numpy as np
import base64
from pathlib import Path
from PIL import Image
import tensorflow as tf

# ==============================
# Konfigurasi halaman
# ==============================
st.set_page_config(page_title="Car or Truck Classification", layout="wide")

# ==============================
# Data model
# - weights_path: path .h5
# - class_names: [negatif, positif]
# - positive_label_index: index kelas positif pada output sigmoid (0 atau 1)
#   Jika prediksi terasa kebalik, ganti 0 <-> 1.
# ==============================
DATA = {
    "Car": {
        "class_names": ["Not a Car", "A Car"],
        "weights_path": "model/Annisa Humaira_Laporan 2.h5",
        "positive_label_index": 0,  # ubah ke 0 jika terasa kebalik
    },
    "Truck": {
        "class_names": ["Not a Truck", "A Truck"],
        "weights_path": "model/Annisa Humaira_Laporan 2.h5",
        "positive_label_index": 1,  # ubah ke 0 jika terasa kebalik
    },
}

# ==============================
# Background (opsional)
# ==============================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

bg_image_encoded = get_base64_image("bg.jpeg")
if bg_image_encoded:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_image_encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .title {{
            text-align: center;
            font-size: 50px;
            color: #f0f0f0;
            font-weight: bold;
            margin-top: -60px;
            text-shadow: 0 2px 6px rgba(0,0,0,0.35);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .title { text-align:center; font-size: 42px; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown('<div class="title">Car / Truck Image Classification</div>', unsafe_allow_html=True)

# ==============================
# Load model (cache)
# ==============================
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {weights_path}")
    model = tf.keras.models.load_model(str(p))
    return model

# ==============================
# Preprocess & Predict
# - Diset ke 128x128 RGB, rescale 1./255
# ==============================
IMG_SIZE = (128, 128)

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")                  # pastikan 3 channel
    img = img.resize(IMG_SIZE)                # samakan dengan training
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)         # (1, H, W, C)
    return arr

def classify_image(
    pil_img: Image.Image,
    model,
    class_names,
    positive_label_index: int = 1,
    threshold: float = 0.5
):
    x = preprocess_image(pil_img)
    preds = model.predict(x, verbose=0)

    # Handle 1-unit sigmoid vs 2-unit softmax
    if preds.shape[-1] == 1:
        p_pos = float(preds.ravel()[0])  # probabilitas kelas "positif" (index=1 pada biner standar)
        # Jika positive_label_index==1 -> label index 1 adalah class_names[1]
        # Jika positive_label_index==0 -> "positif" diartikan class_names[0]
        if positive_label_index == 1:
            idx = 1 if p_pos >= threshold else 0
            conf = p_pos if idx == 1 else (1.0 - p_pos)
        else:
            # kebalikan mapping
            idx = 0 if p_pos >= threshold else 1
            conf = p_pos if idx == 0 else (1.0 - p_pos)
    else:
        # Softmax 2 unit
        probs = preds[0].astype("float32")
        idx = int(np.argmax(probs))
        conf = float(np.max(probs))

    label = class_names[idx]
    return label, conf

# ==============================
# UI
# ==============================
model_name = st.selectbox("Choose a Classification Model", list(DATA.keys()))
model_info = DATA[model_name]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# (Opsional) threshold bisa kamu ubah kalau butuh kalibrasi
# Sembunyikan kalau tak perlu: set show_threshold=False
show_threshold = False
if show_threshold:
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
else:
    threshold = 0.5

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1.2, 0.8, 1.2], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="Uploaded Image")

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        run = st.button("Run Classification", use_container_width=True)

        if run:
            try:
                with st.spinner("Classifying..."):
                    model = load_model(model_info["weights_path"])
                    label, score = classify_image(
                        pil_img=img,
                        model=model,
                        class_names=model_info["class_names"],
                        positive_label_index=model_info.get("positive_label_index", 1),
                        threshold=threshold,
                    )
                st.session_state["prediction"] = {
                    "label": label,
                    "score": score,
                    "chosen": model_name,
                    "weights": model_info["weights_path"],
                }
            except Exception as e:
                st.error(f"Failed to run classification: {e}")

    with col3:
        pred = st.session_state.get("prediction")
        if pred:
            st.markdown(
                f"""
                <br><br>
                <h3>Prediction: <code>{pred['label']}</code></h3>
                <h4>Confidence: <code>{pred['score']:.2f}</code></h4>
                """,
                unsafe_allow_html=True
            )
