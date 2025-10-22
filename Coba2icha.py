import streamlit as st
import numpy as np
import base64
from pathlib import Path
from PIL import Image
import tensorflow as tf

# -----------------------------
# Konfigurasi halaman
st.set_page_config(page_title="Car or Truck Classification", layout="wide")

# -----------------------------
# Data model (edit sesuai file kamu)
DATA = {
    "Car": {
        "class_names": ["Not a Car", "A Car"],
        "weights_path": "model/Annisa Humaira_Laporan 2.h5",   # <— ganti jika perlu
    },
    "Truck": {
        "class_names": ["Not a Truck", "A Truck"],
        "weights_path": "model/Annisa Humaira_Laporan 2.h5",   # <— ganti ke .h5 lain bila ada
    },
}

# -----------------------------
# Background image (opsional)
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

# -----------------------------
# Loader & prediksi
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {weights_path}")
    # Load Keras model
    return tf.keras.models.load_model(str(path))

def preprocess_image(pil_img: Image.Image, img_size=(224, 224)) -> np.ndarray:
    img = pil_img.resize(img_size)
    arr = np.asarray(img).astype("float32") / 255.0
    if arr.ndim == 2:  # grayscale → 3 channel
        arr = np.stack([arr]*3, axis=-1)
    # (H, W, C) → (1, H, W, C)
    return np.expand_dims(arr, axis=0)

def classify_image(pil_img: Image.Image, model, class_names, img_size=(224, 224)):
    x = preprocess_image(pil_img, img_size=img_size)
    preds = model.predict(x, verbose=0)
    # handle output shape: (1, 1) sigmoid atau (1, 2) softmax
    if preds.shape[-1] == 1:
        score_pos = float(preds[0, 0])
        idx = int(score_pos >= 0.5)
        conf = score_pos if idx == 1 else 1.0 - score_pos
    else:
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
    label = class_names[idx]
    return label, conf

# -----------------------------
# UI
model_name = st.selectbox("Choose a Classification Model", list(DATA.keys()))
model_info = DATA[model_name]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1.2, 0.8, 1.2], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="Uploaded Image")

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        run = st.button("Run Classification", use_container_width=True)

        if run:
            with st.spinner("Loading model & classifying..."):
                model = load_model(model_info["weights_path"])
                label, score = classify_image(img, model, model_info["class_names"], img_size=(224, 224))
            st.session_state["prediction"] = (label, score, model_info["weights_path"], model_name)

    with col3:
        pred = st.session_state.get("prediction")
        if pred:
            label, score, weights_used, chosen_model = pred
            st.markdown(
                f"""
                <br><br>
                <h3>Prediction: <code>{label}</code></h3>
                <h4>Confidence: <code>{score:.2f}</code></h4>
                <h5>Model selected: <code>{chosen_model}</code></h5>
                <h5>Weights path: <code>{weights_used}</code></h5>
                """,
                unsafe_allow_html=True
            )
