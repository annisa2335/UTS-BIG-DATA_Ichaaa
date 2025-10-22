import streamlit as st
import numpy as np
import base64
from pathlib import Path
from PIL import Image
import tensorflow as tf

# ============================
# KONFIGURASI
# ============================
st.set_page_config(page_title="Car or Truck Classification", layout="wide")

MODEL_PATH = "model/Annisa Humaira_Laporan 2.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Not a Car", "A Car"]  # sesuai model kamu

# ============================
# BACKGROUND
# ============================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

bg_img = get_base64_image("bg.jpeg")
if bg_img:
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .title {{
            text-align: center;
            font-size: 48px;
            font-weight: 800;
            color: #f0f0f0;
            text-shadow: 0 2px 6px rgba(0,0,0,.35);
        }}
        </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title">Car or Truck Classification</div>', unsafe_allow_html=True)

# ============================
# LOAD MODEL
# ============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(x, 0)

def predict_car_truck(img: Image.Image, model):
    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)
    p_car = float(preds.ravel()[0])

    # ----------------------------
    # LOGIKA TAMBAHAN:
    # jika model yakin itu Car (>= 0.5), kita sebut "Car"
    # jika tidak, asumsikan "Truck"
    # ----------------------------
    if p_car >= 0.5:
        label = "Car"
        conf = p_car
    else:
        label = "Truck"
        conf = 1.0 - p_car

    return label, conf, p_car

# ============================
# UI STREAMLIT
# ============================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1.2, 0.8, 1.2], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="Uploaded Image")

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Run Classification", use_container_width=True):
            with st.spinner("Classifying..."):
                model = load_model()
                label, conf, raw_car = predict_car_truck(img, model)

            st.session_state["prediction"] = {
                "label": label,
                "conf": conf,
                "raw_car": raw_car,
            }

    with col3:
        pred = st.session_state.get("prediction")
        if pred:
            st.markdown(
                f"""
                <br><br>
                <h3>Prediction: <code>{pred['label']}</code></h3>
                <h4>Confidence: <code>{pred['conf']:.2f}</code></h4>
                <p style="font-size:14px;color:gray;">
                    (Model raw Car probability = {pred['raw_car']:.2f})
                </p>
                """,
                unsafe_allow_html=True,
            )
