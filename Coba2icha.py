# Coba2icha.py
import streamlit as st
import numpy as np
import base64
from pathlib import Path
from PIL import Image
import tensorflow as tf

# ==============================
# 1) Konfigurasi halaman
# ==============================
st.set_page_config(page_title="Car or Truck Classification", layout="wide")

# ==============================
# 2) Data model (EDIT path sesuai file kamu)
# ==============================
DATA = {
    "Car": {
        "class_names": ["Not a Car", "A Car"],
        "weights_path": "model/Annisa Humaira_Laporan 2.h5",   # ganti jika perlu
    },
    "Truck": {
        "class_names": ["Not a Truck", "A Truck"],
        "weights_path": "model/Annisa Humaira_Laporan 2.h5",   # ganti ke .h5 lain bila ada
    },
}

# ==============================
# 3) Background (opsional)
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
# 4) Load model & adaptasi input
# ==============================
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {weights_path}")
    model = tf.keras.models.load_model(str(p))
    input_shape = model.inputs[0].shape  # misal: (None, 150, 150, 3) atau (None, 3, 224, 224)
    return model, input_shape

def preprocess_image_adaptive(pil_img: Image.Image, input_shape) -> np.ndarray:
    # input_shape di Keras berbentuk tf.TensorShape; ambil dimensi 1..3
    if len(input_shape) != 4:
        raise ValueError(f"Model expects 4D tensor, got {input_shape}")

    a = input_shape[1]  # bisa None/int
    b = input_shape[2]
    c = input_shape[3]

    # Deteksi channels_first vs channels_last
    # Heuristik: jika dimensi ke-1 adalah 1/3 dan dimensi terakhir bukan 1/3 -> channels_first
    channels_first = False
    if a in (1, 3) and (c not in (1, 3, None)):
        channels_first = True

    if channels_first:
        C = int(a) if a is not None else 3
        H = int(b) if b is not None else 224
        W = int(c) if c is not None else 224
    else:
        H = int(a) if a is not None else 224
        W = int(b) if b is not None else 224
        C = int(c) if c is not None else 3

    # Konversi channel sesuai model
    if C == 1:
        img = pil_img.convert("L")
    else:
        img = pil_img.convert("RGB")

    # Resize sesuai HxW model
    img = img.resize((W, H))
    arr = np.asarray(img).astype("float32") / 255.0

    # Pastikan shape punya channel
    if C == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)

    # Susun ke channels_first bila perlu
    if channels_first:
        if arr.ndim == 2:
            arr = np.expand_dims(arr, -1)
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)

    # Tambahkan batch dim
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C) atau (1, C, H, W)
    return arr

def classify_image(pil_img: Image.Image, model, input_shape, class_names):
    x = preprocess_image_adaptive(pil_img, input_shape)
    preds = model.predict(x, verbose=0)

    # Output bisa sigmoid (1 unit) atau softmax (2 unit)
    if preds.shape[-1] == 1:
        p = float(preds.ravel()[0])
        idx = int(p >= 0.5)
        conf = p if idx == 1 else 1.0 - p
    else:
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))

    return class_names[idx], conf

# ==============================
# 5) UI
# ==============================
model_name = st.selectbox("Choose a Classification Model", list(DATA.keys()))
model_info = DATA[model_name]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Info debug (membantu kalau shape tidak cocok)
with st.expander("ℹ️ Model & Input Info"):
    st.write("Selected model:", model_name)
    st.write("Weights path:", model_info["weights_path"])
    try:
        _m, _shape = load_model(model_info["weights_path"])
        st.write("Model input shape:", tuple(_shape))
    except Exception as e:
        st.write("Model not loaded yet / error:", str(e))

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1.2, 0.8, 1.2], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="Uploaded Image")

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        run = st.button("Run Classification", use_container_width=True)

        if run:
            try:
                with st.spinner("Loading model & classifying..."):
                    model, input_shape = load_model(model_info["weights_path"])
                    label, score = classify_image(img, model, input_shape, model_info["class_names"])
                st.session_state["prediction"] = {
                    "label": label,
                    "score": score,
                    "weights": model_info["weights_path"],
                    "chosen": model_name,
                    "input_shape": tuple(input_shape),
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
                <h5>Model selected: <code>{pred['chosen']}</code></h5>
                <h5>Weights path: <code>{pred['weights']}</code></h5>
                <h5>Input shape: <code>{pred['input_shape']}</code></h5>
                """,
                unsafe_allow_html=True
            )
