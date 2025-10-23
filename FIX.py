# app.py — Combined Dashboard: Car/Truck Classification & Face Detection
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from io import BytesIO
import base64
import datetime

# ML libs
import tensorflow as tf
from ultralytics import YOLO

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="CV Dashboard: Car/Truck & Face Detection", layout="wide")

# =========================
# KONSTAN / PATH MODEL
# =========================
# Car/Truck (TensorFlow .h5)
CT_MODEL_PATH = "model/Annisa Humaira_Laporan 2.h5"
CT_IMG_SIZE = (128, 128)

# Face Detection (YOLO .pt)
FD_MODEL_PATH = "model/Annisa Humaira_Laporan 4.pt"
FD_CONF_THRESH = 0.50
FD_IOU_THRESH  = 0.50
FD_IMGSZ       = 640

# =========================
# UTIL: Background per-halaman
# =========================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def apply_background(img_candidates):
    """
    Coba beberapa path gambar; pertama yang ada dipakai sebagai background.
    """
    for p in img_candidates:
        b64 = get_base64_image(p)
        if b64:
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/jpeg;base64,{b64}");
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
                """,
                unsafe_allow_html=True
            )
            return
    # jika tak ada background, tetap lanjut tanpa CSS khusus

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Car vs Truck", "Face Detection (Real / Sketch / Synthetic)"],
    index=0
)

# =========================
# ========== HALAMAN 1: CAR vs TRUCK ==========
# =========================
if page == "Car vs Truck":
    # background untuk halaman ini
    apply_background(["bg.jpg"])  # pakai bg.jpg seperti di code 1

    st.markdown('<div class="title">Car or Truck Classification</div>', unsafe_allow_html=True)

    # ---------- Model cache ----------
    @st.cache_resource(show_spinner=False)
    def load_ct_model():
        model = tf.keras.models.load_model(CT_MODEL_PATH)
        return model

    # ---------- Preprocess & Predict ----------
    def ct_preprocess_image(img: Image.Image):
        img = img.convert("RGB").resize(CT_IMG_SIZE)
        x = np.asarray(img).astype("float32") / 255.0
        return np.expand_dims(x, 0)

    def ct_predict(img: Image.Image, model):
        """
        Asumsi output model: probabilitas 'Car' di indeks 0 (seperti code 1).
        Jika >= 0.5 -> label 'Car', else 'Truck'. Confidence menyesuaikan.
        """
        x = ct_preprocess_image(img)
        preds = model.predict(x, verbose=0)
        p_car = float(preds.ravel()[0])  # raw prob of Car

        if p_car >= 0.5:
            label = "Car"
            conf = p_car
        else:
            label = "Truck"
            conf = 1.0 - p_car
        return label, conf, p_car

    # ---------- UI ----------
    uploaded_ct = st.file_uploader("Upload an image (JPG/PNG) untuk klasifikasi Car/Truck", type=["jpg", "jpeg", "png"], key="ct_uploader")

    if uploaded_ct:
        ct_img = Image.open(uploaded_ct)

        col1, col2, col3 = st.columns([1.2, 0.8, 1.2], gap="large")

        with col1:
            st.image(ct_img, use_container_width=True, caption="Uploaded Image")

        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("Run Classification", use_container_width=True, key="ct_run"):
                with st.spinner("Classifying..."):
                    ct_model = load_ct_model()
                    label, conf, raw_car = ct_predict(ct_img, ct_model)

                st.session_state["ct_prediction"] = {
                    "label": label,
                    "conf": conf,
                    "raw_car": raw_car,
                }

        with col3:
            pred = st.session_state.get("ct_prediction")
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

# =========================
# ========== HALAMAN 2: FACE DETECTION ==========
# =========================
else:
    # background untuk halaman ini
    apply_background(["bg2.jpg", "bg.jpeg"])  # sesuai code 2

    st.markdown('<div class="title">Face Detection: Real / Sketch / Synthetic</div>', unsafe_allow_html=True)

    # ---------- Model cache ----------
    @st.cache_resource(show_spinner=False)
    def load_fd_model(path: str):
        return YOLO(path)

    def fd_map_class_names(model) -> dict:
        """
        Samakan nama kelas berdasarkan kata kunci agar tidak kebalik:
        Real Face / Sketch Face / Synthetic Face
        """
        raw = model.names if hasattr(model, "names") else {}
        mapped = {}
        for cid, name in raw.items():
            n = str(name).lower().replace("_", " ").strip()
            if any(k in n for k in ["real", "photo", "natural"]):
                mapped[cid] = "Real Face"
            elif any(k in n for k in ["sketch", "draw", "hand", "pencil", "line"]):
                mapped[cid] = "Sketch Face"
            elif any(k in n for k in ["synt", "fake", "gen", "ai", "cg", "render"]):
                mapped[cid] = "Synthetic Face"
            else:
                mapped[cid] = name.replace("_", " ").title()
        return mapped

    def fd_annotate_image(result):
        """Kembalikan PIL image beranotasi dari result YOLO."""
        bgr = result.plot()  # ndarray BGR
        rgb = bgr[:, :, ::-1]
        return Image.fromarray(rgb)

    def fd_top_detection(result, names: dict):
        """
        Ambil 1 deteksi dengan confidence tertinggi.
        Return (label, conf) atau (None, None) jika tidak ada.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return None, None
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy().astype(int)
        idx = int(np.argmax(confs))
        label = names.get(clses[idx], str(clses[idx]))
        conf = float(confs[idx])
        return label, conf

    # ---------- UI ----------
    uploaded_fd = st.file_uploader("Upload an image (JPG/PNG) untuk Face Detection", type=["jpg", "jpeg", "png"], key="fd_uploader")

    if uploaded_fd:
        fd_img = Image.open(uploaded_fd).convert("RGB")

        col1, col2, col3 = st.columns([1.2, 0.9, 1.2], gap="large")

        # kiri: tampilkan gambar
        with col1:
            st.image(fd_img, caption="Uploaded Image", use_container_width=True)

        # tengah: tombol Run
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("Run Classification", use_container_width=True, key="fd_run"):
                with st.spinner("Detecting..."):
                    fd_model = load_fd_model(FD_MODEL_PATH)
                    names = fd_map_class_names(fd_model)
                    results = fd_model(
                        fd_img, conf=FD_CONF_THRESH, iou=FD_IOU_THRESH,
                        imgsz=FD_IMGSZ, verbose=False
                    )
                    result = results[0]

                    pred_label, pred_conf = fd_top_detection(result, names)
                    annotated = fd_annotate_image(result)

                st.session_state["fd_pred"] = {
                    "label": pred_label,
                    "conf": pred_conf,
                    "annotated": annotated
                }

        # kanan: hasil + tombol download
        with col3:
            pred = st.session_state.get("fd_pred")
            st.markdown("<br><br>", unsafe_allow_html=True)
            if pred:
                if pred["label"] is not None:
                    st.markdown(
                        f"""
                        <h2>Prediction: <span style="background:#e6f4ea;border-radius:8px;padding:4px 10px;">
                        {pred['label']}</span></h2>
                        <h3>Confidence: <span style="background:#e6f4ea;border-radius:8px;padding:2px 8px;">
                        {pred['conf']:.2f}</span></h3>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No faces detected.")

                if pred.get("annotated") is not None:
                    buf = BytesIO()
                    pred["annotated"].save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Detection Result",
                        data=buf.getvalue(),
                        file_name=f"faces_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )

