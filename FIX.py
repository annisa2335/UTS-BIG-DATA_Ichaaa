# FIX.py — Combined Dashboard with auto-fix for OpenCV/Ultralytics import
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from io import BytesIO
import base64
import datetime
import sys
import subprocess

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="CV Dashboard: Car/Truck & Face Detection", layout="wide")

# =========================
# PATH & PARAMS
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
# UTILS
# =========================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def apply_background(img_candidates):
    """Try each image path; use the first that exists."""
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

def pip_install(pkgs: list[str]) -> tuple[bool, str]:
    """Install packages via pip; return (ok, log)."""
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--no-input"] + pkgs
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return True, out
    except subprocess.CalledProcessError as e:
        return False, e.output

# ---------- Dependency guards for Ultralytics/OpenCV ----------
def ensure_ultralytics_ready() -> tuple[object, object] | None:
    """
    Ensure cv2 and ultralytics are importable.
    Tries to install opencv-python-headless and ultralytics if missing.
    Returns (ultralytics_module, YOLO_class) on success, else None (and prints error to UI).
    """
    try:
        import cv2  # noqa: F401
        from ultralytics import YOLO
        import ultralytics
        return ultralytics, YOLO
    except Exception:
        with st.spinner("Menyiapkan dependensi YOLO (OpenCV & Ultralytics)..."):
            # NOTE: Banyak wheel saat ini stabil di Python 3.10–3.12.
            # Untuk Python 3.13, beberapa wheel mungkin belum tersedia.
            # Kita coba pasang paket headless yang umum.
            ok1, log1 = pip_install([
                # pin konservatif untuk kompatibilitas luas
                "opencv-python-headless<5.0",
            ])
            # Coba pasang/upgrade ultralytics juga (aman jika sudah ada)
            ok2, log2 = pip_install([
                "ultralytics>=8.2.0,<9.0.0",
                "numpy>=1.23"  # berjaga agar kompatibel dengan cv2
            ])
        # Coba import ulang
        try:
            import cv2  # noqa: F401
            from ultralytics import YOLO
            import ultralytics
            st.success("Dependensi YOLO siap. App akan reload…")
            st.rerun()
        except Exception as e2:
            st.error(
                "Gagal menyiapkan `cv2/ultralytics` secara otomatis.\n\n"
                "Tips cepat:\n"
                "• Pastikan versi Python 3.10–3.12.\n"
                "• Tambahkan ke requirements.txt: `opencv-python-headless<5.0` dan `ultralytics>=8.2.0,<9.0.0`.\n"
                "• Jika di Streamlit Cloud, commit requirements.txt lalu redeploy.\n\n"
                f"Detail error terakhir:\n{e2}"
            )
            return None

# ---------- Lazy import TensorFlow ----------
def ensure_tensorflow_ready():
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except Exception:
        with st.spinner("Menyiapkan TensorFlow…"):
            ok, log = pip_install([
                # Pilih versi yang umum tersedia di banyak env Cloud (ubah bila perlu)
                "tensorflow-cpu>=2.12,<3.0"
            ])
        try:
            import tensorflow as tf  # noqa: F401
            st.success("TensorFlow siap. App akan reload…")
            st.rerun()
        except Exception as e:
            st.error(
                "TensorFlow belum tersedia atau tidak kompatibel dengan environment saat ini.\n\n"
                "Saran:\n"
                "• Gunakan Python 3.10–3.11 untuk kompatibilitas TensorFlow yang lebih stabil.\n"
                "• Tambahkan `tensorflow-cpu>=2.12,<3.0` ke requirements.txt.\n"
                f"Detail error: {e}"
            )
            return False
    return True

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
    # background halaman ini
    apply_background(["bg.jpg"])
    st.markdown('<div class="title">Car or Truck Classification</div>', unsafe_allow_html=True)

    # Pastikan TensorFlow tersedia (lazy)
    if ensure_tensorflow_ready():
        import tensorflow as tf  # safe now

        @st.cache_resource(show_spinner=False)
        def load_ct_model():
            model = tf.keras.models.load_model(CT_MODEL_PATH)
            return model

        def ct_preprocess_image(img: Image.Image):
            img = img.convert("RGB").resize(CT_IMG_SIZE)
            x = np.asarray(img).astype("float32") / 255.0
            return np.expand_dims(x, 0)

        def ct_predict(img: Image.Image, model):
            x = ct_preprocess_image(img)
            preds = model.predict(x, verbose=0)
            p_car = float(preds.ravel()[0])  # raw prob of Car
            if p_car >= 0.5:
                label, conf = "Car", p_car
            else:
                label, conf = "Truck", 1.0 - p_car
            return label, conf, p_car

        uploaded_ct = st.file_uploader(
            "Upload an image (JPG/PNG) untuk klasifikasi Car/Truck",
            type=["jpg", "jpeg", "png"], key="ct_uploader"
        )

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
                        "label": label, "conf": conf, "raw_car": raw_car
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
    # background halaman ini
    apply_background(["bg2.jpg", "bg.jpeg"])
    st.markdown('<div class="title">Face Detection: Real / Sketch / Synthetic</div>', unsafe_allow_html=True)

    # Pastikan ultralytics & cv2 siap (lazy)
    ready = ensure_ultralytics_ready()
    if ready is not None:
        ultralytics, YOLO = ready  # type: ignore

        @st.cache_resource(show_spinner=False)
        def load_fd_model(path: str):
            return YOLO(path)

        def fd_map_class_names(model) -> dict:
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
            bgr = result.plot()               # ndarray BGR
            rgb = bgr[:, :, ::-1]            # to RGB
            return Image.fromarray(rgb)

        def fd_top_detection(result, names: dict):
            if result.boxes is None or len(result.boxes) == 0:
                return None, None
            confs = result.boxes.conf.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy().astype(int)
            idx = int(np.argmax(confs))
            label = names.get(clses[idx], str(clses[idx]))
            conf = float(confs[idx])
            return label, conf

        uploaded_fd = st.file_uploader(
            "Upload an image (JPG/PNG) untuk Face Detection",
            type=["jpg", "jpeg", "png"], key="fd_uploader"
        )

        if uploaded_fd:
            fd_img = Image.open(uploaded_fd).convert("RGB")

            col1, col2, col3 = st.columns([1.2, 0.9, 1.2], gap="large")

            with col1:
                st.image(fd_img, caption="Uploaded Image", use_container_width=True)

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
