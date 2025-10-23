# faces_detector.py
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from io import BytesIO
import datetime

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Face Detector (Real / Sketch / Synthetic)", layout="wide")

MODEL_PATH = "model/Annisa Humaira_Laporan 4.pt"
# Jika urutan/penamaan kelas di model kamu sudah tepat, biarkan None (pakai dari model)
# Kalau mau override manual, set dict berikut:
CLASS_NAMES_OVERRIDE = {0: "Real Face", 1: "Sketch Face", 2: "Synthetic Face"}

# =========================
# CACHE MODEL
# =========================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    model = YOLO(path)  # otomatis pilih device (GPU/CPU)
    return model

def get_class_names(model) -> dict:
    # Ambil nama kelas dari model; kalau kamu ingin pakai override, aktifkan di atas
    names = model.names if hasattr(model, "names") else None
    if CLASS_NAMES_OVERRIDE:
        names = CLASS_NAMES_OVERRIDE
    return names or {}

def draw_and_get_image(result):
    """
    result.plot() -> np.ndarray BGR.
    Ubah ke RGB, bungkus jadi PIL.Image untuk ditampilkan/diunduh.
    """
    bgr = result.plot()  # BGR uint8
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)

def summarize_counts(result, names: dict):
    # Ambil daftar class id untuk semua kotak deteksi
    if result.boxes is None or len(result.boxes) == 0:
        return {}
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    counts = {}
    for cid in cls_ids:
        label = names.get(cid, str(cid))
        counts[label] = counts.get(label, 0) + 1
    return counts

# =========================
# SIDEBAR / KONTROL
# =========================
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.01)
    iou  = st.slider("IoU threshold (NMS)", 0.10, 0.90, 0.50, 0.01)
    imgsz = st.select_slider("Image size (inference)", options=[320, 416, 512, 640, 768, 960], value=640)
    show_labels = st.checkbox("Show labels", value=True)
    show_conf   = st.checkbox("Show confidences", value=True)

st.markdown(
    "<h1 style='text-align:center'>Face Detection: Real / Sketch / Synthetic</h1>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# =========================
# INFERENCE
# =========================
if uploaded:
    # Tampilkan gambar input
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1.2, 1.0], gap="large")

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("Run Detection", use_container_width=True):
            with st.spinner("Running YOLO detection..."):
                model = load_model(MODEL_PATH)
                names = get_class_names(model)

                # Ultralytics YOLO inference
                results = model(
                    img, conf=conf, iou=iou, imgsz=imgsz, verbose=False,
                    show_labels=show_labels, show_conf=show_conf
                )
                result = results[0]

                # Gambar hasil + ringkasan
                out_img = draw_and_get_image(result)
                counts = summarize_counts(result, names)

            # Simpan ke session untuk panel hasil
            st.session_state["det_out"] = {
                "annotated": out_img,
                "counts": counts,
                "names": names,
                "boxes": result.boxes  # bisa dipakai jika ingin tabel detail
            }

# =========================
# PANEL HASIL
# =========================
det = st.session_state.get("det_out")
if det:
    a1, a2 = st.columns([1.2, 1.0], gap="large")

    with a1:
        st.image(det["annotated"], caption="Detections", use_container_width=True)

        # Tombol download hasil anotasi
        buf = BytesIO()
        det["annotated"].save(buf, format="PNG")
        filename = f"faces_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        st.download_button(
            label="⬇️ Download annotated image",
            data=buf.getvalue(),
            file_name=filename,
            mime="image/png",
            use_container_width=True
        )

    with a2:
        st.markdown("### Detection Summary")
        if det["counts"]:
            for k, v in det["counts"].items():
                st.write(f"- **{k}**: {v}")
        else:
            st.info("No faces detected.")

        # (Opsional) tampilkan tabel detail bbox
        if det["boxes"] is not None and len(det["boxes"]) > 0:
            st.markdown("### Boxes (preview)")
            boxes = det["boxes"]
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
            rows = []
            for i in range(len(cls)):
                rows.append({
                    "label": det["names"].get(cls[i], str(cls[i])),
                    "conf": float(confs[i]),
                    "x1": float(xyxy[i][0]),
                    "y1": float(xyxy[i][1]),
                    "x2": float(xyxy[i][2]),
                    "y2": float(xyxy[i][3]),
                })
            # ringkas: tampilkan 10 baris pertama
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df.head(10), use_container_width=True)
