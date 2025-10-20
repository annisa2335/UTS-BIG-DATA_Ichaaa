import os, io, tempfile, glob
import streamlit as st
from ultralytics import YOLO

# --- util ---
def debug_env():
    with st.expander("🔎 Debug path (klik jika butuh)"):
        st.write("**cwd**:", os.getcwd())
        # tampilkan beberapa file/folder tingkat atas
        st.write("**Isi direktori saat ini:**", os.listdir("."))
        # cari kandidat .pt
        candidates = glob.glob("**/*.pt", recursive=True)
        st.write("**Semua .pt yang ditemukan (recursive):**", candidates[:50])

def locate_vehicle_model(filename="Annisa Humaira_Laporan 4.pt"):
    # lokasi yang lazim di repo
    candidates = [
        f"model/{filename}",
        f"models/{filename}",
        filename,                     # di root
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

@st.cache_resource
def load_vehicle_model_from_path(path: str):
    return YOLO(path)  # diasumsikan model klasifikasi YOLOv8-cls

def get_vehicle_model():
    """Coba load dari repo; jika tidak ada, minta upload .pt."""
    path = locate_vehicle_model()
    if path:
        return load_vehicle_model_from_path(path), f"✅ Memuat dari repo: `{path}`"

    # fallback: minta upload
    st.warning("File model kendaraan (.pt) belum ditemukan di repo. "
               "Silakan upload file model YOLOv8 **klasifikasi** di bawah ini.", icon="⚠️")
    uploaded_pt = st.file_uploader("Upload model kendaraan (.pt)", type=["pt"], key="upload_pt")
    if uploaded_pt is not None:
        # simpan ke file sementara (YOLO butuh path, bukan file-like)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(uploaded_pt.read())
            tmp_path = tmp.name
        return load_vehicle_model_from_path(tmp_path), f"✅ Memuat dari upload: `{os.path.basename(tmp_path)}`"

    debug_env()
    st.stop()
