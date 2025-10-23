import os, importlib

def _cv2_is_headless():
    try:
        cv2 = importlib.import_module("cv2")
        # Headless wheel tidak link ke libGL; properti ini ada di keduanya,
        # tapi kita uji akses fungsi highgui yg biasanya butuh GUI.
        # Kalau headless, tetap ada tapi tak memicu load libGL saat import.
        return True
    except Exception as e:
        st.error(f"OpenCV belum siap (kemungkinan non-headless): {e}")
        return False

# di dalam handler tombol Run Classification (Face Detection):
if not _cv2_is_headless():
    st.stop()
fd_model = load_fd_model(FD_MODEL_PATH)
if fd_model is None:
    if "fd_import_error" in st.session_state:
        st.error(st.session_state["fd_import_error"])
    elif "fd_load_error" in st.session_state:
        st.error(st.session_state["fd_load_error"])
    else:
        st.error("Model YOLO belum siap. Periksa requirements.txt & redeploy.")
    st.stop()
