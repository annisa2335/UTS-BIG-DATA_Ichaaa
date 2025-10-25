# app_ui_minimal.py (dengan Registrasi, Login & fix st.rerun)
# =========================================================
import io
import os
import json
import time
import base64
import hashlib
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

# -------- Optional imports --------
_HAS_ULTRA = True
try:
    from ultralytics import YOLO
except Exception as e:
    _HAS_ULTRA = False
    _ULTRA_ERR = e

_HAS_TF = True
try:
    import tensorflow as tf
except Exception as e:
    _HAS_TF = False
    _TF_ERR = e

# =========================
# KONFIG & STATE
# =========================
st.set_page_config(page_title="Dashboard_Annisa", layout="wide", page_icon="🪄")

if "page" not in st.session_state:
    st.session_state.page = "home"  # home | detect | classify | about | help
if "det_output" not in st.session_state:
    st.session_state.det_output = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None  # dict: {name, npm, email}

# =========================
# IDENTITAS & MODEL PARAM
# =========================
AUTHOR_NAME = "Annisa Humaira"
AUTHOR_NPM  = "2208108010070"
LOGO_PATH   = "LOGO-USK-MASTER.png"

YOLO_MODEL_PATH   = "model/Annisa Humaira_Laporan 4.pt"
KERAS_MODEL_PATH  = "model/Annisa Humaira_Laporan 2.h5"
IMG_SIZE          = (128, 128)

# --- UI sizes (px) ---
PREVIEW_WIDTH = 480
OUTPUT_WIDTH  = 640

# Default parameter
YOLO_DEFAULT_CONF = 0.5
YOLO_DEFAULT_IOU  = 0.5
YOLO_INFER_SIZE   = 640
SHOW_DOWNLOAD_BTN = True

# =========================
# AUTH STORAGE (local JSON)
# =========================
USERS_FILE = Path("users.json")

def _read_users():
    if not USERS_FILE.exists():
        return []
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def _write_users(users: list):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")

def _hash_password(password: str, salt: bytes) -> str:
    h = hashlib.sha256()
    h.update(salt + password.encode("utf-8"))
    return h.hexdigest()

def register_user(name: str, npm: str, email: str, password: str):
    users = _read_users()
    email_norm = email.strip().lower()
    if any(u["email"] == email_norm for u in users):
        return False, "Email sudah terdaftar."
    if len(password) < 6:
        return False, "Password minimal 6 karakter."

    salt = os.urandom(16)
    user = {
        "name": name.strip(),
        "npm": npm.strip(),
        "email": email_norm,
        "salt": base64.b64encode(salt).decode("utf-8"),
        "password_hash": _hash_password(password, salt),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    users.append(user)
    _write_users(users)
    return True, "Registrasi berhasil. Silakan login."

def login_user(email: str, password: str):
    users = _read_users()
    email_norm = email.strip().lower()
    user = next((u for u in users if u["email"] == email_norm), None)
    if not user:
        return False, "Email belum terdaftar."
    try:
        salt = base64.b64decode(user["salt"].encode("utf-8"))
    except Exception:
        return False, "Data akun rusak."
    if _hash_password(password, salt) != user["password_hash"]:
        return False, "Password salah."
    profile = {"name": user["name"], "npm": user["npm"], "email": user["email"]}
    return True, profile

# =========================
# THEME & BACKGROUND
# =========================
def get_base64_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

bg_img = ""
for cand in ["bg.jpg", "bg.jpeg"]:
    bg_img = get_base64_image(cand)
    if bg_img:
        break

PRIMARY = "#7C3AED"
TEXT_MUTED = "#6B7280"

st.markdown(
    f"""
    <style>
    .stApp {{
        {"background-image: url('data:image/jpeg;base64," + bg_img + "');" if bg_img else ""}
        background-size: cover; background-position: center; background-repeat: no-repeat;
    }}
    .hero {{
        background: linear-gradient(135deg, rgba(124,58,237,.85), rgba(16,185,129,.85));
        border-radius: 24px; padding: 36px; color: #fff;
        box-shadow: 0 18px 40px rgba(0,0,0,.25);
    }}
    .card {{ background: rgba(255,255,255,.96); border-radius: 18px; padding: 20px;
             box-shadow: 0 10px 26px rgba(0,0,0,.12); }}
    .topbar {{
        background: rgba(255,255,255,.95); border-radius: 14px; padding: 10px 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,.10); margin-bottom: 8px;
        display:flex; align-items:center; gap:12px;
    }}
    .topbar .title {{ font-weight: 800; font-size: 18px; line-height:1.2; }}
    .topbar .sub   {{ color:{TEXT_MUTED}; font-size: 12px; }}
    .result-label {{ font-weight: 900; font-size: 36px; margin: 8px 0 6px 0; letter-spacing: .5px; color: #111827; }}
    .footer {{ color:{TEXT_MUTED}; font-size:12px; text-align:center; margin-top:36px; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HELPERS YOLO
# =========================
@st.cache_resource(show_spinner=False)
def load_yolo_model(path: str):
    if not _HAS_ULTRA:
        raise RuntimeError(f"Ultralytics/YOLO belum tersedia: {_ULTRA_ERR}")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model YOLO tidak ditemukan: {path}")
    return YOLO(path)

def get_class_names(model) -> dict:
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

def draw_and_get_image(result):
    bgr = result.plot()
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)

def summarize_counts(result, names: dict):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    return [(names.get(cid, str(cid)), float(conf)) for cid, conf in zip(cls_ids, confs)]

# =========================
# HELPERS CLASSIFIER
# =========================
@st.cache_resource
def load_keras_model():
    if not _HAS_TF:
        raise RuntimeError(f"TensorFlow/Keras belum tersedia: {_TF_ERR}")
    p = Path(KERAS_MODEL_PATH)
    if not p.exists():
        raise RuntimeError(f"Model Keras tidak ditemukan: {KERAS_MODEL_PATH}")
    return tf.keras.models.load_model(KERAS_MODEL_PATH)

def preprocess_image(img: Image.Image, size=(128, 128)):
    img = img.convert("RGB").resize(size)
    x = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(x, 0)

def predict_car_truck(img: Image.Image, model):
    x = preprocess_image(img, size=IMG_SIZE)
    preds = model.predict(x, verbose=0)
    p_car = float(preds.ravel()[0])
    if p_car >= 0.5:
        label, conf = "Car", p_car
    else:
        label, conf = "Truck", 1.0 - p_car
    return label, conf, p_car

# =========================
# UTIL: Rerun kompatibel versi
# =========================
def do_rerun():
    """Panggil st.rerun() dengan fallback ke st.experimental_rerun() untuk Streamlit lama."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()  # type: ignore[attr-defined]
        except Exception:
            pass

# =========================
# TOP BAR (hanya saat login)
# =========================
def topbar():
    logo_b64 = get_base64_image(LOGO_PATH)
    cols = st.columns([0.1, 0.6, 0.3])
    with cols[0]:
        if logo_b64:
            st.markdown(
                f"<div class='topbar' style='justify-content:center;'><img src='data:image/png;base64,{logo_b64}' height='48' /></div>",
                unsafe_allow_html=True
            )
    with cols[1]:
        st.markdown(
            "<div class='topbar'><div><div class='title'>Universitas Syiah Kuala</div><div class='sub'>Fakultas MIPA — Statistika</div></div></div>",
            unsafe_allow_html=True
        )
    with cols[2]:
        right = f"<div class='title'>{AUTHOR_NAME}</div><div class='sub'>NPM: {AUTHOR_NPM}</div>"
        if st.session_state.auth_user:
            u = st.session_state.auth_user
            right += f"<div class='sub'>Login sebagai: {u['name']} ({u['email']})</div>"
        st.markdown(f"<div class='topbar' style='justify-content:flex-end;'><div style='text-align:right'>{right}</div></div>", unsafe_allow_html=True)
        if st.session_state.auth_user:
            if st.button("Logout", key="logout_btn"):
                st.session_state.auth_user = None
                st.session_state.page = "home"
                st.success("Anda telah logout.")
                do_rerun()

# =========================
# NAVBAR (hanya saat login)
# =========================
def navbar():
    tabs = ["🏠 Home", "🧭 Detect", "🏷️ Classify", "ℹ️ About", "❓ Help"]
    ids  = ["home", "detect", "classify", "about", "help"]
    idx_default = ids.index(st.session_state.page) if st.session_state.page in ids else 0
    choice = st.radio("Navigation", tabs, horizontal=True, index=idx_default, label_visibility="collapsed")
    mapping = dict(zip(tabs, ids))
    st.session_state.page = mapping[choice]

# =========================
# AUTH PAGE
# =========================
def page_auth():
    st.markdown(
        """
        <div class="hero">
          <h1>Welcome 👋</h1>
          <p>Silakan <b>Login</b> atau <b>Daftar</b> untuk mengakses Dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    tab_login, tab_register = st.tabs(["🔐 Login", "📝 Daftar"])

    # ---- LOGIN ----
    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="nama@domain.com")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Masuk")
        if submit:
            ok, data = login_user(email, password)
            if ok:
                st.session_state.auth_user = data  # {name,npm,email}
                st.success(f"Login berhasil. Selamat datang, {data['name']}!")
                # 🔧 FIX: gunakan st.rerun() (fallback ke experimental bila ada)
                do_rerun()
            else:
                st.error(data)

    # ---- REGISTER ----
    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            name = st.text_input("Nama Lengkap")
            npm  = st.text_input("NPM")
            email= st.text_input("Email", placeholder="nama@domain.com")
            colp1, colp2 = st.columns(2)
            with colp1:
                pw1 = st.text_input("Password", type="password")
            with colp2:
                pw2 = st.text_input("Ulangi Password", type="password")
            agree = st.checkbox("Saya setuju menyimpan data ini secara lokal di perangkat ini.")
            submit_reg = st.form_submit_button("Daftar")
        if submit_reg:
            if not agree:
                st.warning("Centang persetujuan terlebih dahulu.")
            elif pw1 != pw2:
                st.error("Password tidak sama.")
            elif not name or not npm or not email:
                st.error("Nama, NPM, dan Email wajib diisi.")
            else:
                ok, msg = register_user(name, npm, email, pw1)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.markdown(
        "<p class='muted' style='text-align:center;margin-top:24px;'>Akun disimpan lokal pada <code>users.json</code> menggunakan hash + salt (tanpa plaintext password).</p>",
        unsafe_allow_html=True
    )

# =========================
# DASHBOARD PAGES
# =========================
def page_home():
    st.markdown(
        """
        <div class="hero">
          <h1>Dual Vision Dashboard</h1>
          <p>Deteksi wajah & klasifikasi kendaraan — cepat, ringan, dan mudah.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><h3>🧭 Face Detection</h3><p class='muted'>Deteksi wajah (Real/Sketch/Synthetic).</p></div>", unsafe_allow_html=True)
        if st.button("Mulai Deteksi", use_container_width=True):
            st.session_state.page = "detect"
    with col2:
        st.markdown("<div class='card'><h3>🏷️ Car vs Truck</h3><p class='muted'>Klasifikasi kendaraan (Car atau Truck).</p></div>", unsafe_allow_html=True)
        if st.button("Mulai Klasifikasi", use_container_width=True):
            st.session_state.page = "classify"

def page_detect():
    st.markdown("### 🧭 Face Detection — Real / Sketch / Synthetic")
    uploaded = st.file_uploader("📤 Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="up_det")
    if uploaded:
        img_preview = Image.open(uploaded).convert("RGB")
        st.image(img_preview, caption="Pratinjau Gambar", width=PREVIEW_WIDTH, use_container_width=False)
    else:
        st.info("Upload gambar untuk memulai deteksi.")
        return

    run = st.button("🔎 Jalankan Deteksi", use_container_width=True)
    if run:
        try:
            start = time.time()
            with st.spinner("Detecting faces..."):
                model = load_yolo_model(YOLO_MODEL_PATH)
                names = get_class_names(model)
                results = model(img_preview, conf=YOLO_DEFAULT_CONF, iou=YOLO_DEFAULT_IOU, imgsz=YOLO_INFER_SIZE)
                result = results[0]
                annotated = draw_and_get_image(result)
                detections = summarize_counts(result, names)
            elapsed = time.time() - start
            st.session_state.det_output = {"annotated": annotated, "detections": detections, "elapsed": elapsed}
        except Exception as e:
            st.error(f"Error: {e}")

    out = st.session_state.det_output
    if out:
        c1, c2 = st.columns([1.2, 0.8], gap="large")
        with c1:
            st.image(out["annotated"], caption="🖼️ Detections", width=OUTPUT_WIDTH, use_container_width=False)
            if SHOW_DOWNLOAD_BTN:
                buf = io.BytesIO()
                out["annotated"].save(buf, format="PNG")
                filename = f"faces_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                st.download_button("⬇️ Download annotated image", buf.getvalue(), file_name=filename, mime="image/png")
        with c2:
            st.markdown("#### 📊 Ringkasan")
            count = len(out["detections"]) if out["detections"] else 0
            mcol = st.columns(3)
            mcol[0].metric("Detections", count)
            mcol[1].metric("ImgSize", YOLO_INFER_SIZE)
            mcol[2].metric("Latency (s)", f"{out['elapsed']:.2f}")
            st.markdown("#### 🔖 Detail")
            if out["detections"]:
                for i, (label, conf) in enumerate(out["detections"], start=1):
                    st.markdown(f"{i}. **{label}** — `{conf:.2f}`")
            else:
                st.info("Tidak ada wajah terdeteksi.")

def page_classify():
    st.markdown("### 🏷️ Car vs Truck Classification")
    uploaded = st.file_uploader("📤 Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="up_cls")
    if uploaded:
        img_preview = Image.open(uploaded)
        st.image(img_preview, caption="Pratinjau Gambar", width=PREVIEW_WIDTH, use_container_width=False)
    else:
        st.info("Upload gambar untuk memulai klasifikasi.")
        return

    run = st.button("🧠 Jalankan Klasifikasi", use_container_width=True)
    if run:
        try:
            start = time.time()
            with st.spinner("Classifying..."):
                model = load_keras_model()
                label, conf, raw_car = predict_car_truck(img_preview, model)
            elapsed = time.time() - start
            st.session_state.prediction = {"label": label, "conf": conf, "raw_car": raw_car, "elapsed": elapsed}
        except Exception as e:
            st.error(f"Error: {e}")

    pred = st.session_state.prediction
    if pred:
        c1, c2 = st.columns([1.2, 0.8], gap="large")
        with c1:
            st.image(img_preview, width=OUTPUT_WIDTH, use_container_width=False)
        with c2:
            st.markdown("#### Hasil")
            st.markdown(f"<div class='result-label'>{pred['label']}</div>", unsafe_allow_html=True)
            with st.expander("Detail"):
                st.markdown(f"- **Confidence:** `{pred['conf']:.2f}`")
                st.markdown(f"- **Raw prob (Car):** `{pred['raw_car']:.2f}`")
                st.caption(f"Latency: {pred['elapsed']:.2f}s")

def page_about():
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.info("Aplikasi sederhana untuk deteksi wajah dan klasifikasi kendaraan.")
    st.markdown(f"**Disusun oleh:** {AUTHOR_NAME}  \n**NPM:** {AUTHOR_NPM}  \n**Universitas Syiah Kuala**")

def page_help():
    st.markdown("### ❓ Panduan Penggunaan")
    st.markdown("1️⃣ Login/Daftar terlebih dahulu.  \n2️⃣ Masuk ke halaman **Detect** atau **Classify**.  \n3️⃣ Upload gambar (JPG/PNG).  \n4️⃣ Klik tombol proses.  \n5️⃣ Lihat hasil di panel kanan.")

# =========================
# RENDER
# =========================
if not st.session_state.auth_user:
    page_auth()
else:
    topbar()
    navbar()

    page = st.session_state.page
    if page == "home":
        page_home()
    elif page == "detect":
        page_detect()
    elif page == "classify":
        page_classify()
    elif page == "about":
        page_about()
    else:
        page_help()

    st.markdown(
        f"<div class='footer'>© {datetime.datetime.now().year} — {AUTHOR_NAME} • NPM {AUTHOR_NPM} • Universitas Syiah Kuala</div>",
        unsafe_allow_html=True
    )
