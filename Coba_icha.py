# Coba_icha.py
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path

# =========================
# KONFIGURASI & KONSTANTA
# =========================
CLASS_NAMES = ["real faces", "sketch faces", "synthetic faces"]
DEVICE = torch.device("cpu")
IMG_SIZE = 224

st.set_page_config(page_title="Face Type Classifier (3-class)", layout="centered")
st.title("🧠 Face Type Classifier — Real / Sketch / Synthetic")
st.caption("Memuat model .pt (TorchScript atau state_dict) dan memprediksi 3 kelas wajah.")

# =========================
# UTIL: cari file model .pt
# =========================
def auto_find_model() -> str:
    # cari di folder 'model/' dulu, kalau tidak ada, cari di root repo
    candidates = list(Path("model").glob("*.pt")) + list(Path(".").glob("*.pt"))
    return str(candidates[0]) if candidates else ""

# =========================
# TRANSFORM (samakan dg training)
# =========================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =========================
# BUILDER ARSITEKTUR
# =========================
def build_model(arch_name: str, num_classes: int):
    """
    Bangun model kosong sesuai arsitektur pilihan, ganti head ke num_classes.
    Tambahkan arsitektur lain jika perlu.
    """
    arch_name = arch_name.lower().strip()
    if arch_name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch_name == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch_name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch_name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if arch_name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if hasattr(models, "efficientnet_b0") and arch_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m

    # fallback
    raise ValueError(f"Arsitektur '{arch_name}' belum di-handle. Pilih salah satu dari menu.")

# =========================
# LOAD MODEL (robust + UI error)
# =========================
@st.cache_resource
def load_model(model_path: str, arch_name: str):
    """
    1) Coba TorchScript: torch.jit.load
    2) Jika gagal, bangun model arsitektur 'arch_name', lalu load state_dict
    Bila tetap gagal, tampilkan error di UI & hentikan app.
    """
    # Validasi path
    if not model_path or not Path(model_path).exists():
        st.error(f"❌ File model tidak ditemukan: `{model_path}`")
        st.stop()

    # --- 1) TorchScript ---
    ts_err = None
    try:
        scripted = torch.jit.load(model_path, map_location=DEVICE)
        scripted.eval()
        return scripted, "torchscript"
    except Exception as e:
        ts_err = e  # simpan error untuk ditampilkan nanti

    # --- 2) state_dict ke arsitektur yang dipilih ---
    sd_err = None
    try:
        base = build_model(arch_name, len(CLASS_NAMES))
        state = torch.load(model_path, map_location=DEVICE)

        # kadang tersimpan di key 'state_dict' (Lightning)
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("model.", "").replace("module.", ""): v
                     for k, v in state["state_dict"].items()}

        # hapus prefix 'module.' (DataParallel)
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        base.load_state_dict(state, strict=False)  # strict=False biar toleran bbp key
        base.eval()
        return base, f"{arch_name}-state_dict"

    except Exception as e:
        sd_err = e
        # Tampilkan error rinci di UI agar ketahuan masalah aslinya
        st.error("❌ Gagal memuat model sebagai TorchScript maupun state_dict.")
        with st.expander("Lihat detail error TorchScript"):
            st.exception(ts_err)
        with st.expander("Lihat detail error State_dict"):
            st.exception(sd_err)
        st.stop()

# =========================
# PREDIKSI
# =========================
@torch.no_grad()
def predict(pil_img, model):
    x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()
    order = np.argsort(probs)[::-1]
    top = [(CLASS_NAMES[i], float(probs[i])) for i in order]
    return top, probs

# =========================
# UI: pilih model & arsitektur
# =========================
st.subheader("⚙️ Pengaturan Model")
default_path = auto_find_model()
model_path = st.text_input("Path file model (.pt):", value=default_path, help="Contoh: model/Annisa_Humaira_Laporan_4.pt")
arch_choice = st.selectbox(
    "Arsitektur (gunakan arsitektur yang sama saat training jika file adalah state_dict):",
    ["resnet18", "resnet34", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"],
    index=0
)

load_btn = st.button("Muat Model")

# Cache akan membuat model tetap tersimpan sampai app direstart / param berubah
if load_btn or (model_path and default_path and model_path == default_path):
    model, load_mode = load_model(model_path, arch_choice)
    st.success(f"✅ Model dimuat sebagai **{load_mode}** dari `{model_path}`")

    st.subheader("🖼️ Prediksi Gambar")
    uploaded = st.file_uploader("Upload gambar (jpg/png):", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Confidence minimum:", 0.0, 1.0, 0.5, 0.05)

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Gambar diunggah", use_container_width=True)

        topk, probs = predict(img, model)

        st.markdown("### 📊 Probabilitas Kelas")
        for label, p in topk:
            st.write(f"**{label}** : {p:.3f}")
            st.progress(min(max(p, 0.0), 1.0))

        best_label, best_prob = topk[0]
        if best_prob >= threshold:
            st.success(f"Prediksi: **{best_label}** (confidence {best_prob:.2f})")
        else:
            st.warning(f"Keyakinan rendah ({best_prob:.2f}). Coba gambar lain atau turunkan threshold.")
else:
    st.info("Pilih/isi path model lalu klik **Muat Model** untuk mulai.")
