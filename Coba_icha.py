import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ========================
# KONFIGURASI DASAR
# ========================
MODEL_PATH = "model/Annisa_Humaira_Laporan4.pt"  # pastikan path dan nama file benar
CLASS_NAMES = ["real faces", "sketch faces", "synthetic faces"]
DEVICE = torch.device("cpu")

st.set_page_config(page_title="Face Type Classifier", layout="centered")
st.title("🧠 Face Type Classifier — Real / Sketch / Synthetic")
st.caption("Aplikasi ini menggunakan model PyTorch (.pt) untuk mengklasifikasikan tipe wajah.")

# ========================
# PREPROCESSING
# ========================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========================
# LOAD MODEL (robust)
# ========================
@st.cache_resource
def load_model():
    ts_err = None
    sd_err = None

    # --- 1. Coba load sebagai TorchScript ---
    try:
        scripted = torch.jit.load(MODEL_PATH, map_location=DEVICE)
        scripted.eval()
        return scripted, "torchscript"
    except Exception as e:
        ts_err = e  # simpan error pertama

    # --- 2. Coba load sebagai state_dict ke ResNet18 ---
    try:
        base = models.resnet18(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, len(CLASS_NAMES))

        state = torch.load(MODEL_PATH, map_location=DEVICE)

        # jika file mengandung key 'state_dict'
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("model.", "").replace("module.", ""): v
                     for k, v in state["state_dict"].items()}

        # hapus prefix 'module.' kalau ada
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        base.load_state_dict(state, strict=False)
        base.eval()
        return base, "resnet18-state_dict"

    except Exception as e:
        sd_err = e
        raise RuntimeError(
            "❌ Gagal memuat model.\n"
            f"TorchScript error: {repr(ts_err)}\n"
            f"State_dict error: {repr(sd_err)}\n"
            "➡️ Periksa path file model, arsitektur, dan format penyimpanan."
        )

model, load_mode = load_model()
st.success(f"✅ Model berhasil dimuat sebagai **{load_mode}** dari `{MODEL_PATH}`")

# ========================
# FUNGSI PREDIKSI
# ========================
@torch.no_grad()
def predict(pil_img, topk=3):
    img_tensor = preprocess(pil_img.convert("RGB")).unsqueeze(0)
    outputs = model(img_tensor.to(DEVICE))

    if isinstance(outputs, (list, tuple)):  # kalau output berupa tuple/list
        outputs = outputs[0]

    probs = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()
    idx_sorted = np.argsort(probs)[::-1][:topk]
    return [(CLASS_NAMES[i], float(probs[i])) for i in idx_sorted], probs

# ========================
# STREAMLIT DASHBOARD
# ========================
uploaded = st.file_uploader("📤 Upload gambar wajah (jpg/png):", type=["jpg", "jpeg", "png"])
threshold = st.slider("Confidence minimum (tampilkan jika di atas nilai ini):", 0.0, 1.0, 0.5, 0.05)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    topk, probs = predict(image, topk=3)

    st.subheader("📊 Hasil Prediksi")
    for label, p in topk:
        bar_color = "🟩" if p >= threshold else "⬜"
        st.write(f"{bar_color} **{label}** : {p:.3f}")
        st.progress(min(max(p, 0.0), 1.0))

    best_label, best_prob = topk[0]
    if best_prob >= threshold:
        st.success(f"Prediksi utama: **{best_label}** (confidence: {best_prob:.2f})")
    else:
        st.warning(f"Keyakinan rendah ({best_prob:.2f}). Coba gambar lain atau turunkan threshold.")
