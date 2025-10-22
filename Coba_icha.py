import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ====== CONFIG ======
MODEL_PATH = "model/annisa_laporan4.pt"  # ganti sesuai nama file kamu
CLASS_NAMES = ["real faces", "sketch faces", "synthetic faces"]
IMG_SIZE = 224
DEVICE = torch.device("cpu")

st.set_page_config(page_title="Face Type Classifier (3-class)", layout="centered")
st.title("Face Type Classifier — Real / Sketch / Synthetic")
st.caption("Model: PyTorch .pt — mencoba load TorchScript dulu, jika gagal load state_dict ke ResNet18 (3 kelas).")

# ====== Transforms (asumsi standar ImageNet; ubah jika kamu pakai transform lain saat training) ======
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),  # [0,1], CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ====== Loader model yang robust ======
@st.cache_resource
def load_model():
    # 1) Coba sebagai TorchScript
    try:
        scripted = torch.jit.load(MODEL_PATH, map_location=DEVICE)
        scripted.eval()
        return scripted, "torchscript"
    except Exception as e_script:
        pass  # lanjut coba state_dict

    # 2) Coba sebagai state_dict ke ResNet18 3 kelas
    try:
        base = models.resnet18(weights=None)          # tanpa pretrained head
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, len(CLASS_NAMES))
        state = torch.load(MODEL_PATH, map_location=DEVICE)

        # state bisa wrap di 'state_dict' (mis. saat save from Lightning)
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("model.", "").replace("module.", ""): v
                     for k, v in state["state_dict"].items()}

        # hapus prefix 'module.' kalau disave dari DataParallel
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        base.load_state_dict(state, strict=False)     # strict=False biar toleran
        base.eval()
        return base, "resnet18-state_dict"
    except Exception as e_sd:
        raise RuntimeError(
            f"Gagal memuat model.\n\nTorchScript error: {repr(e_script)}\nState_dict error: {repr(e_sd)}\n"
            "Pastikan path benar dan arsitektur saat training sesuai."
        )

model, load_mode = load_model()
st.info(f"Model loaded as **{load_mode}** from `{MODEL_PATH}` (device: {DEVICE}).")

# ====== Inferensi ======
@torch.no_grad()
def predict(pil_img, topk=3):
    x = preprocess(pil_img.convert("RGB")).unsqueeze(0)  # 1xCxHxW
    logits = model(x.to(DEVICE))
    if isinstance(logits, (list, tuple)):   # kalau model mengeluarkan beberapa output
        logits = logits[0]
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()  # 3, float
    idx_sorted = np.argsort(probs)[::-1][:topk]
    return [(CLASS_NAMES[i], float(probs[i])) for i in idx_sorted], probs

# ====== UI ======
uploaded = st.file_uploader("Upload gambar wajah (jpg/png):", type=["jpg", "jpeg", "png"])
thresh = st.slider("Confidence minimum untuk highlight kelas teratas", 0.0, 0.99, 0.50, 0.01)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Preview", use_container_width=True)

    topk, probs = predict(img, topk=3)

    st.subheader("Probabilitas Kelas")
    for name, p in topk:
        st.write(f"**{name}**: {p:.3f}")
        st.progress(min(max(p, 0.0), 1.0))

    pred_label, pred_conf = topk[0]
    if pred_conf >= thresh:
        st.success(f"Prediksi: **{pred_label}** (conf {pred_conf:.2f})")
    else:
        st.warning(f"Keyakinan model rendah ({pred_conf:.2f}). Coba gambar lain atau turunkan threshold.")
