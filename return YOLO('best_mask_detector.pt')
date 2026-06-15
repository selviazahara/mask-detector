app_code = """
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Deteksi Masker YOLOv8", page_icon="😷")

st.title("😷 Deteksi Masker Wajah")
st.markdown("Deteksi otomatis menggunakan YOLOv8")

@st.cache_resource
def load_model():
    return YOLO('runs/detect/train/weights/best.pt')

model = load_model()

# Sidebar
menu = st.sidebar.radio("Pilih Mode", ["📸 Upload Gambar", "📷 Kamera"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** YOLOv8n")
st.sidebar.markdown("**Dataset:** Face Mask Detection")

def proses_deteksi(img):
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Mendeteksi..."):
        results = model.predict(img_bgr, conf=0.5, verbose=False)

    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes
    mask_count = sum(1 for c in boxes.cls if model.names[int(c)] == 'mask')
    no_mask_count = sum(1 for c in boxes.cls if model.names[int(c)] == 'no_mask')

    return annotated_rgb, mask_count, no_mask_count

# ===== MODE UPLOAD =====
if menu == "📸 Upload Gambar":
    st.subheader("Upload Gambar")
    uploaded = st.file_uploader("Pilih gambar", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded)
        annotated_rgb, mask_count, no_mask_count = proses_deteksi(img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Gambar Asli", use_column_width=True)
        with col2:
            st.image(annotated_rgb, caption="Hasil Deteksi", use_column_width=True)

        st.markdown("### Hasil:")
        col1, col2 = st.columns(2)
        col1.metric("✅ Pakai Masker", mask_count)
        col2.metric("❌ Tidak Pakai Masker", no_mask_count)

# ===== MODE KAMERA =====
elif menu == "📷 Kamera":
    st.subheader("Deteksi via Kamera")
    st.info("Klik tombol di bawah untuk mengaktifkan kamera")

    img_file = st.camera_input("📷 Ambil Foto")

    if img_file is not None:
        img = Image.open(img_file)
        annotated_rgb, mask_count, no_mask_count = proses_deteksi(img)

        st.image(annotated_rgb, caption="Hasil Deteksi", use_column_width=True)

        st.markdown("### Hasil:")
        col1, col2 = st.columns(2)
        col1.metric("✅ Pakai Masker", mask_count)
        col2.metric("❌ Tidak Pakai Masker", no_mask_count)
"""

with open('app.py', 'w') as f:
    f.write(app_code)

print("app.py berhasil diupdate!")
