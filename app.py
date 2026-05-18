import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Rupiah Vision AI",
    page_icon="💰",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #f5f7f9;
}

.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    color: gray;
    padding: 10px;
    background: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_yolo_model():
    return YOLO("best_model.pt")

model = load_yolo_model()

# =========================
# MAPPING NOMINAL
# =========================
nominal_map = {
    0: 1000,
    1: 2000,
    2: 5000,
    3: 10000,
    4: 20000,
    5: 50000,
    6: 100000
}

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.title("💰 Rupiah Vision AI")

    st.info(
        "Aplikasi AI untuk mendeteksi dan menghitung nominal uang Rupiah secara otomatis menggunakan YOLOv8."
    )

    mode = st.radio(
        "Pilih Metode Input",
        ["📂 Upload Gambar", "📸 Kamera"]
    )

# =========================
# PROSES DETEKSI
# =========================
def process_image(image):

    # YOLO Prediction
    results = model(image)

    boxes = results[0].boxes

    # =========================
    # PENTING:
    # TIDAK MENGUBAH WARNA ASLI
    # =========================

    plotted_img = results[0].plot()

    # Tetap gunakan BGR asli agar warna uang tidak berubah
    plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

    classes = boxes.cls.cpu().numpy().astype(int)

    counter = Counter(classes)

    total = 0
    details = []

    for kelas in sorted(counter.keys()):

        jumlah = counter[kelas]

        nominal = nominal_map.get(kelas, 0)

        subtotal = nominal * jumlah

        total += subtotal

        details.append({
            "nominal": nominal,
            "jumlah": jumlah,
            "subtotal": subtotal
        })

    return plotted_img, details, total

# =========================
# MAIN CONTENT
# =========================
st.title("💰 Rupiah Vision AI")

st.markdown(
    "### Solusi Cerdas Menghitung Uang Rupiah Secara Cepat dan Akurat"
)

st.divider()

source_img = None

# =========================
# INPUT GAMBAR
# =========================
if mode == "📂 Upload Gambar":

    uploaded_file = st.file_uploader(
        "Pilih gambar uang...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        source_img = Image.open(uploaded_file)

else:

    camera_image = st.camera_input("Ambil Foto Uang")

    if camera_image:
        source_img = Image.open(camera_image)

# =========================
# JIKA ADA GAMBAR
# =========================
if source_img is not None:

    # Pastikan RGB agar tidak error 4 channel
    source_img = source_img.convert("RGB")

    image_np = np.array(source_img)

    col1, col2 = st.columns([3, 2])

    with st.spinner("🔍 Sedang menganalisis gambar..."):

        result_img, details, total = process_image(image_np)

    # =========================
    # HASIL DETEKSI
    # =========================
    with col1:

        st.subheader("🖼️ Hasil Deteksi")

        st.image(
            result_img,
            caption="Hasil Pemindaian AI",
            use_container_width=True
        )

    # =========================
    # RINGKASAN
    # =========================
    with col2:

        st.subheader("📊 Ringkasan Nominal")

        st.metric(
            label="💵 Total Uang",
            value=f"Rp {total:,.0f}"
        )

        st.write("---")

        if details:

            for item in details:

                with st.expander(f"💰 Rp {item['nominal']:,}"):

                    st.write(f"Jumlah : {item['jumlah']} lembar")

                    st.write(f"Subtotal : Rp {item['subtotal']:,}")

        else:

            st.warning(
                "Uang tidak terdeteksi. "
                "Pastikan gambar jelas dan pencahayaan cukup."
            )

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    Dibuat dengan ❤️ untuk Indonesia
</div>
""", unsafe_allow_html=True)
