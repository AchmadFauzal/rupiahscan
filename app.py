import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image
import os

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="RupiScan AI",
    page_icon="💰",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #f5f7fb;
}

/* TITLE */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 5px;
}

.main-title span {
    color: #7c3aed;
}

.subtitle {
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 20px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #ececec;
}

/* METRIC */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#7c3aed,#8b5cf6);
    border-radius: 18px;
    padding: 20px;
    color: white;
    box-shadow: 0 4px 15px rgba(124,58,237,0.2);
}

[data-testid="metric-container"] label {
    color: white !important;
}

[data-testid="metric-container"] div {
    color: white !important;
}

/* CARD */
.custom-card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border: 1px solid #ececec;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 12px;
    background: linear-gradient(135deg,#7c3aed,#8b5cf6);
    color: white;
    font-weight: 600;
    font-size: 16px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    opacity: 0.95;
}

/* FOOTER */
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
    padding-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_yolo_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "best.pt")

    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan: {model_path}")
        st.stop()

    try:
        return YOLO(model_path)

    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()

model = load_yolo_model()

# =====================================
# NOMINAL MAP
# =====================================
nominal_map = {
    0: 1000,
    1: 2000,
    2: 5000,
    3: 10000,
    4: 20000,
    5: 50000,
    6: 100000
}

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:

    st.image(
        "https://www.bi.go.id/id/fungsi-utama/pengedaran-uang/penerbitan-uang/Uang%20Tahun%20Emisi/Uang%20Kertas%20TE%202022.jpg",
        use_container_width=True
    )

    st.title("⚙️ Pengaturan")

    mode = st.radio(
        "Metode Input",
        ["📂 Upload Gambar", "📸 Kamera"]
    )

    st.info(
        "Aplikasi AI untuk mendeteksi dan menghitung uang Rupiah menggunakan YOLOv8."
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        0.4
    )

# =====================================
# PROCESS FUNCTION
# =====================================
def process_image(image):

    results = model(
        image,
        conf=conf_threshold,
        verbose=False
    )

    boxes = results[0].boxes

    img_plotted = results[0].plot(
        labels=True,
        conf=True
    )

    img_rgb = cv2.cvtColor(
        img_plotted,
        cv2.COLOR_BGR2RGB
    )

    classes = boxes.cls.cpu().numpy().astype(int)

    counter = Counter(classes)

    total = 0

    details = []

    for kelas in sorted(
        counter.keys(),
        reverse=True
    ):

        jumlah = counter[kelas]

        nominal = nominal_map.get(kelas, 0)

        subtotal = nominal * jumlah

        total += subtotal

        details.append({
            "nominal": nominal,
            "jumlah": jumlah,
            "subtotal": subtotal
        })

    return img_rgb, details, total

# =====================================
# HEADER
# =====================================
st.markdown("""
<div class="main-title">
    💰 Rupi<span>Scan</span> AI
</div>

<div class="subtitle">
    Solusi Cerdas Deteksi dan Hitung Uang Rupiah Secara Otomatis
</div>
""", unsafe_allow_html=True)

st.divider()

# =====================================
# INPUT IMAGE
# =====================================
source_img = None

if mode == "📂 Upload Gambar":

    uploaded_file = st.file_uploader(
        "Pilih file gambar...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        source_img = Image.open(uploaded_file)

else:

    camera_image = st.camera_input(
        "Ambil Foto Uang"
    )

    if camera_image:

        source_img = Image.open(camera_image)

# =====================================
# MAIN CONTENT
# =====================================
if source_img is not None:

    col1, col2 = st.columns([3,2])

    image_np = np.array(source_img)

    with st.spinner("🔍 Sedang menganalisis gambar..."):

        result_img, details, total = process_image(image_np)

    # =================================
    # LEFT
    # =================================
    with col1:

        st.markdown("""
        <div class="custom-card">
        """, unsafe_allow_html=True)

        st.subheader("🖼️ Hasil Deteksi")

        st.image(
            result_img,
            use_container_width=True,
            caption="Hasil Pemindaian AI"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # =================================
    # RIGHT
    # =================================
    with col2:

        st.markdown("""
        <div class="custom-card">
        """, unsafe_allow_html=True)

        st.subheader("📊 Ringkasan")

        st.metric(
            label="💵 Total Nilai Uang",
            value=f"Rp {total:,.0f}"
        )

        st.write("")

        if details:

            for item in details:

                with st.expander(f"💸 Rp {item['nominal']:,}"):

                    st.write(
                        f"**Jumlah:** {item['jumlah']} lembar"
                    )

                    st.write(
                        f"**Subtotal:** Rp {item['subtotal']:,}"
                    )

        else:

            st.warning(
                "Uang tidak terdeteksi. "
                "Coba atur threshold atau gunakan gambar yang lebih jelas."
            )

        st.markdown("</div>", unsafe_allow_html=True)

else:

    st.info(
        "📷 Upload gambar atau gunakan kamera untuk memulai deteksi."
    )

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">
    Dibuat dengan ❤️ menggunakan Streamlit & YOLOv8
</div>
""", unsafe_allow_html=True)
