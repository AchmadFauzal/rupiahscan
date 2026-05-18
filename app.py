import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="RupiScan",
    page_icon="💰",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.main {
    background-color: #f5f7f9;
}

.stMetric {
    background-color: white;
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

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

model = load_model()

# ==========================================
# MAPPING NOMINAL
# ==========================================
nominal_map = {
    0: 1000,
    1: 2000,
    2: 5000,
    3: 10000,
    4: 20000,
    5: 50000,
    6: 100000
}

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.title("💰 Rupiah Vision AI")

    st.info(
        "Aplikasi AI untuk mendeteksi dan menghitung nominal uang Rupiah menggunakan YOLOv8."
    )

    mode = st.radio(
        "Metode Input",
        ["📂 Upload Gambar", "📸 Kamera"]
    )

# ==========================================
# FUNGSI DETEKSI
# ==========================================
def process_image(image):

    # Prediksi YOLO
    results = model(image)

    boxes = results[0].boxes

    # ==========================================
    # PENTING:
    # JANGAN UBAH WARNA GAMBAR
    # ==========================================

    # Salin gambar asli
    output_image = image.copy()

    # Ambil bounding box manual
    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        nominal = nominal_map.get(cls, 0)

        label = f"Rp {nominal:,}"

        # Bounding box hijau
        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Background label
        cv2.rectangle(
            output_image,
            (x1, y1 - 35),
            (x1 + 180, y1),
            (0, 255, 0),
            -1
        )

        # Text label
        cv2.putText(
            output_image,
            label,
            (x1 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

    # ==========================================
    # HITUNG TOTAL
    # ==========================================

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

    return output_image, details, total

# ==========================================
# MAIN CONTENT
# ==========================================
st.title("💰 Rupiah Vision AI")

st.markdown(
    "### Solusi Cerdas Menghitung Uang Rupiah Secara Cepat dan Akurat"
)

st.divider()

source_img = None

# ==========================================
# INPUT
# ==========================================
if mode == "📂 Upload Gambar":

    uploaded_file = st.file_uploader(
        "Upload gambar uang Rupiah",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        source_img = Image.open(uploaded_file)

else:

    camera_image = st.camera_input("Ambil Foto Uang")

    if camera_image:

        source_img = Image.open(camera_image)

# ==========================================
# JIKA ADA GAMBAR
# ==========================================
if source_img is not None:

    # Pastikan RGB agar tidak error
    source_img = source_img.convert("RGB")

    # Convert ke numpy TANPA mengubah warna
    image_np = np.array(source_img)

    # ==========================================
    # PREVIEW GAMBAR ASLI
    # ==========================================
    st.subheader("🖼️ Preview Gambar Asli")

    st.image(
        image_np,
        caption="Gambar Sebelum Deteksi",
        use_container_width=True
    )

    st.write("")

    # ==========================================
    # DETEKSI
    # ==========================================
    with st.spinner("🔍 Sedang mendeteksi uang..."):

        result_img, details, total = process_image(image_np)

    # ==========================================
    # HASIL
    # ==========================================
    col1, col2 = st.columns([3, 2])

    with col1:

        st.subheader("🎯 Hasil Deteksi Bounding Box")

        st.image(
            result_img,
            caption="Gambar Setelah Deteksi",
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Ringkasan")

        st.metric(
            label="💵 Total Uang",
            value=f"Rp {total:,.0f}"
        )

        st.write("---")

        if details:

            for item in details:

                with st.expander(f"💰 Rp {item['nominal']:,}"):

                    st.write(
                        f"Jumlah : {item['jumlah']} lembar"
                    )

                    st.write(
                        f"Subtotal : Rp {item['subtotal']:,}"
                    )

        else:

            st.warning(
                "Uang tidak terdeteksi."
            )

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    Dibuat oleh Kelompok Ibu Kota Jawa Barat
</div>
""", unsafe_allow_html=True)
