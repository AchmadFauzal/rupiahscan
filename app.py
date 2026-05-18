import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
import pandas as pd
import os

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="RupiScan",
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

/* BACKGROUND */
.stApp {
    background: #f5f7fb;
}

/* NAVBAR */
.navbar {
    background: white;
    padding: 20px 30px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

.logo {
    font-size: 34px;
    font-weight: 700;
}

.logo span {
    color: #7c3aed;
}

.subtitle {
    color: #6b7280;
    margin-top: 5px;
}

/* CARD */
.card {
    background: white;
    padding: 25px;
    border-radius: 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border: 1px solid #ececec;
}

/* TITLE */
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 20px;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    border: none;
    padding: 14px;
    border-radius: 14px;
    background: linear-gradient(135deg,#7c3aed,#8b5cf6);
    color: white;
    font-weight: 600;
    font-size: 16px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    opacity: 0.95;
}

/* METRIC */
.metric-box {
    background: linear-gradient(135deg,#7c3aed,#8b5cf6);
    padding: 22px;
    border-radius: 18px;
    color: white;
    margin-top: 20px;
}

.metric-title {
    font-size: 15px;
    opacity: 0.9;
}

.metric-money {
    font-size: 34px;
    font-weight: bold;
    margin-top: 10px;
}

.metric-sheet {
    margin-top: 10px;
    font-size: 15px;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 40px;
    color: gray;
}

/* IMAGE */
img {
    border-radius: 16px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# NAVBAR
# =====================================
st.markdown("""
<div class="navbar">
    <div class="logo">
        💰 Rupi<span>Scan</span>
    </div>
    <div class="subtitle">
        Deteksi dan Hitung Uang Rupiah Menggunakan AI YOLOv8
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "best.pt")

    if not os.path.exists(model_path):
        st.error(f"❌ Model tidak ditemukan: {model_path}")
        st.stop()

    try:
        return YOLO(model_path)

    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        st.stop()

model = load_model()

# =====================================
# COLORS
# =====================================
colors = {
    "1000": (255, 0, 0),
    "2000": (0, 255, 0),
    "5000": (0, 255, 255),
    "10000": (255, 255, 0),
    "20000": (255, 0, 255),
    "50000": (0, 128, 255),
    "100000": (0, 0, 255)
}

# =====================================
# DETECTION FUNCTION
# =====================================
def detect_money(image):

    results = model(image, verbose=False)

    img = image.copy()

    names = model.names

    money_data = defaultdict(lambda: {
        "count": 0,
        "subtotal": 0
    })

    total_money = 0
    total_sheet = 0

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        label = names[cls_id]

        try:
            nominal = int(label)
        except:
            continue

        money_data[label]["count"] += 1
        money_data[label]["subtotal"] += nominal

        total_money += nominal
        total_sheet += 1

        # BOX
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = colors.get(label, (0,255,0))

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # TEXT
        cv2.putText(
            img,
            f"Rp{label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    return img, money_data, total_money, total_sheet

# =====================================
# LAYOUT
# =====================================
col1, col2 = st.columns([1.1, 1])

# =====================================
# LEFT SIDE
# =====================================
with col1:

    st.markdown("""
    <div class="card">
        <div class="section-title">
            📷 Preview Gambar
        </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Pilih Metode Input",
        ["Upload Gambar", "Kamera"],
        horizontal=True
    )

    source_img = None

    if input_mode == "Upload Gambar":

        uploaded_file = st.file_uploader(
            "Pilih gambar uang rupiah",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            source_img = Image.open(uploaded_file)

    else:

        camera_image = st.camera_input("Ambil Foto")

        if camera_image:
            source_img = Image.open(camera_image)

    if source_img is not None:

        st.image(
            source_img,
            use_container_width=True,
            caption="Preview Gambar"
        )

    detect_btn = st.button("🔍 Deteksi Sekarang")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# RIGHT SIDE
# =====================================
with col2:

    st.markdown("""
    <div class="card">
        <div class="section-title">
            💵 Hasil Deteksi
        </div>
    """, unsafe_allow_html=True)

    if detect_btn:

        if source_img is None:

            st.warning("⚠️ Upload gambar atau gunakan kamera terlebih dahulu.")

        else:

            image_np = np.array(source_img)

            with st.spinner("Sedang mendeteksi uang..."):

                result_img, money_data, total_money, total_sheet = detect_money(image_np)

            if len(money_data) == 0:

                st.warning("Tidak ada uang terdeteksi.")

            else:

                # =========================
                # TABLE DATA
                # =========================
                table_data = []

                for money in sorted(
                    money_data.keys(),
                    key=lambda x: int(x),
                    reverse=True
                ):

                    item = money_data[money]

                    table_data.append({
                        "Pecahan": f"Rp {int(money):,}",
                        "Jumlah": item["count"],
                        "Subtotal": f"Rp {item['subtotal']:,}"
                    })

                df = pd.DataFrame(table_data)

                st.table(df)

                # =========================
                # TOTAL BOX
                # =========================
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">
                        Total Uang Terdeteksi
                    </div>

                    <div class="metric-money">
                        Rp {total_money:,}
                    </div>

                    <div class="metric-sheet">
                        Total Lembar: {total_sheet}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 🖼️ Gambar Hasil")

                st.image(
                    result_img,
                    use_container_width=True
                )

    else:

        st.info("Belum ada hasil deteksi.")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">
    Dibuat dengan ❤️ menggunakan Streamlit & YOLOv8
</div>
""", unsafe_allow_html=True)
