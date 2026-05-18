import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
import tempfile
import os

# =====================================
# CONFIG PAGE
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

.main {
    background: #f5f7fb;
}

/* NAVBAR */
.navbar {
    background: white;
    padding: 18px 35px;
    border-radius: 16px;
    margin-bottom: 25px;
    box-shadow: 0 2px 15px rgba(0,0,0,0.05);
}

.logo {
    display: flex;
    align-items: center;
    gap: 10px;
}

.logo-icon {
    width: 45px;
    height: 45px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6d28d9, #8b5cf6);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.logo-text {
    font-size: 28px;
    font-weight: 700;
}

.logo-purple {
    color: #7c3aed;
}

.subtitle {
    color: #6b7280;
    margin-top: 4px;
    font-size: 14px;
}

/* CARD */
.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border: 1px solid #ececec;
}

/* TITLE */
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
    color: #111827;
}

/* RESULT TABLE */
.result-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}

.result-table th {
    background: #f3f4f6;
    padding: 12px;
    text-align: left;
    font-size: 14px;
}

.result-table td {
    padding: 12px;
    border-bottom: 1px solid #e5e7eb;
    font-size: 15px;
}

/* TOTAL BOX */
.total-box {
    margin-top: 20px;
    background: linear-gradient(135deg, #7c3aed, #8b5cf6);
    color: white;
    padding: 22px;
    border-radius: 18px;
}

.total-title {
    font-size: 15px;
    opacity: 0.9;
}

.total-money {
    font-size: 34px;
    font-weight: bold;
    margin-top: 10px;
}

.total-sheet {
    margin-top: 8px;
    font-size: 15px;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #8b5cf6);
    color: white;
    border: none;
    padding: 12px;
    border-radius: 12px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    opacity: 0.95;
}

/* IMAGE */
.preview-img {
    border-radius: 16px;
    border: 1px solid #e5e7eb;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 35px;
    color: #9ca3af;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

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
# NAVBAR
# =====================================
st.markdown("""
<div class="navbar">
    <div class="logo">
        <div class="logo-icon">💰</div>
        <div>
            <div class="logo-text">
                Rupi<span class="logo-purple">Scan</span>
            </div>
            <div class="subtitle">
                Deteksi dan Hitung Uang Rupiah Menggunakan AI
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================
# DETECT FUNCTION
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
        conf = float(box.conf[0])

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
            0.8,
            color,
            2
        )

    return img, money_data, total_money, total_sheet

# =====================================
# LAYOUT
# =====================================
left_col, right_col = st.columns([1.2, 1])

# =====================================
# LEFT SIDE
# =====================================
with left_col:

    st.markdown("""
    <div class="card">
        <div class="section-title">
            📷 Preview
        </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Pilih Input",
        ["Upload Gambar", "Kamera"],
        horizontal=True
    )

    source_image = None

    if input_mode == "Upload Gambar":

        uploaded_file = st.file_uploader(
            "Pilih gambar uang",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            source_image = Image.open(uploaded_file)

    else:

        camera_image = st.camera_input("Ambil Foto")

        if camera_image:
            source_image = Image.open(camera_image)

    if source_image is not None:
        st.image(
            source_image,
            use_container_width=True,
            caption="Preview Gambar",
            output_format="PNG"
        )

    detect_button = st.button("🔍 Deteksi Sekarang")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# RIGHT SIDE
# =====================================
with right_col:

    st.markdown("""
    <div class="card">
        <div class="section-title">
            💵 Hasil Deteksi
        </div>
    """, unsafe_allow_html=True)

    if detect_button:

        if source_image is None:
            st.warning("Silakan upload gambar atau gunakan kamera.")
        else:

            image_np = np.array(source_image)

            with st.spinner("Sedang mendeteksi uang..."):

                result_img, money_data, total_money, total_sheet = detect_money(image_np)

            st.success("Deteksi selesai!")

            # TABLE
            if len(money_data) > 0:

                table_html = """
                <table class="result-table">
                    <thead>
                        <tr>
                            <th>Pecahan</th>
                            <th>Jumlah</th>
                            <th>Subtotal</th>
                        </tr>
                    </thead>
                    <tbody>
                """

                for money in sorted(money_data.keys(), key=lambda x: int(x), reverse=True):

                    item = money_data[money]

                    table_html += f"""
                    <tr>
                        <td>Rp {int(money):,}</td>
                        <td>{item['count']}</td>
                        <td>Rp {item['subtotal']:,}</td>
                    </tr>
                    """

                table_html += """
                    </tbody>
                </table>
                """

                st.markdown(table_html, unsafe_allow_html=True)

                # TOTAL BOX
                st.markdown(f"""
                <div class="total-box">
                    <div class="total-title">Total Uang Terdeteksi</div>
                    <div class="total-money">
                        Rp {total_money:,}
                    </div>
                    <div class="total-sheet">
                        Total Lembar: {total_sheet}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.warning("Tidak ada uang terdeteksi.")

            st.markdown("### 🖼️ Gambar Hasil")

            st.image(
                result_img,
                use_container_width=True,
                output_format="PNG"
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
