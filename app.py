import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter
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

.stApp {
    background: #f5f7fb;
}

/* HIDE STREAMLIT */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* NAVBAR */
.navbar {
    background: white;
    padding: 18px 35px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

.logo {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
}

.logo span {
    color: #7c3aed;
}

/* CARD */
.card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border: 1px solid #ececec;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg,#7c3aed,#8b5cf6);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 14px;
    font-weight: 600;
    font-size: 16px;
}

.stButton > button:hover {
    opacity: 0.9;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    border: 2px dashed #d1d5db;
    border-radius: 16px;
    padding: 10px;
}

/* RESULT BOX */
.result-box {
    background: linear-gradient(135deg,#7c3aed,#8b5cf6);
    padding: 22px;
    border-radius: 18px;
    color: white;
    margin-top: 18px;
}

.result-title {
    font-size: 16px;
    opacity: 0.9;
}

.result-money {
    font-size: 34px;
    font-weight: 700;
}

/* TABLE */
.money-item {
    background: #f9fafb;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
    border: 1px solid #ececec;
}

.money-top {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    margin-bottom: 6px;
}

.money-sub {
    color: #6b7280;
    font-size: 14px;
}

/* IMAGE */
.result-image img {
    border-radius: 18px;
}

/* EMPTY */
.empty-box {
    text-align: center;
    padding: 60px 20px;
    color: #6b7280;
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
def load_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "best.pt")

    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan: {model_path}")
        st.stop()

    return YOLO(model_path)

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
        💰 Rupi<span>Scan</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================
# LAYOUT
# =====================================
left_col, right_col = st.columns([1.3, 1])

# =====================================
# LEFT SIDE
# =====================================
with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📷 Upload Gambar")

    uploaded_file = st.file_uploader(
        "Pilih gambar uang Rupiah",
        type=["jpg", "jpeg", "png"]
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        0.4
    )

    detect_btn = st.button("🔍 Deteksi Sekarang")

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Preview Gambar",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# RIGHT SIDE
# =====================================
with right_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("💵 Hasil Deteksi")

    if uploaded_file and detect_btn:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        with st.spinner("Mendeteksi uang..."):

            results = model(
                image_np,
                conf=conf_threshold,
                verbose=False
            )

        img = image_np.copy()

        names = model.names

        money_data = {}

        total_money = 0

        total_sheet = 0

        # =========================
        # LOOP DETECTION
        # =========================
        for box in results[0].boxes:

            cls_id = int(box.cls[0])

            label = names[cls_id]

            try:
                nominal = int(label)

            except:
                continue

            if label not in money_data:

                money_data[label] = {
                    "count": 0,
                    "subtotal": 0
                }

            money_data[label]["count"] += 1

            money_data[label]["subtotal"] += nominal

            total_money += nominal

            total_sheet += 1

            # BOX
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            color = colors.get(
                label,
                (0,255,0)
            )

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

        # =================================
        # TOTAL BOX
        # =================================
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">
                Total Nilai Uang
            </div>

            <div class="result-money">
                Rp {total_money:,.0f}
            </div>

            <div style="margin-top:8px;">
                Total Lembar: {total_sheet}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # =================================
        # DETAILS
        # =================================
        if money_data:

            sorted_money = sorted(
                money_data.items(),
                key=lambda x: int(x[0]),
                reverse=True
            )

            for money, item in sorted_money:

                st.markdown(f"""
                <div class="money-item">

                    <div class="money-top">
                        <span>💸 Rp {int(money):,}</span>
                        <span>{item['count']} Lembar</span>
                    </div>

                    <div class="money-sub">
                        Subtotal: Rp {item['subtotal']:,}
                    </div>

                </div>
                """, unsafe_allow_html=True)

        else:

            st.warning("Tidak ada uang terdeteksi.")

        st.write("")

        # =================================
        # RESULT IMAGE
        # =================================
        result_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_rgb,
            caption="Hasil Deteksi AI",
            use_container_width=True
        )

    else:

        st.markdown("""
        <div class="empty-box">

            <h3>Belum Ada Hasil</h3>

            <p>
                Upload gambar lalu klik
                <b>Deteksi Sekarang</b>
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">
    Dibuat dengan ❤️ menggunakan Streamlit & YOLOv8
</div>
""", unsafe_allow_html=True)
