import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="RupiScan",
    page_icon="💰",
    layout="wide"
)

# ====================================
# CUSTOM CSS
# ====================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-color: #f4f7fb;
}

/* hide streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* navbar */
.navbar {
    background: white;
    padding: 20px 30px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

.logo {
    font-size: 30px;
    font-weight: 700;
    color: #111827;
}

.logo span {
    color: #7c3aed;
}

/* card */
.card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border: 1px solid #ececec;
}

/* upload */
[data-testid="stFileUploader"] {
    border: 2px dashed #d1d5db;
    border-radius: 16px;
    padding: 15px;
}

/* button */
.stButton>button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 14px;
    background: linear-gradient(135deg,#7c3aed,#8b5cf6);
    color: white;
    font-size: 16px;
    font-weight: 600;
}

.stButton>button:hover {
    opacity: 0.92;
}

/* result box */
.total-box {
    background: linear-gradient(135deg,#7c3aed,#8b5cf6);
    color: white;
    padding: 22px;
    border-radius: 18px;
    margin-top: 18px;
}

.total-title {
    font-size: 16px;
    opacity: 0.9;
}

.total-money {
    font-size: 36px;
    font-weight: 700;
}

/* money item */
.money-item {
    background: #f9fafb;
    border: 1px solid #ececec;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.money-top {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
}

.money-sub {
    margin-top: 6px;
    color: #6b7280;
    font-size: 14px;
}

/* image */
.result-image img {
    border-radius: 18px;
}

/* footer */
.footer {
    text-align: center;
    margin-top: 35px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# ====================================
# LOAD MODEL
# ====================================
@st.cache_resource
def load_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "best.pt")

    if not os.path.exists(model_path):
        st.error("❌ File best.pt tidak ditemukan")
        st.stop()

    if os.path.getsize(model_path) == 0:
        st.error("❌ File best.pt kosong / corrupt")
        st.stop()

    try:
        model = YOLO(model_path)
        return model

    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        st.stop()

model = load_model()

# ====================================
# COLORS
# ====================================
colors = {
    "1000": (255, 0, 0),
    "2000": (0, 255, 0),
    "5000": (0, 255, 255),
    "10000": (255, 255, 0),
    "20000": (255, 0, 255),
    "50000": (0, 128, 255),
    "100000": (0, 0, 255)
}

# ====================================
# NAVBAR
# ====================================
st.markdown("""
<div class="navbar">
    <div class="logo">
        💰 Rupi<span>Scan</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================
# LAYOUT
# ====================================
left_col, right_col = st.columns([1.1, 1])

# ====================================
# LEFT
# ====================================
with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📤 Upload Gambar")

    uploaded_file = st.file_uploader(
        "Pilih gambar uang",
        type=["jpg", "jpeg", "png"]
    )

    conf = st.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        0.4
    )

    detect_btn = st.button("🔍 Deteksi Sekarang")

    if uploaded_file:

        preview = Image.open(uploaded_file)

        st.image(
            preview,
            caption="Preview Gambar",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ====================================
# RIGHT
# ====================================
with right_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("💵 Hasil Deteksi")

    if uploaded_file and detect_btn:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        with st.spinner("Mendeteksi uang..."):

            results = model(
                image_np,
                conf=conf,
                verbose=False
            )

        img = image_np.copy()

        names = model.names

        money_data = {}

        total_money = 0

        total_sheet = 0

        # ====================================
        # DETECTION LOOP
        # ====================================
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

            # box
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

            # text
            cv2.putText(
                img,
                f"Rp{label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # ====================================
        # TOTAL BOX
        # ====================================
        st.markdown(f"""
        <div class="total-box">

            <div class="total-title">
                Total Nilai Uang
            </div>

            <div class="total-money">
                Rp {total_money:,.0f}
            </div>

            <div style="margin-top:8px;">
                Total Lembar: {total_sheet}
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # ====================================
        # DETAILS
        # ====================================
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

            st.warning("Tidak ada uang terdeteksi")

        st.write("")

        # ====================================
        # RESULT IMAGE
        # ====================================
        result_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_rgb,
            caption="Hasil Deteksi",
            use_container_width=True
        )

    else:

        st.info(
            "Upload gambar lalu klik Deteksi Sekarang"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ====================================
# FOOTER
# ====================================
st.markdown("""
<div class="footer">
    Dibuat dengan ❤️ menggunakan YOLOv8 & Streamlit
</div>
""", unsafe_allow_html=True)
