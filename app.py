import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
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
# CSS
# =====================================
st.markdown("""
<style>

.main {
    background: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* NAVBAR */
.navbar {
    background: white;
    padding: 18px 28px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 2px 15px rgba(0,0,0,0.05);
}

.logo {
    font-size: 32px;
    font-weight: 700;
}

.logo span {
    color: #7c3aed;
}

/* CARD */
.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 2px 15px rgba(0,0,0,0.05);
    border: 1px solid #ececec;
}

/* TITLE */
.title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 18px;
}

/* TABLE */
.result-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}

.result-table th {
    background: #f3f4f6;
    padding: 12px;
    text-align: left;
}

.result-table td {
    padding: 12px;
    border-bottom: 1px solid #e5e7eb;
}

/* TOTAL BOX */
.total-box {
    margin-top: 20px;
    background: linear-gradient(135deg,#7c3aed,#8b5cf6);
    color: white;
    padding: 20px;
    border-radius: 18px;
}

.total-money {
    font-size: 32px;
    font-weight: bold;
}

.footer {
    text-align: center;
    margin-top: 30px;
    color: gray;
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
    <div style="color:gray;">
        Deteksi dan Hitung Uang Rupiah
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
        st.error(f"Model tidak ditemukan: {model_path}")
        st.stop()

    try:
        return YOLO(model_path)

    except Exception as e:
        st.error(f"Gagal load model: {e}")
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
col1, col2 = st.columns([1.2,1])

# =====================================
# PREVIEW
# =====================================
with col1:

    st.markdown("""
    <div class="card">
        <div class="title">📷 Preview</div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Pilih Input",
        ["Upload Gambar", "Kamera"]
    )

    source_img = None

    if mode == "Upload Gambar":

        uploaded_file = st.file_uploader(
            "Pilih gambar",
            type=["jpg","jpeg","png"]
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
            use_container_width=True
        )

    detect_btn = st.button("🔍 Deteksi")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# RESULT
# =====================================
with col2:

    st.markdown("""
    <div class="card">
        <div class="title">💵 Hasil Deteksi</div>
    """, unsafe_allow_html=True)

    if detect_btn:

        if source_img is None:

            st.warning("Upload gambar atau gunakan kamera.")

        else:

            image_np = np.array(source_img)

            with st.spinner("Sedang mendeteksi..."):

                result_img, money_data, total_money, total_sheet = detect_money(image_np)

            if len(money_data) == 0:

                st.warning("Tidak ada uang terdeteksi.")

            else:

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

                for money in sorted(
                    money_data.keys(),
                    key=lambda x: int(x),
                    reverse=True
                ):

                    item = money_data[money]

                    table_html += f"""
                    <tr>
                        <td>Rp {int(money):,}</td>
                        <td>{item['count']}</td>
                        <td>Rp {item['subtotal']:,}</td>
                    </tr>
                    """

                table_html += "</tbody></table>"

                st.markdown(
                    table_html,
                    unsafe_allow_html=True
                )

                st.markdown(f"""
                <div class="total-box">
                    <div>Total Lembar: {total_sheet}</div>
                    <div class="total-money">
                        Rp {total_money:,}
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
    Dibuat dengan Streamlit + YOLOv8
</div>
""", unsafe_allow_html=True)
