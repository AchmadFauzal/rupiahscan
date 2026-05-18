import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="RupiScan Dashboard",
    page_icon="💰",
    layout="wide"
)

# ==========================================
# MODERN UI CSS (DASHBOARD STYLE)
# ==========================================
st.markdown("""
<style>

body {
    background-color: #f5f7fb;
}

.block-container {
    padding: 2rem 2rem;
    max-width: 1200px;
}

.card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}

.title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    color: #666;
    margin-bottom: 20px;
}

.stMetric {
    background: white;
    padding: 10px;
    border-radius: 12px;
}

img {
    border-radius: 12px;
}

.footer {
    text-align: center;
    padding: 20px;
    color: gray;
    font-size: 13px;
    margin-top: 40px;
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
# NOMINAL + COLOR MAP
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

color_map = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (0, 0, 255),
    4: (128, 0, 128),
    5: (0, 165, 255),
    6: (255, 255, 0)
}

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("💰 RupiScan AI")
    st.info("Deteksi uang Rupiah menggunakan YOLOv8")
    mode = st.radio("Input", ["Upload Gambar", "Kamera"])

# ==========================================
# DETECTION ENGINE
# ==========================================
def detect(image):

    results = model(image)
    boxes = results[0].boxes

    output = image.copy()

    box_thickness = 2
    font_scale = 0.6

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        nominal = nominal_map.get(cls, 0)
        label = f"Rp {nominal:,}"

        color = color_map.get(cls, (0, 255, 0))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

        cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 8, y1), color, -1, cv2.LINE_AA)

        cv2.putText(output, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 2, cv2.LINE_AA)

    classes = boxes.cls.cpu().numpy().astype(int)
    counter = Counter(classes)

    total = 0
    details = []

    for k in sorted(counter.keys()):
        jumlah = counter[k]
        nominal = nominal_map.get(k, 0)

        subtotal = jumlah * nominal
        total += subtotal

        details.append({
            "nominal": nominal,
            "jumlah": jumlah,
            "subtotal": subtotal
        })

    return output, details, total

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="title">💰 RupiScan Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI untuk deteksi & menghitung uang Rupiah secara otomatis</div>', unsafe_allow_html=True)

st.divider()

# ==========================================
# INPUT
# ==========================================
img = None

col_input, col_info = st.columns([1, 1])

with col_input:
    if mode == "Upload Gambar":
        file = st.file_uploader("Upload gambar uang", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file)
    else:
        cam = st.camera_input("Ambil gambar")
        if cam:
            img = Image.open(cam)

with col_info:
    st.markdown("### 📊 Info")
    st.info("Upload gambar atau ambil foto untuk mulai deteksi uang")

# ==========================================
# PROCESS RESULT
# ==========================================
if img is not None:

    img = img.convert("RGB")
    image_np = np.array(img)

    with st.spinner("🔍 Mendeteksi uang..."):
        result, details, total = detect(image_np)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    st.divider()

    # ==========================================
    # MAIN DASHBOARD LAYOUT
    # ==========================================
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### 📷 Original")
        st.image(image_np, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Detection Result")
        st.metric("💵 Total Uang", f"Rp {total:,.0f}")
        st.image(result, use_container_width=True)

    with col3:
        st.markdown("### 📦 Detail")
        if details:
            for d in details:
                st.write(f"💵 Rp {d['nominal']:,}")
                st.write(f"Jumlah: {d['jumlah']} lembar")
                st.write(f"Subtotal: Rp {d['subtotal']:,}")
                st.write("---")
        else:
            st.warning("Tidak ada uang terdeteksi")

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    RupiScan AI • Clean Dashboard UI Version
</div>
""", unsafe_allow_html=True)
