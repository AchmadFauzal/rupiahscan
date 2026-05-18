import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="RupiScan",
    page_icon="💰",
    layout="wide"
)

# ==========================================
# CUSTOM CSS (UI MODERN)
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

.stImage img {
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
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
    font-size: 13px;
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
# NOMINAL MAP
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
    st.title("💰 RupiScan")

    st.info("AI untuk deteksi & menghitung uang Rupiah menggunakan YOLOv8")

    mode = st.radio(
        "Metode Input",
        ["📂 Upload Gambar", "📸 Kamera"]
    )

# ==========================================
# DETEKSI FUNCTION
# ==========================================
def process_image(image):

    results = model(image)
    boxes = results[0].boxes

    output = image.copy()

    h, w = output.shape[:2]

    box_thickness = max(2, int(w * 0.0025))
    font_scale = max(0.5, w / 2200)

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        nominal = nominal_map.get(cls, 0)
        label = f"Rp {nominal:,}"

        # bounding box kecil & clean
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            box_thickness
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            2
        )

        # label background dinamis
        cv2.rectangle(
            output,
            (x1, y1 - th - 8),
            (x1 + tw + 10, y1),
            (0, 200, 0),
            -1
        )

        cv2.putText(
            output,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2
        )

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

    # resize output biar seimbang UI
    output = cv2.resize(output, (0, 0), fx=0.85, fy=0.85)

    return output, details, total

# ==========================================
# TITLE
# ==========================================
st.title("💰 RupiScan AI")
st.markdown("### Deteksi & Hitung Uang Rupiah Secara Otomatis")

st.divider()

# ==========================================
# INPUT
# ==========================================
source_img = None

if mode == "📂 Upload Gambar":
    uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if uploaded:
        source_img = Image.open(uploaded)

else:
    camera = st.camera_input("Ambil foto")

    if camera:
        source_img = Image.open(camera)

# ==========================================
# PROCESS
# ==========================================
if source_img is not None:

    source_img = source_img.convert("RGB")
    image_np = np.array(source_img)

    with st.spinner("🔍 Mendeteksi uang..."):
        result_img, details, total = process_image(image_np)

    # ==========================================
    # LAYOUT BALANCED
    # ==========================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Preview")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("🎯 Hasil Deteksi")

        st.metric("💵 Total Uang", f"Rp {total:,.0f}")

        st.image(result_img, use_container_width=True)

        st.write("---")

        if details:
            for d in details:
                with st.expander(f"Rp {d['nominal']:,}"):
                    st.write(f"Jumlah: {d['jumlah']} lembar")
                    st.write(f"Subtotal: Rp {d['subtotal']:,}")
        else:
            st.warning("Tidak ada uang terdeteksi")

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    RupiScan AI • YOLOv8 Object Detection
</div>
""", unsafe_allow_html=True)
