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
    page_title="RupiScan Pro",
    page_icon="💰",
    layout="wide"
)

# ==========================================
# CSS (FIXED DASHBOARD STYLE)
# ==========================================
st.markdown("""
<style>

.main {
    background-color: #f4f6f9;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}

.footer {
    text-align: center;
    padding: 12px;
    color: gray;
    font-size: 13px;
}

img {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

model = load_model()

# ==========================================
# NOMINAL & COLOR
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
    st.title("💰 RupiScan Pro")
    st.info("Deteksi uang Rupiah pakai YOLOv8 (Laravel-like quality)")
    mode = st.radio("Input", ["Upload", "Kamera"])

# ==========================================
# DETECTION ENGINE (FIXED QUALITY)
# ==========================================
def detect(image):

    results = model(image)
    boxes = results[0].boxes

    output = image.copy()

    h, w = output.shape[:2]

    # FIX: scaling stabil seperti web canvas
    box_thickness = 2
    font_scale = 0.6

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        nominal = nominal_map.get(cls, 0)
        label = f"Rp {nominal:,}"

        color = color_map.get(cls, (0, 255, 0))

        # BOX (sharp seperti Laravel canvas)
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            box_thickness,
            lineType=cv2.LINE_AA
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            2
        )

        # label bg
        cv2.rectangle(
            output,
            (x1, y1 - th - 6),
            (x1 + tw + 8, y1),
            color,
            -1,
            lineType=cv2.LINE_AA
        )

        # text
        cv2.putText(
            output,
            label,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

    # hitung total
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
# TITLE
# ==========================================
st.title("💰 RupiScan AI Pro")
st.markdown("### Hasil Deteksi dibuat seperti dashboard Laravel (sharp & stabil)")

st.divider()

# ==========================================
# INPUT
# ==========================================
img = None

if mode == "Upload":
    up = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)

else:
    cam = st.camera_input("Ambil gambar")
    if cam:
        img = Image.open(cam)

# ==========================================
# PROCESS
# ==========================================
if img is not None:

    img = img.convert("RGB")
    image_np = np.array(img)

    with st.spinner("Detecting..."):
        result, details, total = detect(image_np)

    # FIX: jangan auto resize Streamlit
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📷 Original")
        st.image(image_np, width=500)

    with col2:
        st.markdown("### 🎯 Detection Result")

        st.metric("Total Uang", f"Rp {total:,.0f}")

        # FIX LARAVEL STYLE: fixed width (bukan container width)
        st.image(result, width=500)

        st.write("---")

        if details:
            for d in details:
                st.write(f"💵 Rp {d['nominal']:,} → {d['jumlah']} lembar")
        else:
            st.warning("Tidak terdeteksi")

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    RupiScan Pro • Laravel-like Detection UI
</div>
""", unsafe_allow_html=True)
