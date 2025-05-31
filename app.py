import streamlit as st
import pydicom
import numpy as np
import cv2
from PIL import Image
import os
import io

# Configure Streamlit page
st.set_page_config(page_title="PyDICOM Web", layout="wide")
st.title("DICOM Viewer with Brightness, Contrast, Filters & GIF Creation")

# Upload multiple DICOM files
uploaded_files = st.file_uploader("Upload multiple DICOM (.dcm) files", type="dcm", accept_multiple_files=True)

# CLAHE function
def apply_clahe(img):
    img_normalized = (img - img.min()) / (img.max() - img.min())
    img_8bit = np.uint8(img_normalized * 255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_8bit)

# Image processing pipeline
def process_image(img, brightness, contrast, filter_type, threshold, zoom):
    img_normalized = (img - img.min()) / (img.max() - img.min())
    img_processed = np.clip(img_normalized * contrast + (brightness / 100), 0, 1)

    # Filters
    if filter_type == 'Gaussian':
        img_processed = cv2.GaussianBlur(img_processed, (5, 5), 0)
    elif filter_type == 'Sharpen':
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_processed = cv2.filter2D(img_processed, -1, kernel)
    elif filter_type == 'Edge':
        img_8bit = np.uint8(img_processed * 255)
        img_processed = cv2.Canny(img_8bit, 50, 150)
    elif filter_type == 'CLAHE':
        img_processed = apply_clahe(img)
    elif filter_type == 'Threshold':
        img_8bit = np.uint8(img_processed * 255)
        _, thresh_img = cv2.threshold(img_8bit, threshold, 255, cv2.THRESH_BINARY)
        img_processed = thresh_img / 255.0
    elif filter_type == 'None':
        pass  # لا تطبق أي فلتر

    # Zooming
    if zoom != 1.0:
        h, w = img_processed.shape[:2]
        center_x, center_y = w // 2, h // 2
        new_w, new_h = int(w / zoom), int(h / zoom)
        x0, x1 = max(0, center_x - new_w // 2), min(w, center_x + new_w // 2)
        y0, y1 = max(0, center_y - new_h // 2), min(h, center_y + new_h // 2)
        img_processed = img_processed[y0:y1, x0:x1]
        img_processed = cv2.resize(img_processed, (w, h))

    return img_processed

# Create animated GIF
def create_slice_gif(dcm_files):
    images = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        img = ds.pixel_array.astype(float)
        img_norm = (img - img.min()) / (img.max() - img.min())
        img_8bit = np.uint8(img_norm * 255)
        pil_img = Image.fromarray(img_8bit).convert("L")
        images.append(pil_img)

    gif_bytes = io.BytesIO()
    images[0].save(gif_bytes, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0)
    gif_bytes.seek(0)
    return gif_bytes

# Convert metadata to string safely
def safe_str(value):
    try:
        return str(value)
    except:
        return "Unknown"

# Main logic
if uploaded_files:
    temp_files = {}
    for file in uploaded_files:
        temp_files[file.name] = file.read()

    for filename, content in temp_files.items():
        with open(filename, "wb") as f:
            f.write(content)

    dcm_files = list(temp_files.keys())

    # Try sorting by InstanceNumber or SliceLocation
    try:
        dcm_files.sort(key=lambda f: pydicom.dcmread(f).InstanceNumber)
    except:
        try:
            dcm_files.sort(key=lambda f: pydicom.dcmread(f).SliceLocation)
        except:
            st.warning("Warning: Using default file order")

    st.success(f"Loaded {len(dcm_files)} DICOM slices")

    # Sidebar controls
    st.sidebar.header("Controls")
    slice_num = st.sidebar.slider("Slice Number", 0, len(dcm_files) - 1, 0)
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
    filter_type = st.sidebar.selectbox("Filter", ['None', 'Gaussian', 'Sharpen', 'Edge', 'CLAHE', 'Threshold'])
    zoom = st.sidebar.slider("Zoom", 0.5, 3.0, 1.0, 0.1)
    st.sidebar.header("Threshold (Only applies if selected)")
    threshold = st.sidebar.slider("Threshold Level", 0, 255, 100)

    try:
        ds = pydicom.dcmread(dcm_files[slice_num])
        img = ds.pixel_array.astype(float)
    except Exception as e:
        st.error(f"Failed to load DICOM slice: {e}")
        st.stop()

    img_processed = process_image(img, brightness, contrast, filter_type, threshold, zoom)

    st.image(img_processed, clamp=True, caption=f"Slice {slice_num+1} / {len(dcm_files)}: {dcm_files[slice_num]}")

    metadata = {}
    for tag in ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'SliceThickness', 'PixelSpacing']:
        val = getattr(ds, tag, 'Unknown')
        metadata[tag] = safe_str(val)

    st.subheader("DICOM Metadata")
    st.table(metadata)

    if st.button("Create Slice GIF"):
        gif_data = create_slice_gif(dcm_files)
        st.success("GIF created!")
        st.image(gif_data)
        st.download_button("Download GIF", data=gif_data, file_name="dicom_slices.gif", mime="image/gif")

    # Cleanup
    for filename in dcm_files:
        if os.path.exists(filename):
            os.remove(filename)
else:
    st.info("Upload multiple DICOM (.dcm) files to begin.")
