import streamlit as st# Streamlit for creating web app interface
import pydicom# Library to handle DICOM files
import numpy as np# Library to handle DICOM files
import cv2# OpenCV for image processing
from PIL import Image# Pillow for image manipulation and saving as GIF
import os# File system operations
import io# To create byte stream for GIFs
# Configure the Streamlit page layout
st.set_page_config(page_title="PyDICOM Web", layout="wide")
st.title("DICOM Viewer with Brightness, Contrast, Filters & GIF Creation")

# --- Upload multiple DICOM files --- #
uploaded_files = st.file_uploader("Upload multiple DICOM (.dcm) files", type="dcm", accept_multiple_files=True)
# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(img):
    """Apply CLAHE on 8-bit grayscale image"""
    img_normalized = (img - img.min()) / (img.max() - img.min()) # Normalize image to 0-1
    img_8bit = np.uint8(img_normalized * 255) # Convert to 8-bit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Create CLAHE object
    return clahe.apply(img_8bit)
# Image processing function
# This function handles brightness, contrast, filters, thresholding, and zoom
def process_image(img, brightness, contrast, filter_type, threshold, zoom):
    """Fixed processing pipeline"""
    # Normalize and adjust brightness/contrast
    img_normalized = (img - img.min()) / (img.max() - img.min())# Normalize to 0-1
    img_processed = np.clip(img_normalized * contrast + (brightness/100), 0, 1)# Brightness/contrast adjustment

    # Apply selected filter
    if filter_type == 'Gaussian':
        img_processed = cv2.GaussianBlur(img_processed, (5,5), 0)# Smooth the image
    elif filter_type == 'Sharpen':
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_processed = cv2.filter2D(img_processed, -1, kernel)# Sharpen edges
    elif filter_type == 'Edge':
        img_8bit = np.uint8(img_processed * 255)
        img_processed = cv2.Canny(img_8bit, 50, 150)# Detect edges
    elif filter_type == 'CLAHE':
        img_processed = apply_clahe(img)# CLAHE on original image (not brightness-adjusted)
    elif filter_type == 'Threshold':
        img_8bit = np.uint8(img_processed * 255)
        _, thresh_img = cv2.threshold(img_8bit, threshold, 255, cv2.THRESH_BINARY)
        img_processed = thresh_img / 255.0  # # Normalize again for display (0 to 1)

    # Apply zoom by cropping center and resizing
    if zoom != 1.0:
        h, w = img_processed.shape[:2]
        center_x, center_y = w//2, h//2
        new_w, new_h = int(w/zoom), int(h/zoom)# Calculate the new width and height for zoomed cropping by dividing original dimensions by zoom factor
        x0, x1 = max(0, center_x-new_w//2), min(w, center_x+new_w//2)# Calculate the cropping boundaries to keep the crop centered around the image center,
        y0, y1 = max(0, center_y-new_h//2), min(h, center_y+new_h//2)# ensuring the crop region stays within image bounds (0 to width or height)
        img_processed = img_processed[y0:y1, x0:x1]# Crop
        img_processed = cv2.resize(img_processed, (w, h))# Resize to original size

    return img_processed

def create_slice_gif(dcm_files):
    images = []# List to store PIL images for each DICOM slice
    for f in dcm_files:
        ds = pydicom.dcmread(f)# Read the DICOM file
        img = ds.pixel_array.astype(float) # Extract pixel data as float array
        img_norm = (img - img.min()) / (img.max() - img.min())# Normalize pixel values to 0-1 range
        img_8bit = np.uint8(img_norm * 255) # Convert normalized image to 8-bit grayscale (0-255)
        pil_img = Image.fromarray(img_8bit).convert("L")# Convert to PIL grayscale image
        images.append(pil_img)
    gif_bytes = io.BytesIO()# Create a BytesIO stream to hold GIF data in memory
    # Save all images as an animated GIF in the byte stream, with 100ms per frame, looping forever
    images[0].save(gif_bytes, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0)
    gif_bytes.seek(0)# Move to beginning for reading
    return gif_bytes# Return the in-memory GIF data
## Helper function to safely convert metadata to string (avoid crash)
def safe_str(value):
    """Convert pydicom DSfloat or other special types to string for display"""
    try:
        return str(value)
    except:
        return "Unknown"
#only runs if files are uploaded
if uploaded_files:
    # Save uploaded files temporarily in memory to sort by InstanceNumber or SliceLocation
    temp_files = {}
    for file in uploaded_files:
        temp_files[file.name] = file.read()

    # Write temp files to disk for pydicom to read
    for filename, content in temp_files.items():
        with open(filename, "wb") as f:
            f.write(content)

    dcm_files = list(temp_files.keys()) # Get list of saved filenames

    # Sort DICOM files by InstanceNumber or SliceLocation
    try:
        # Try sorting files based on slice order (InstanceNumber > SliceLocation)
        dcm_files.sort(key=lambda f: pydicom.dcmread(f).InstanceNumber)
    except:
        try:
            # If InstanceNumber is missing, sort by SliceLocation as a fallback
            dcm_files.sort(key=lambda f: pydicom.dcmread(f).SliceLocation)
        except:
            # If both sorting methods fail, warn the user and keep the original order
            st.warning("Warning: Using default file order")
    # Notify the user that the DICOM slices have been loaded successfully
    st.success(f"Loaded {len(dcm_files)} DICOM slices")

    # Sidebar controls
    st.sidebar.header("Controls")
    slice_num = st.sidebar.slider("Slice Number", 0, len(dcm_files)-1, 0)
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
    filter_type = st.sidebar.selectbox("Filter", ['None', 'Gaussian', 'Sharpen', 'Edge', 'CLAHE', 'Threshold'])
    zoom = st.sidebar.slider("Zoom", 0.5, 3.0, 1.0, 0.1)
    st.sidebar.header("You should apply the Threshold filter to do the following order:")
    threshold = st.sidebar.slider("Threshold Level", 0, 255, 100)

    # Load selected DICOM slice
    try:
        ds = pydicom.dcmread(dcm_files[slice_num])# Read the DICOM file
        img = ds.pixel_array.astype(float)# Extract pixel data from the DICOM dataset and convert it to float for processing
    except Exception as e:
        st.error(f"Failed to load DICOM slice: {e}")
        st.stop()# Stop running the app here if loading the DICOM slice fails, to prevent errors later

    # Process image
    img_processed = process_image(img, brightness, contrast, filter_type, threshold, zoom)

    # Show image
    st.image(img_processed, clamp=True, caption=f"Slice {slice_num+1} / {len(dcm_files)}: {dcm_files[slice_num]}")

    # Extract and display metadata safely
    metadata = {}
    for tag in ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'SliceThickness', 'PixelSpacing']:
        val = getattr(ds, tag, 'Unknown')
        # Convert metadata value safely to string to avoid errors before storing
        metadata[tag] = safe_str(val)

    st.subheader("DICOM Metadata")
    st.table(metadata)

    # Button to create GIF
    if st.button("Create Slice GIF"):
        # When user clicks the button, generate an animated GIF from all DICOM slices
        gif_data = create_slice_gif(dcm_files)
        st.success("GIF created!")
        st.image(gif_data) # Display the generated GIF animation in the app
         # Provide a download button so user can save the GIF locally
        st.download_button(
            label="Download GIF",
            data=gif_data,
            file_name="dicom_slices.gif",
            mime="image/gif"
        )

    # Cleanup: remove temp files from disk
    for filename in dcm_files:
        if os.path.exists(filename):
            os.remove(filename)

else:
    # Show this message if no DICOM files have been uploaded yet
    st.info("Upload multiple DICOM (.dcm) files to begin.")
