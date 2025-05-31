import streamlit as st  # Import Streamlit for creating the web app
import pydicom  # Library to read DICOM medical image files
import numpy as np  # For numerical operations on arrays
import cv2  # OpenCV for image processing
from PIL import Image  # Pillow for image manipulation and saving
import io  # For handling in-memory file operations

# Configure the Streamlit page with a title and wide layout
st.set_page_config(page_title="PyDICOM Web", layout="wide")

# Set the main title of the app
st.title("DICOM Viewer with Brightness, Contrast, Filters & GIF Creation")

# File uploader widget allowing users to upload multiple DICOM (.dcm) files
uploaded_files = st.file_uploader("Upload multiple DICOM (.dcm) files", type="dcm", accept_multiple_files=True)

# Initialize a session state variable to track whether metadata should be shown
if 'show_metadata' not in st.session_state:
    st.session_state.show_metadata = True

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast locally
def apply_clahe(img):
    """
    Enhance image contrast using CLAHE method.
    Steps:
    - Normalize image pixels to 0-1 range.
    - Convert to 8-bit grayscale.
    - Apply CLAHE with set parameters.
    """
    img_normalized = (img - img.min()) / (img.max() - img.min())  # Normalize to 0-1
    img_8bit = np.uint8(img_normalized * 255)  # Convert normalized image to 8-bit grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Create CLAHE object with clip limit and grid size
    return clahe.apply(img_8bit)  # Apply CLAHE and return result

# Main image processing function applying brightness, contrast, filters, thresholding, and zoom
def process_image(img, brightness, contrast, filter_type, threshold, zoom):
    """
    Process the input image by:
    - Normalizing pixel values,
    - Adjusting brightness and contrast,
    - Applying the selected filter,
    - Applying threshold if chosen,
    - Zooming (cropping and resizing).
    """
    # Normalize input image pixels to 0-1 range for consistent processing
    img_normalized = (img - img.min()) / (img.max() - img.min())
    
    # Apply brightness and contrast adjustments
    # brightness is divided by 100 to scale appropriately
    img_processed = np.clip(img_normalized * contrast + (brightness / 100), 0, 1)

    # Apply the chosen filter based on filter_type
    if filter_type == 'Gaussian':
        # Apply Gaussian blur to smooth the image, reducing noise
        img_processed = cv2.GaussianBlur(img_processed, (5,5), 0)
    elif filter_type == 'Sharpen':
        # Sharpen the image using a kernel that enhances edges
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_processed = cv2.filter2D(img_processed, -1, kernel)
    elif filter_type == 'Edge':
        # Use Canny edge detector to highlight edges
        img_8bit = np.uint8(img_processed * 255)  # Convert to 8-bit for Canny
        edges = cv2.Canny(img_8bit, 50, 150)  # Detect edges
        img_processed = edges / 255.0  # Normalize edges back to 0-1
    elif filter_type == 'CLAHE':
        # Apply CLAHE to improve local contrast
        img_processed = apply_clahe(img_processed)
    elif filter_type == 'Threshold':
        # Apply binary thresholding to isolate certain intensity regions
        img_8bit = np.uint8(img_processed * 255)
        _, thresh_img = cv2.threshold(img_8bit, threshold, 255, cv2.THRESH_BINARY)
        img_processed = thresh_img / 255.0

    # Zoom the image by cropping the center and resizing back to original size
    if zoom != 1.0:
        h, w = img_processed.shape[:2]
        center_x, center_y = w // 2, h // 2  # Find center coordinates
        new_w, new_h = int(w / zoom), int(h / zoom)  # Calculate new cropped dimensions
        x0 = max(0, center_x - new_w // 2)  # Calculate cropping boundaries
        x1 = min(w, center_x + new_w // 2)
        y0 = max(0, center_y - new_h // 2)
        y1 = min(h, center_y + new_h // 2)
        img_processed = img_processed[y0:y1, x0:x1]  # Crop the image
        img_processed = cv2.resize(img_processed, (w, h))  # Resize back to original size

    return img_processed  # Return the processed image

# Function to create a GIF animation from multiple DICOM slices
def create_slice_gif(dcm_datasets):
    """
    Convert the list of DICOM slices into an animated GIF.
    Steps:
    - Normalize each slice's pixel data,
    - Convert each slice to an 8-bit PIL image,
    - Save all frames as a GIF in memory,
    - Return the GIF bytes.
    """
    images = []
    for ds in dcm_datasets:
        img = ds.pixel_array.astype(float)
        img_norm = (img - img.min()) / (img.max() - img.min())
        img_8bit = np.uint8(img_norm * 255)
        pil_img = Image.fromarray(img_8bit).convert("L")  # Convert to grayscale PIL image
        images.append(pil_img)

    gif_bytes = io.BytesIO()  # Create in-memory byte stream for GIF
    images[0].save(
        gif_bytes,
        format="GIF",
        save_all=True,
        append_images=images[1:],  # Append remaining images
        duration=100,  # Frame duration in ms
        loop=0  # Loop indefinitely
    )
    gif_bytes.seek(0)  # Reset pointer to start of GIF data
    return gif_bytes

# Utility function to safely convert DICOM metadata values to string
def safe_str(value):
    """
    Convert value to string; if it fails, return 'Unknown'.
    Used to avoid crashes on missing or unusual metadata.
    """
    try:
        return str(value)
    except:
        return "Unknown"

# Main logic when files are uploaded
if uploaded_files:
    # Read all uploaded files into DICOM datasets
    dcm_datasets = []
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()  # Read file content as bytes
        file_like = io.BytesIO(bytes_data)  # Create in-memory file object
        ds = pydicom.dcmread(file_like)  # Parse the DICOM file
        dcm_datasets.append(ds)

    # Attempt to sort the slices by InstanceNumber (usual for multi-slice series)
    # If InstanceNumber not available, try SliceLocation as fallback
    try:
        dcm_datasets.sort(key=lambda ds: ds.InstanceNumber)
    except:
        try:
            dcm_datasets.sort(key=lambda ds: ds.SliceLocation)
        except:
            st.warning("Warning: Using default file order")

    # Inform user about the number of slices loaded
    st.success(f"Loaded {len(dcm_datasets)} DICOM slices")

    # Sidebar widgets for controlling the viewer settings
    st.sidebar.header("Controls")
    slice_num = st.sidebar.slider("Slice Number", 0, len(dcm_datasets) - 1, 0)  # Select which slice to view
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)  # Brightness adjustment slider
    contrast = st.sidebar.slider("Contrast", 0.1, 3.0, 1.0, 0.1)  # Contrast adjustment slider
    filter_type = st.sidebar.selectbox("Filter", ['None', 'Gaussian', 'Sharpen', 'Edge', 'CLAHE', 'Threshold'])  # Filter selection
    zoom = st.sidebar.slider("Zoom", 0.5, 3.0, 1.0, 0.1)  # Zoom slider (cropping and resizing)
    threshold = st.sidebar.slider("Threshold Level", 0, 255, 100)  # Threshold level for binary filter

    # Button to toggle the visibility of metadata table
    if st.sidebar.button("Toggle Metadata"):
        st.session_state.show_metadata = not st.session_state.show_metadata

    # Allow user to select which metadata tags to display
    metadata_options = ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'SliceThickness', 'PixelSpacing']
    selected_metadata = st.sidebar.multiselect("Select Metadata to Display", metadata_options, default=[])

    # Load the pixel data from the selected slice number
    try:
        ds = dcm_datasets[slice_num]
        img = ds.pixel_array.astype(float)  # Convert pixel data to float for processing
    except Exception as e:
        st.error(f"Failed to load DICOM slice: {e}")
        st.stop()

    # Process the image with current brightness, contrast, filter, threshold, and zoom settings
    img_processed = process_image(img, brightness, contrast, filter_type, threshold, zoom)

    # Display the processed image on the main page with a caption showing slice info
    st.image(img_processed, clamp=True, caption=f"Slice {slice_num + 1} / {len(dcm_datasets)}")

    # Show the metadata table if user enabled it
    if st.session_state.show_metadata:
        metadata = {}
        for tag in selected_metadata:
            val = getattr(ds, tag, 'Unknown')  # Get the metadata value safely
            metadata[tag] = safe_str(val)
        st.subheader("DICOM Metadata")
        st.table(metadata)  # Display metadata in a table format

    # Button to create and show a GIF animation of all slices
    if st.button("Create Slice GIF"):
        gif_data = create_slice_gif(dcm_datasets)  # Create the GIF
        st.success("GIF created!")
        st.image(gif_data)  # Show the GIF animation
        # Provide a download button for the GIF file
        st.download_button(
            label="Download GIF",
            data=gif_data,
            file_name="dicom_slices.gif",
            mime="image/gif"
        )

else:
    # If no files uploaded yet, show a prompt message
    st.info("Upload multiple DICOM (.dcm) files to begin.")
