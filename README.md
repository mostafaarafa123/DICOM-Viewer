
 Hi,Names:mostafa tayel :Aly saad   :Omar tawfik  :Abdallah hisham: Mohamed abdelfattah
    ID:202402876        :202403246  :202401118    :202400592       : 202400144
   and this was our first big project using PyDICOM.


##  Project Goal
Build a web app that allows users to upload multiple `.dcm` files (DICOM), view them as slices, adjust brightness/contrast, apply filters (CLAHE, Gaussian Blur, Edge Detection), zoom in/out, and generate animated GIFs from the slices.



##  Uploading Files
```python
uploaded_files = st.file_uploader("Upload multiple DICOM (.dcm) files", type="dcm", accept_multiple_files=True)
```
 I forgot to add `accept_multiple_files=True` at first, so I couldn't upload more than one file. Fixing that made everything work.

  
## Image Processing Functions

###  CLAHE Filter
```python
def apply_clahe(img):
    img_normalized = (img - img.min()) / (img.max() - img.min())
    img_8bit = np.uint8(img_normalized * 255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img_8bit)
```
 I initially applied CLAHE to a float image and got a completely black output. The fix was to normalize and convert the image to 8-bit.

---

## Brightness and Contrast
```python
img_processed = np.clip(img_normalized * contrast + (brightness/100), 0, 1)
```
 Brightness didn't work at first. I was adjusting the value, but nothing changed!  
 Solution: use `np.clip()` after adjustment to keep pixel values within [0,1].


### Filters
```python
if filter_type == 'Edge Detection':
    img_8bit = np.uint8(img_processed * 255)
    img_processed = cv2.Canny(img_8bit, 50, 150)
```
 Edge Detection failed and gave black images until I converted the image to 8-bit before applying Canny.


###  Zooming
```python
if zoom != 1.0:
    # crop from center and resize
```
At first, zooming messed up the image. Then I realized I had to crop from the center and then resize.


##  GIF Creation
```python
images[0].save("output.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
```
I had crashes trying to create GIFs because I forgot to convert images to 8-bit. After fixing it, GIF creation worked smoothly.


##  Sorting and Metadata
```python
dcm_files.sort(key=lambda f: pydicom.dcmread(f).InstanceNumber)
```
Slices were showing out of order. The fix was to sort using DICOM tags like `InstanceNumber` or `SliceLocation`.

---

### Metadata Display
```python
val = getattr(ds, tag, 'Unknown')
```
Some DICOM fields were missing, so I used `getattr()` to avoid attribute errors.


##  Download Buttons and Cleanup
```python
st.download_button("Download GIF", data=gif_bytes, file_name="dicom_slices.gif")
```
I forgot to add a download button at first—easy fix.

```python
for filename in dcm_files:
    if os.path.exists(filename):
        os.remove(filename)
```
Important for cleaning up temporary files.


##  Lessons Learned
- Always check data types (float vs. uint8).###I struggled a lot to understand this concept, but when I finally got it, I saw how much it affects the entire program. It made a big difference.
- Be careful with image processing (cropping, resizing, etc.).
- OpenCV filters need specific formats.
- Brightness/contrast adjustments must be clipped.
- Slice order matters – sort using metadata.
- Step-by-step debugging is the best teacher.

---

 That’s my journey with this project. Hopefully, it helps anyone else starting with medical image processing!
