import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
import io
import requests
import base64

PERCENT_MODEL_URL = "http://localhost:8000/predict-percent"
MASK_MODEL_URL = "http://localhost:8000/predict-mask"

if "show_top" not in st.session_state:
    st.session_state.show_top = True
    st.session_state.project_type = 'Deforestation Project'
    st.session_state.selected_place = None
    st.session_state.selected_image = None
    st.session_state.selected_mask = None


col1, col2 = st.columns([1,3])
with col1:
    st.image('/Users/alejandrosantamaria/code/drdro1/canopy_watch/streamlit/pics/canopy_watch_logo.png', width=150, use_column_width=False)
with col2:
    st.title("Canopy Watch")

# Function to display images for the selected region
def show_images(region):
    if region == 'Cameroon':
        st.image('/Users/eleonoredemarnhac/Documents/google_eath_images/image_2015_2018_cameroon_defo_1_end.jpeg', caption='Cameroon Image 1')

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_image_files(country: str, type: str):
    dir_name = "pics" + "/" + country + "_" + type
    image_files = list(Path(dir_name).glob("*.jpeg"))
    return image_files

def normalize_image(image):
    """Ensure image is converted to RGB and has consistent shape"""
    # Convert to RGB if it's grayscale
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)

    # If image has alpha channel, remove it
    if image.shape[2] == 4:
        image = image[:,:,:3]

    return image

project_type = None

if st.session_state.show_top:
    st.session_state.project_type = st.radio(
    "Select Project Type",
    ('Deforestation Project', 'Reforestation Project')
)

    columns_per_row = 4
    countries = ['Cameroon', 'Brazil', 'Borneo']
    title_iterator = 0

    st.write(f"### Cameroon")
    image_files = get_image_files('cameroon', 'forest')
    cols = st.columns(4)  # 4 columns per table
    for col_id, col in enumerate(cols):
        image_path = image_files[col_id]
        number = col_id + 1

        with col:
            st.image(str(image_path), use_column_width=True, caption=f"Picture {number}")

            if st.button(f"Predict Picture {number}", key=f"button_{number}"):
                try:
                    response = requests.post(MASK_MODEL_URL, files={"file": open(image_path, "rb")})
                    if response.status_code == 200:
                        st.session_state.selected_image = Image.open(str(image_path))#str(image_path)
                        st.session_state.selected_mask = Image.open(io.BytesIO(response.content))
                        st.session_state.show_top = False
                        st.experimental_rerun()

                    else:
                        st.error(f"API call failed: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {e}")

                #     response.raise_for_status()
                #     prediction = response.json()

                #     st.write("Prediction:", prediction)
                # except requests.exceptions.RequestException as e:
                #     st.error(f"Error: {e}")

if not st.session_state.show_top:
    st.title( st.session_state.project_type)
    image1 = st.session_state.selected_image
    image2 = st.session_state.selected_mask

    # Convert images to numpy arrays
    img1_array = normalize_image(np.array(image1))
    img2_array = normalize_image(np.array(image2))

     # Resize images to match
    height, width = min(img1_array.shape[0], img2_array.shape[0]), min(img1_array.shape[1], img2_array.shape[1])
    img1_array = img1_array[:height, :width]
    img2_array = img2_array[:height, :width]

    # Slider to control overlay
    overlay_percentage = st.slider(
        "Image Overlay",
        min_value=0,
        max_value=100,
        value=1,
        step=1
    )

    # Calculate blend
    blend_weight = overlay_percentage / 100
    blended_image = (
        img1_array * (1 - blend_weight) +
        img2_array * blend_weight
    ).astype(np.uint8)

    col1, col2, col3 = st.columns([3, 6, 2])
    with col2:
        st.image(blended_image, width=300, caption="Blended Image")
    with col3:
        if st.button(f"Go Back"):
            st.session_state.show_top = True
            st.experimental_rerun()

    col1, col2, col3 = st.columns([3, 1, 2])
    with col1:
        st.image(st.session_state.selected_image, caption="Original Image", width=200)
    with col3:
        st.image(st.session_state.selected_mask, caption="Binary Mask", width=200)
