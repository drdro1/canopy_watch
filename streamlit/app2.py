import streamlit as st
from PIL import Image
from pathlib import Path
import datetime
import requests
import base64

FASTAPI_URL = "http://localhost:8000/predict-from-picture"

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

def get_image_files(country: str, type: str):
    dir_name = "pics" + "/" + country + "_" + type
    print('+++++++++++++++++++' + dir_name + '+++++++++++++++++++++')
    image_files = list(Path(dir_name).glob("*.jpeg"))
    return image_files

# Buttons to select project type
project_type = st.radio(
    "Select Project Type",
    ('Check Deforestation Project', 'Check Reforestation Project')
)

def get_table_html(title:str, image_files):
    html = "<h3>" + title + "</h3>"
    html += "<table style='width:100%; text-align:center;'><tr>"
    for i, image_file in enumerate(image_files):
        # Convert the image to base64
        encoded_img = encode_image_to_base64(image_file)

        # Add the image and button in a cell
        html += "<td style='padding:10px;'><img src='data:image/jpeg;base64," + encoded_img + "' style='border:1px solid #ddd; border-radius:4px; width:150px; height:150px;'><br><button style='margin-top:10px; padding:5px 10px;'>Select</button></td>"

     # Close the last row and the table
    html += "</tr></table>"
    return html

# Start the HTML for the table
columns_per_row = 4

countries = ['Cameroon', 'Brazil', 'Borneo']
title_iterator = 0

html = get_table_html('Cameroon', get_image_files('cameroon', 'forest'))
html += get_table_html('Brazil', get_image_files('brazil', 'forest'))
html += get_table_html('Borneo', get_image_files('borneo', 'forest'))

# Display the HTML
st.markdown(html, unsafe_allow_html=True)

# Handle the "Check Deforestation Project" button
if project_type == 'Check Deforestation Project':
    st.write("### Select the region for the deforestation project:")
    # Buttons for different regions
    if st.button('Cameroon'):
        show_images('Cameroon')


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}

        try:
            response = requests.post(FASTAPI_URL, files={"file": uploaded_file})
            response.raise_for_status()
            prediction = response.json()

            st.write("Prediction:", prediction)
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
