import numpy as np
import io
import shutil
import time
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from canopywatch.ml_logic.registry import load_model
from canopywatch.params import DATA_PATH_PREDICT
from canopywatch.interface.linear_model_out import linear_model_predict, unet_model_predict
from canopywatch.ml_logic.model import percent_cover_from_binary_mask
from pathlib import Path


app = FastAPI()
app.state.linear_model = load_model('linear_model_2.h5', "gcs")
app.state.unet_model = load_model("20241201-205530.h5", "gcs")

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict-percent")
def predict_from_picture(file: UploadFile = File(...)):
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    file_path = Path(DATA_PATH_PREDICT) / f"{timestamp}.jpeg"

    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Make the prediction
    pred = linear_model_predict(file_path, app.state.linear_model)

    # Clean up the file if it exists
    if file_path.exists():
        file_path.unlink()

    return {"pred": pred}

@app.post("/predict-mask")
def predict_from_picture(file: UploadFile = File(...)):
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    file_path = Path(DATA_PATH_PREDICT) / f"{timestamp}.jpeg"

    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Make the prediction
    mask = unet_model_predict(file_path, app.state.unet_model)

    # Clean up the file if it exists
    if file_path.exists():
        file_path.unlink()

    mask = mask.reshape(256, 256)
    binary_mask = (mask > 0.5).astype("uint8")
    percent_cover = percent_cover_from_binary_mask(binary_mask)

    mask = mask * 255
    mask = mask.astype(np.uint8)

    # Convert mask to an image
    mask_image = Image.fromarray(mask)

    # Encode the image as Base64
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Return both the image and the percent_cover as JSON
    response_data = {
        "percent_cover": percent_cover,
        "image": image_base64
    }

    return JSONResponse(content=response_data)
