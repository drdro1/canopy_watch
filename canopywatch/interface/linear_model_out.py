import os
from canopywatch.ml_logic.preprocessor import image_preprocessing
from canopywatch.params import BASE_DIR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

def linear_model_predict(image_name: str, model: keras.Model):
    np_image = image_preprocessing(os.path.join(BASE_DIR, "data_predict", "images", image_name))
    pred = model.predict(np.expand_dims(np_image, axis=0))
    return pred.item()

def unet_model_predict(image_name: str, model: keras.Model):
    np_image = image_preprocessing(os.path.join(BASE_DIR, "data_predict", "images", image_name))
    y_pred = model.predict(np.expand_dims(np_image, axis=0))
    return y_pred
