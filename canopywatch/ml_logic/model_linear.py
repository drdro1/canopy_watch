import pandas as pd
from canopywatch.ml_logic.preprocessor import image_preprocessing, image_preprocessing_bulk, read_metadata_forest_file_to_df
from keras.callbacks import EarlyStopping
from canopywatch.params import DATA_PATH

# KERAS
from tensorflow.keras import models, layers, utils, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def conv_block(x, n_filters, kernel_size: int = 3):
   x = layers.Conv2D(n_filters, kernel_size, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters, kernel_size: int = 3, dropout = 0.2):
   f = conv_block(x, n_filters, kernel_size)
   p = layers.MaxPool2D(2)(f)
   if dropout > 0:
        p = layers.Dropout(dropout)(p)
   return n_filters, f, p

def build_linear_model(params: dict):
    inputs = layers.Input((256, 256, 3))

    x = inputs
    n_filters = params['start_filters']
    feature_maps = []

    for i in range(params['num_blocks']):
        if i == 0:
            n_filters, f, x = downsample_block(x, n_filters, params['kernel_size'], params['dropout'])
        else:
            n_filters, f, x = downsample_block(x, n_filters*2, params['kernel_size'], params['dropout'])
        feature_maps.append(f)

    flatten = layers.Flatten()(x)
    dense = layers.Dense(n_filters, activation="relu")(flatten)
    dense_dropout = layers.Dropout(params['dropout'])(dense)
    outputs = layers.Dense(1, activation="linear")(dense_dropout)
    model = Model([inputs], [outputs], name="Linear")
    return model

def compile_linear_model(model, params):
    model.compile(loss='mse', optimizer=params['optimizer'], metrics=['mae'])
    return model

def fit_linear_model(model, X_train, y_train, patience=15, batch_size = 32, epochs=200, validation_split=0.15, restore_best_weights = True):
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights = restore_best_weights, verbose=1)

    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[es],
                        validation_split = validation_split)
    return history
