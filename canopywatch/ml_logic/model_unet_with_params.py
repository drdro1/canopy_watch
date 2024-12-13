import os
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# from tensorflow.keras.optimizers import Adam

from canopywatch.params import *

"""
To setup your model you can:
- change main parameters in params.py

- run model_unet (and add parameters to get summary or bigger model) - to get compiled model
- then fit_model_with_earlystopping (and change parameters to what is needed) - to fit it
"""
#This layer creates 2 convolution kernels that are convolved with their input layer
def double_conv_block(x, n_filters, kernel_size= 3, twice: bool = False):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, kernel_size, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   if twice:
    x = layers.Conv2D(n_filters, kernel_size, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

#Encoder part of our Unet model - Convolution followed by a Maxpool and then Dropout to control overfitting
def downsample_block(x, n_filters, kernel_size = 3, twice: bool = False, dropout = 0.2, maxpool = 2):
   f = double_conv_block(x, n_filters, kernel_size, twice)
   p = layers.MaxPool2D(maxpool)(f)
   if dropout > 0:
        p = layers.Dropout(dropout)(p)
   return f, p

#Decoder part of our Unet model - Conv2DTranspose (to "unpack our image"),
#                                 followed by a concatenate (to keep the information that was lost during the encoder),
#                                 then Dropout to control overfitting,
#                                 and then Convolution
def upsample_block(x, conv_features, n_filters, kernel_size= 3, twice: bool = False, dropout = 0.2):
   # upsample
   x = layers.Conv2DTranspose(n_filters, kernel_size, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(dropout)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters, kernel_size, twice)
   return x

#Model Unet build
def build_unet_model(down_filters = [8,16,32,64], up_filters = [512,256,128,64], kernel_size =3, twice=False, dropout=0.2, num_blocks=4, n_bottleneck = 1024, strides = (1,1),activation = "sigmoid"):

    inputs = layers.Input((256,256,3))
    x = inputs
    down_filters = down_filters
    up_filters = up_filters
    feature_maps = []

    # encoder: contracting path - downsample
    for i in range(num_blocks):
        if i == 0:
            f, x = downsample_block(x, down_filters[i], kernel_size, twice, dropout)
        else:
            f, x = downsample_block(x, down_filters[i], kernel_size, twice, dropout)
        feature_maps.append(f)

    # bottleneck
    x = double_conv_block(x, n_bottleneck, kernel_size, twice)

    # decoder: expanding path - upsample
    for i in range(num_blocks):
        if i == 0:
            x = upsample_block(x, feature_maps[-(i+1)], up_filters[i], kernel_size, twice, dropout)
        else:
            x = upsample_block(x, feature_maps[-(i+1)], up_filters[i], kernel_size, twice, dropout)

    # outputs
    outputs = layers.Conv2D(1, 1, strides=strides, padding="same", activation = activation)(x)
    # unet model with Keras Functional API
    unet_model = Model([inputs], [outputs], name="U-Net")
    return unet_model

#Compile model
def model_compile(model, optimizer="adam", loss="binary_crossentropy", metrics="accuracy"):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

# Get information - to get summary call the function with summary=True
# Here we build our model and apply the compiler
# Returns the model compiled
def model_unet(down_filters = PARAMS_DICT["DOWN_FILTERS"], up_filters = PARAMS_DICT["UP_FILTERS"], kernel_size = PARAMS_DICT["KERNEL_SIZE"], twice=PARAMS_DICT["TWICE"], dropout=PARAMS_DICT["DROPOUT"], num_blocks=PARAMS_DICT["NUM_BLOCKS"], n_bottleneck =PARAMS_DICT["N_BOTTLENECK"], activation =PARAMS_DICT["ACTIVATION"], summary= PARAMS_DICT['SUMMARY'], optimizer=PARAMS_DICT["OPTIMIZER"], loss=PARAMS_DICT["LOSS"],strides=PARAMS_DICT["STRIDES"], metrics=PARAMS_DICT["METRICS"]):
    unet_model = build_unet_model(down_filters = down_filters, up_filters = up_filters, kernel_size = kernel_size, twice=twice, dropout=dropout, num_blocks=num_blocks, n_bottleneck = n_bottleneck, strides=strides, activation = activation)
    if summary==True:
        unet_model.summary()
    unet_model_compiled = model_compile(unet_model, optimizer=optimizer, loss=loss, metrics=metrics)
    return unet_model_compiled

#Fit the model with EarlyStopping - returns history and model compiled
def fit_model_with_earlystopping(unet_model_compiled, X_train, y_train, validation_data=None, batch_size=PARAMS_DICT["BATCH_SIZE"], patience=PARAMS_DICT["PATIENCE"], epochs=PARAMS_DICT['EPOCHS'], restore_best_weights = PARAMS_DICT['RESTORE_BEST_WEIGHTS']):
    es = EarlyStopping(patience=patience, restore_best_weights = restore_best_weights)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    mc = ModelCheckpoint(filepath=os.path.join(LOCAL_REGISTRY_PATH, "checkpoints", f"{timestamp}.h5"),
                         save_best_only=True,
                         monitor='loss',
                         mode='min',
                         verbose=1)

    history = unet_model_compiled.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks = [es, mc])
    return history, unet_model_compiled
