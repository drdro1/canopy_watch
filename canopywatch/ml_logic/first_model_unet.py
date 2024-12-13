from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

"""
To setup your model you can:
- run model_unet (and add parameters to get summary or bigger model) - to get compiled model
- then fit_model_with_earlystopping (and change parameters to what is needed) - to fit it
"""


#This layer creates 2 convolution kernels that are convolved with their input layer
def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

#Encoder part of our Unet model - Convolution followed by a Maxpool and then Dropout to control overfitting
def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

#Decoder part of our Unet model - Conv2DTranspose (to "unpack our image"),
#                                 followed by a concatenate (to keep the information that was lost during the encoder),
#                                 then Dropout to control overfitting,
#                                 and then Convolution
def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

#Model Unet optimized
def build_fast_unet_model():
    inputs = layers.Input((256,256,3))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 8)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 16)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 32)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 64)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)
    # unet model with Keras Functional API
    unet_model = Model([inputs], [outputs], name="U-Net")
    return unet_model

#Model Unet with more neurons
def build_slow_unet_model():
    inputs = layers.Input((256,256,3))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)
    # unet model with Keras Functional API
    unet_model = Model([inputs], [outputs], name="U-Net")
    return unet_model

#Compile model
def model_compile(model, optimizer=Adam(), loss="binary_crossentropy", metrics="accuracy"):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

# CHOOSE YOUR UNET MODEL ('fast', or 'slow')
# Get information - to get summary call the function with summary='Yes'
# Here we build our model and apply the compiler
# Returns the model compiled
def model_unet(type='fast', summary="None"):
    if type=='slow':
        unet_model = build_slow_unet_model()
        if summary=='Yes':
            unet_model.summary()
        unet_model_compiled = model_compile(unet_model)
        return unet_model_compiled
    elif type=='fast':
        unet_model = build_fast_unet_model()
        if summary=='Yes':
            unet_model.summary()
        unet_model_compiled = model_compile(unet_model)
        return unet_model_compiled
    else:
        print('Wrong type for Unet.')

#Fit the model with EarlyStopping - returns history and model compiled
def fit_model_with_earlystopping(unet_model_compiled, X_train, y_train, validation_data=None, patience=3, epochs=10, restore_best_weights = True):
    es = EarlyStopping(patience=patience, restore_best_weights = restore_best_weights)

    history = unet_model_compiled.fit(X_train, y_train, epochs=epochs, validation_data=validation_data, callbacks = [es])
    return history, unet_model_compiled
