import os
import numpy as np

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

SENTINEL_CLIENT_ID = os.environ.get("SENTINEL_CLIENT_ID")
SENTINEL_CLIENT_SECRET = os.environ.get("SENTINEL_CLIENT_SECRET")

# Define the dynamic base path based on the current file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths built from BASE_DIR
DATA_PATH = os.path.join(BASE_DIR, "data")
DATA_PATH_PREDICT = os.path.join(BASE_DIR, "data_predict", "images")
DATA_PATH_RESULT = os.path.join(BASE_DIR, "data_predict", "masks")
META_DATA_PATH = os.path.join(DATA_PATH, "forest_segmented", "meta_data.csv")
MASKS_PATH = os.path.join(DATA_PATH, "forest_segmented", "masks")
IMAGES_PATH = os.path.join(DATA_PATH, "forest_segmented", "images")

# Path for outputs (for example, trained models, logs, etc.)
LOCAL_REGISTRY_PATH = os.path.join(BASE_DIR, "training_outputs")


################   PARAMS for U-Net model in "canopywatch/ml_logic/model_unet_with_params.py"   ################

NUMBER_OF_IMAGES = 100              # the number of images to train on (int), do = None to train on every image in meta_data.csv

# In the model - here DOWN blocks/sample refer to Encoder and UP blocks/sample refer to Decoder
DOWN_FILTERS = [8,16,32,64]         # number of filters in convolutions for down blocks - is a List (if NUM_BLOCKS= 4, place 4 numbers per list, ex: [32,64,128,512] )
UP_FILTERS = [256,128,64,32]        # number of filters in convolutions for up blocks - is a List (if NUM_BLOCKS= 4, place 4 numbers per list, ex: [512,256,128,64] )
KERNEL_SIZE: int =3                 # Kernel size used in our Conv2D and Conv2DTranspose layers
TWICE=False                         # To use 1 or 2 Conv2D layers in each block
DROPOUT=0.3                         # Dropout value in each block (down and up sample)
NUM_BLOCKS=4                        # Number of down and up sample blocks wanted
N_BOTTLENECK = 512                  # number of filters in convolutions for the bottleneck
ACTIVATION = "sigmoid"              # activation function for our output (a Conv2D layer)
STRIDES = (1, 1)                    # strides for our output (a Conv2D layer)
SUMMARY = True                      # to see model.summary

# In compile
OPTIMIZER="adam"                    # optimizer used in compiler
LOSS="binary_crossentropy"          # loss function
METRICS="accuracy"                  # metrics

# To fit the model
BATCH_SIZE = 32                     # batch_size of .fit
PATIENCE = 20                       # patience for EarlyStopping
EPOCHS = 500                        # number of epochs wanted
RESTORE_BEST_WEIGHTS = True         # RESTORE_BEST_WEIGHTS in EarlyStopping


PARAMS_DICT = {
               "NUMBER_OF_IMAGES":NUMBER_OF_IMAGES,"DOWN_FILTERS":DOWN_FILTERS,"UP_FILTERS":UP_FILTERS,"KERNEL_SIZE":KERNEL_SIZE,
               "TWICE":TWICE,"DROPOUT":DROPOUT,"NUM_BLOCKS":NUM_BLOCKS,"N_BOTTLENECK":N_BOTTLENECK,"ACTIVATION":ACTIVATION,
               "BATCH_SIZE":BATCH_SIZE,"PATIENCE":PATIENCE,"EPOCHS":EPOCHS,"RESTORE_BEST_WEIGHTS":RESTORE_BEST_WEIGHTS, "SUMMARY":SUMMARY,
               "OPTIMIZER":OPTIMIZER,"LOSS":LOSS, "STRIDES":STRIDES,"METRICS":METRICS
               }

###########################   Predictions   ###########################

MODEL_NAME = "20241201-205530.h5"       # name of model to use in predictions (has to be saved locally or to Google Cloud Storage)
MODEL_LOAD_TYPE = 'gcs'                 # to load a model from 'local' or 'gcs'(Google Cloud Storage)
SAVE_MODEL_TYPE = True                  # True to save model to GCS (will always save locally)
IMAGES_LOAD_TYPE = True                 # True to load images to predict from GCS (else they are loaded locally)
SAVE_PREDICT_TYPE = True                # True to save predicted masks to GCS (will always save locally)
FOREST_ON_IMAGE = 0.5                   # Amount of forest wanted on our binary mask (0.5 = 50%, the closer it is to 0 the more forest will be put on the binary mask)
