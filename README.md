
A VOIR POUR COMPLETER:
- How to use CNN for percentage of forest on image
- streamlit and api ? (how to use, show what folders to create in requirement if needed, etc.)


# 🌳🌳 canopy_watch 🌳🌳

Climate change is still increasing globally (by 24% between 2022 and 2023) despite 140 governments promosing to end it by 2030 increasing global warming.
But there is also a lesser known issue with forest management in reforestation projects. Some of these reforestation projects can be scams and either never take place or plant monoculture, which is less rich for biodiversity than natural ecosystems.

Thats why we came up with the idea of Canopy Watch, a smart tool to monitor forest globally thanks to satellite images. The goal of Canopy Watch is to predict the percentage of forest on satellite images in order to identify the evolution of forest at a given place between two dates.
To do so, we trained a deep learning model (U-Net) on 5000 satellite images of forests and their associated binary masks.

# The Data📝:

The data can be downloaded on Kaggle at:
- 🌐 [https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation/data](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation/data)

**The data folder currently looks like this:**
```bash
.
└── data
    └── Forest Segmented
        ├── meta_data.csv
        └── Forest Segmented
            └── images                  # 🖼️ all our images
            └── masks                   # 🖼️ all our masks
```

**You will need to reorganize the data folder as follows:**
```bash
.
└── data
    ├── meta_data.csv
    └── forest_segmented
        └── images                  # 🖼️ all our images
        └── masks                   # 🖼️ all our masks
```

# Requirements📦:

**To use our project, you must create the following folders, hidden by Gitignore:**
```bash
.
├── .env                            # ⚙️ Where you can place your Google Cloud Storage bucket name (BUCKET_NAME =...)
├── data                            # Our reorganized data folder
│   ├── meta_data.csv
│   └── forest_segmented
│       └── images                  # 🖼️ all our images
│       └── masks                   # 🖼️ all our masks
│
├── training_outputs                # 💻 Folder to save our models and its informations
│   ├── checkpoints                 # Folder to save checkpoints while training the model
│   ├── history                     # Folder to save history of the model
│   ├── metrics                     # Folder to save metrics of the model
│   ├── models                      # Folder to save our model
│   └── params                      # Folder to save parameters of the model
│
└── data_predict                    # 🎯 Folder for our final predictions
    ├── images                      # Folder to place the images whose mask we want to predict
    └── masks                       # Folder
```

**Setting up requirements:**

This command will install the required packages listed in **requirements.txt**:
```bash
pip install -e .
```

# How to use canopy_watch🌳:

We have two possible types of models to use:
- U-Net (Output is a mask of the forest of an image)
- CNN (Output is a percentage of forest of an image)

## For Model U-Net:

📝 Here "GCS" refers to "Google Cloud Storage".

### To train a model:

- make sure you followed the `Requirements📦` part of this README, and created every folder needed
- have the data placed in `data/forest_segmented`
- Use your wanted model parameters in `canopywatch/params.py`
- run the command `make run_preprocess_train_evaluate`

#### a) How to use `canopywatch/params.py`:

- **NUMBER_OF_IMAGES** refers to the number of images to train on (int), do = None to train on every image in meta_data.csv

##### In the model - here DOWN blocks/sample refer to Encoder and UP blocks/sample refer to Decoder
- **DOWN_FILTERS** the number of filters in convolutions for down blocks - is a List (if NUM_BLOCKS= 4, place 4 numbers per list, ex: [32,64,128,512])
- **UP_FILTERS** the number of filters in convolutions for up blocks - is a List (if NUM_BLOCKS= 4, place 4 numbers per list, ex: [512,256,128,64])
- **KERNEL_SIZE** the Kernel size used in our Conv2D and Conv2DTranspose layers (must be an int)
- **TWICE** to use 1 or 2 Conv2D layers in each block
- **DROPOUT** the Dropout value in each block (down and up sample)
- **NUM_BLOCKS** the Number of down and up sample blocks wanted
- **N_BOTTLENECK** the number of filters in convolutions for the bottleneck
- **ACTIVATION** the activation function for our output (a Conv2D layer)
- **STRIDES** the strides for our output (a Conv2D layer)
- **SUMMARY** to see model.summary when running `make run_preprocess_train_evaluate`

##### In compile
- **OPTIMIZER** the optimizer used in our compiler
- **LOSS** our loss function
- **METRICS** the wanted metrics

##### To fit the model
- **BATCH_SIZE** the batch_size of .fit
- **PATIENCE** the patience for EarlyStopping
- **EPOCHS** the number of epochs wanted
- **RESTORE_BEST_WEIGHTS** to RESTORE_BEST_WEIGHTS in EarlyStopping

#### b) Makefile command to create your model:

When you have your wanted parameters you can do:
- `make run_preprocess_train_evaluate`

This will run evaluate() in `canopywatch/ml_logic/model.py` and do in order:
- Preprocess and split our data into train/test/val
- Build and compile a model with the wanted parameters
- Fit our model with EarlyStopping parameters
- Save the results of our model locally
- Save the model locally, and optionally on GCS too
- Evaluate our model on the test set
- Save the results of the evaluate locally

### To make predictions from a model:

- make sure you followed the `Requirements📦` part of this README, and created every folder needed
- have proper images to predict, they must be of shape (256, 256, 3) and placed in your `data_predict/images` folder (locally or on GCS)
- Use your wanted prediction parameters in `canopywatch/params.py`
- run the command `make run_prediction`

The predictions will return masks representing the forest on the inputed images:
- one mask with shades of grey (the more white it is the more likely it is to be forest)
- one binary mask in black and white (with white being forest)

The masks will be saved in `data_predict/masks`.

#### a) Predictions parameters in params.py:

**MODEL_NAME** the name of the model to use (as a .h5 file like: "20241201-205530.h5") in predictions (has to be saved locally or to GCS in `training_outputs/models`)
**MODEL_LOAD_TYPE** choose ="local" or ="gcs" to load a model from 'local' or 'gcs'(Google Cloud Storage)
**SAVE_MODEL_TYPE** choose True to save model to GCS (will always save locally)
**IMAGES_LOAD_TYPE** choose True to load images to predict from GCS (else they are loaded locally)
**SAVE_PREDICT_TYPE** choose True to save the predicted masks to GCS (will always save locally)
**FOREST_ON_IMAGE** the amount of forest wanted on our binary mask (0.5 = 50%, the closer it is to 0 the more forest will be put on the binary mask)

#### b) Makefile command to run predictions:

When you have your wanted parameters you can do:
- `make run_prediction`

This will run prediction() in `canopywatch/ml_logic/model.py` and do in order:
- load our model locally or from GCS
- load the image to predict and preprocess them
- make our predictions and create the wanted masks
- save our predicted masks locally, and optionally on GCS too

## For CNN:

#################################### TO COMPLETE ###################################
