from canopywatch.ml_logic.preprocessor import mask_preprocessing_bulk, image_preprocessing_bulk, create_full_filepath
from canopywatch.ml_logic.model_unet_with_params import model_unet, fit_model_with_earlystopping
from canopywatch.ml_logic.registry import save_model, save_results, load_model, save_predict, load_predict_images

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from canopywatch.params import *

def full_preprocessing():
    img_path = IMAGES_PATH
    masks_path = MASKS_PATH
    data_path = META_DATA_PATH
    data_path = pd.read_csv(data_path)
    file_paths_image, file_paths_mask = create_full_filepath(data_path.iloc[:NUMBER_OF_IMAGES], img_path, masks_path)
    images_preprocessed = image_preprocessing_bulk(file_paths_image)
    masks_preprocessed = mask_preprocessing_bulk(file_paths_mask)

    print("✅ preprocess() done \n")
    return images_preprocessed, masks_preprocessed

def train():
    images_preprocessed, masks_preprocessed = full_preprocessing()
    #Split our data
    X_hold, X_test, y_hold, y_test = train_test_split(images_preprocessed, masks_preprocessed, test_size=0.2, random_state=42) #20% of our data for test
    X_train, X_val, y_train, y_val = train_test_split(X_hold, y_hold, test_size=0.2, random_state=42) #20% of train for val

    #Build and compile our model
    unet_model_compiled = model_unet(down_filters = PARAMS_DICT["DOWN_FILTERS"], up_filters = PARAMS_DICT["UP_FILTERS"], kernel_size = PARAMS_DICT["KERNEL_SIZE"], twice=PARAMS_DICT["TWICE"], dropout=PARAMS_DICT["DROPOUT"], num_blocks=PARAMS_DICT["NUM_BLOCKS"], n_bottleneck =PARAMS_DICT["N_BOTTLENECK"], activation =PARAMS_DICT["ACTIVATION"],strides=PARAMS_DICT["STRIDES"], summary= PARAMS_DICT['SUMMARY'])
    history, unet_model_fit = fit_model_with_earlystopping(unet_model_compiled, X_train, y_train, validation_data=[X_val, y_val], batch_size=PARAMS_DICT["BATCH_SIZE"], patience=PARAMS_DICT["PATIENCE"], epochs=PARAMS_DICT['EPOCHS'], restore_best_weights = PARAMS_DICT['RESTORE_BEST_WEIGHTS'])

    print("✅ train() done \n")

    val_binary_crossentropy = np.min(history.history['loss'])

    params = dict(
        context="train",
        row_count=len(X_train),
    )

    # Save results on the hard drive using ml_logic.registry
    save_results(model_name='Unet_train',params=params, metrics=dict(binary_crossentropy=val_binary_crossentropy), history=history)

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=unet_model_fit,GCS=True)

    return history, unet_model_fit, X_test, y_test


def evaluate():
    history, unet_model_fit, X_test, y_test  = train()

    metrics = unet_model_fit.evaluate(
        x=X_test,
        y=y_test,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    binary_crossentropy = metrics["loss"]

    params = dict(
        context="evaluate", # Package behavior
        row_count=len(X_test)
    )

    save_results(model_name='Unet_evaluate',params=params, metrics=metrics, history=history)

    print(f"✅ Model evaluated, binary_crossentropy: {binary_crossentropy}")

def prediction():
    model = load_model(model_name=MODEL_NAME, MODEL_TARGET='gcs')
    datapathlist = load_predict_images(GCS=True)

    X_processed = image_preprocessing_bulk(datapathlist)
    y_pred = model.predict(X_processed)

    save_predict(y_pred, datapathlist, GCS=True)

    print("\n✅ prediction done: ", y_pred.shape, "\n")

def percent_cover_from_binary_mask(binary_mask):
    count_forest = np.count_nonzero(binary_mask==1)
    count_non_forest = np.count_nonzero(binary_mask==0)
    percentage_forest = round(count_forest/(count_forest + count_non_forest) * 100, 2)
    return percentage_forest
