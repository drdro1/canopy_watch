import os
import time
import pickle
from PIL import Image
from pathlib import Path

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

from canopywatch.params import *
from tensorflow.keras.callbacks import History

def save_results(model_name: str, params: dict, metrics: dict, history: History) -> None:  # this function can be modified to add a part to save results on GCS
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/history/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        with open(os.path.join(LOCAL_REGISTRY_PATH, "params", model_name + "_" + timestamp + ".pickle"), "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        with open(os.path.join(LOCAL_REGISTRY_PATH, "metrics", model_name + "_" + timestamp + ".pickle"), "wb") as file:
            pickle.dump(metrics, file)

    # Save history locally
    if history is not None:
        with open(os.path.join(LOCAL_REGISTRY_PATH, "history", model_name + "_" + timestamp + ".pickle"), "wb") as file:
            pickle.dump(history.history, file)

    print("✅ Results saved locally")



def save_model(model: keras.Model = None, GCS=False) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")
    if GCS==True:
        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"training_outputs/models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")
    else:
        print("❌ Model NOT saved to GCS")

    return None


def load_model(model_name, MODEL_TARGET="local") -> keras.Model:
    """
    Return a saved model:
    - locally
    - or from GCS if MODEL_TARGET=='gcs'

    Return None (but do not Raise) if no model is found
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the wanted model version name
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", model_name)

        if not local_model_directory:
            print("❌ No model loaded/found from local disk")
            return None

        print(Fore.BLUE + f"\nLoad chosen model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(local_model_directory)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        model_path_in_bucket = f"training_outputs/models/{model_name}"
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        print(Fore.BLUE + f"\nLoading model '{model_name}' from GCS path: {model_path_in_bucket}..." + Style.RESET_ALL)

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(model_path_in_bucket)
        try:
            # Define the local path where the model will be downloaded
            local_model_path = os.path.join(local_model_directory, model_name)
            Path(local_model_path).parent.mkdir(parents=True, exist_ok=True)

            # Download the model from GCS to the local path
            blob.download_to_filename(local_model_path)

            # Load the model using TensorFlow/Keras
            loaded_model = keras.models.load_model(local_model_path)

            print(f"✅ Model successfully downloaded and loaded from: {local_model_path}")

            return loaded_model
        except Exception as e:
            print(Fore.RED + f"\n❌ Failed to load model '{model_name}' from GCS bucket '{BUCKET_NAME}'" + Style.RESET_ALL)
            print(f"Error: {e}")
            return None

def local_load_metrics(metric_name: str):
    '''
    metric_name: should only include the name WITHOUT the .pickle part
    output is a dictionary
    '''
    with open(LOCAL_REGISTRY_PATH + '/metrics/' + metric_name + '.pickle', 'rb') as file:
        metrics_out = pickle.load(file)
    return metrics_out


def local_load_params(param_name: str):
    '''
    param_name: should only include the name WITHOUT the .pickle part
    output is a dictionary
    '''
    with open(LOCAL_REGISTRY_PATH + '/params/' + param_name + '.pickle', 'rb') as file:
        params_out = pickle.load(file)
    return params_out

def load_predict_images(GCS=False):

    if GCS == False:
        print(Fore.BLUE + f"\nLoad latest images from local registry..." + Style.RESET_ALL)

        datapathlist = []

        for image in os.listdir(DATA_PATH_PREDICT):
            if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png"): #CHANGE HERE IF YOU WANT TO HAD ANOTHER TYPE OF IMAGE
                    datapathlist.append(DATA_PATH_PREDICT + "/" + image)

        print("✅ Images loaded from local disk")

        return datapathlist

    elif GCS == True:
        datapathlistbucket = "data_predict/images/"  # GCS folder path
        local_predict_directory = DATA_PATH_PREDICT
        print(Fore.BLUE + f"\nLoading images from GCS path: {datapathlistbucket}..." + Style.RESET_ALL)

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        # List all blobs (files) in the specified GCS folder
        blobs = bucket.list_blobs(prefix=datapathlistbucket)  # Prefix ensures we only get files in the folder

        downloaded_images = []

        try:
            for blob in blobs:
                # Skip directories
                if not blob.name.endswith(('.jpg', '.jpeg', '.png')):
                    continue

                # Extract the file name
                file_name = os.path.basename(blob.name)

                # Define local file path
                local_file_path = os.path.join(local_predict_directory, file_name)

                # Download each image to the local directory
                blob.download_to_filename(local_file_path)
                downloaded_images.append(local_file_path)

            print(f"\n✅ Successfully downloaded {len(downloaded_images)} images to: {local_predict_directory}")
            return downloaded_images
        except Exception as e:
            print(Fore.RED + f"\n❌ Failed to load images from GCS bucket '{BUCKET_NAME}'" + Style.RESET_ALL)
            print(f"Error: {e}")
            return None

def save_predict(y_pred = None, datapathlist = None, GCS=False):

    # Save predicted mask locally
    i = 0
    for mask in y_pred:
        pred_path = os.path.join(DATA_PATH_RESULT, f"{os.path.basename(datapathlist[i])}_mask{i}.jpg")
        pred_path_binary = os.path.join(DATA_PATH_RESULT, f"{os.path.basename(datapathlist[i])}_binary_mask{i}.jpg")
        binary_predictions = (mask > FOREST_ON_IMAGE).astype("uint8")

        mask = 255*mask.reshape(256,256)
        mask_img = Image.fromarray(mask).convert('L')
        mask_img.save(pred_path)

        binary_predictions = 255*binary_predictions.reshape(256,256)
        binary_mask_img = Image.fromarray(binary_predictions).convert('L')
        binary_mask_img.save(pred_path_binary)
        i+=1

        if GCS==True:
            model_filename = pred_path.split("/")[-1] # e.g. "855_sat_01.jpg_binary_mask0.jpg" for instance
            model_filename_binary = pred_path_binary.split("/")[-1]
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)

            blob = bucket.blob(f"data_predict/masks/{model_filename}")
            blob.upload_from_filename(pred_path)

            blob = bucket.blob(f"data_predict/masks/{model_filename_binary}")
            blob.upload_from_filename(pred_path_binary)

        else:
            print("❌ Prediction NOT saved to GCS")

    print("✅ Predictions (Masks) saved to GCS and locally")

    return None


def upload_files_to_cloud_storage(subdir: str):
    """
    subdir should be for example 'images' or 'masks'
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    root_path = os.path.join(DATA_PATH, subdir)

    i = 0
    for image in os.listdir(root_path):
        if image.endswith(".jpg"):
            blob = bucket.blob(subdir + "/" + image)
            blob.upload_from_filename(os.path.join(root_path, image))
            print(f"Uploaded {image} to {BUCKET_NAME} imgNum={i}")
            i += 1
