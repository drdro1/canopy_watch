from canopywatch.params import BASE_DIR
from canopywatch.ml_logic.preprocessor import compute_percentage_forest, create_full_filepath, image_preprocessing_bulk
from canopywatch.ml_logic.model_linear import build_linear_model, compile_linear_model, fit_linear_model
from canopywatch.ml_logic.registry import save_results, save_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import History

def preprocess(max_num_images: int = 0):
    metadata_df = pd.read_csv(BASE_DIR + "/data/forest_segmented/meta_data.csv")
    if max_num_images > 0:
        metadata_df = metadata_df.head(100)

    masks_path = BASE_DIR + "/data/forest_segmented/masks"
    images_path = BASE_DIR + "/data/forest_segmented/images"
    file_paths_image, file_paths_mask = create_full_filepath(metadata_df, images_path, masks_path)

    df_forest = compute_percentage_forest(file_paths_mask, metadata_df, False)
    preprocessed_images = image_preprocessing_bulk(file_paths_image)
    X = np.stack(preprocessed_images, axis=0)
    y = np.array(df_forest['percentage_forest'])

    return X, y


def train(X_train: np.array, y_train: np.array, model_params: dict, patience: 15, batch_size:32, epochs: 200, validation_split=0.15, restore_best_weights = True):
    model = build_linear_model(model_params)
    compile_linear_model(model, model_params)
    history = fit_linear_model(model, X_train, y_train, patience, batch_size, epochs, validation_split, restore_best_weights)
    return history, model


def evaluate(model, X_test, y_test):
    return model.evaluate(X_test, y_test, return_dict=True)

def save(model_name: str, params: dict, metrics: dict, history: History):
    save_results(model_name, params, metrics, history)

def save_one_model():
    X, y = preprocess(2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)
    params = {'start_filters': 32, 'kernel_size': 5, 'dropout': 0.3, 'num_blocks': 4, 'optimizer': 'adam', 'batch_size': 32}
    history, model = train(X_train, y_train, params, 10, params['batch_size'], 1000, 0.15, True)
    save_model(model)

def get_XTest():
    X, y = preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_test

def industrial():
    X, y = preprocess(1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    for batch_size in [32, 64]:
        for kernel_size in [3, 5]:
            for start_filters in [8, 16]:
                for dropout in [0.2, 0.3]:
                    for optimizer in ['adam', 'RMSprop']:
                        for num_blocks in [3, 4, 5]:
                            params = {
                                "start_filters": start_filters,
                                "kernel_size": kernel_size,
                                "dropout": dropout,
                                "num_blocks": num_blocks,
                                "optimizer": optimizer,
                                "batch_size": batch_size
                            }
                            # params = {
                            #     "start_filters": 16,
                            #     "kernel_size": 3,
                            #     "twice": False,
                            #     "dropout": 0.2,
                            #     "num_blocks": 4,
                            #     "optimizer": 'adam',
                            #     "batch_size": 32
                            # }
                            print(f'iteration filters={start_filters}, kernel_size={kernel_size} dropout={dropout} num_blocks={num_blocks} optimizer={optimizer} batch_size={batch_size}')
                            history, model = train(X_train, y_train, params, 9, batch_size, 300, 0.15, True)
                            metrics = evaluate(model, X_test, y_test)
                            save_model(model)
                            save('linear', params, metrics, history)


if __name__ == '__main__':
    industrial()
    save_one_model()
