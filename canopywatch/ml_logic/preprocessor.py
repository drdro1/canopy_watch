import os
import pandas as pd
import numpy as np
from PIL import Image

from canopywatch.params import DATA_PATH

def create_full_filepath(data : pd.DataFrame, images_path : str, masks_path : str) -> list :
    '''
    Take the metadata csv as input as well as the images and masks folder path
    Complete the metadata with complete filepath (folder path + file name)
    return a list of file paths for images and masks
    '''
    data['image'] = images_path + '/' + data['image']
    data['mask'] = masks_path + '/' + data['mask']
    file_paths_image = list(data['image'])
    file_paths_mask = list(data['mask'])
    return file_paths_image, file_paths_mask

def image_preprocessing(image_filepath : str) -> np.ndarray :
    """
    Creates an image with a filepath
    Transforms it to an array of shape (256,256,3)
    Normalize the array between 0 and 1
    Return the normalized image as np array
    """
    image = Image.open(image_filepath)
    image = np.asarray(image)
    image = image/255

    return image

def mask_preprocessing(mask_filepath : str) -> np.ndarray :
    """
    Creates a mask with a filepath
    Transforms it to an array of shape (256,256,1)
    Normalize the array between 0 and 1
    Return the normalized mask as np array
    """
    mask = Image.open(mask_filepath)
    mask= mask.convert('L')
    mask = np.asarray(mask)
    mask = mask/255
    mask = np.expand_dims(mask, axis =2)
    return mask


def image_preprocessing_bulk(images_filepath : list) -> list :
    """
    Creates an image with a filepath
    Transforms it to an array of shape (256,256,3)
    Normalize the array between 0 and 1
    Return a list of images normalized as np.arrays
    """
    images = []

    for i in range(0, len(images_filepath)):
        image_preprocessed = image_preprocessing(images_filepath[i])
        images.append(image_preprocessed)

    images = np.stack(images)

    return images

def mask_preprocessing_bulk(masks_filepath : list) -> list :
    """
    Creates a mask with a filepath
    Transforms it to an array of shape (256,256,1)
    Normalize the array between 0 and 1
    Return a list of images normalized as np.arrays
    """
    masks = []

    for i in range(0,len(masks_filepath)):
        mask_preprocessed = mask_preprocessing(masks_filepath[i])
        masks.append(mask_preprocessed)

    masks = np.stack(masks)

    return masks

def compute_percentage_forest(mask_filepath : list, data : pd.DataFrame, as_percent: bool = False) -> pd.DataFrame:
    """
    Count the number of pixels with value 1 ie the white ones ie the forest in mask
    Count the number of pixels with value 0 ie the black ones ie the non forest in mask
    Compute the number of white pixels over the total number of pixels
    Return dataframe with the % of forest on the mask in data
    """
    percentages = []

    for i in range(0,len(data)):
        mask = mask_preprocessing(mask_filepath[i])
        count_forest = np.count_nonzero(mask==1)
        count_non_forest = np.count_nonzero(mask==0)
        percentage_forest = count_forest/(count_forest+count_non_forest)
        if as_percent:
            percentage_forest = percentage_forest * 100
        percentages.append(percentage_forest)

    data['percentage_forest'] = percentages
    return data

def create_metadata_forest_file():
    """
    Uses compute_percentage_forest to create dataframe with forest percentange
    and writes result to file
    """
    metadata_path = os.path.join(DATA_PATH, 'meta_data.csv')
    data = pd.read_csv(metadata_path)
    data = compute_percentage_forest(data=data)
    data.to_csv(path_or_buf=os.path.join(DATA_PATH, 'meta_data_forest.csv'))


def read_metadata_forest_file_to_df() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_PATH, 'meta_data_forest.csv'))
