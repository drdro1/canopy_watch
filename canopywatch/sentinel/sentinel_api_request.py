import pandas as pd
from sentinelhub import SHConfig, BBox, bbox_to_dimensions, SentinelHubRequest, DataCollection, CRS, MimeType
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import cv2
from PIL import Image
from canopywatch.ml_logic.preprocessor import mask_preprocessing



def get_date_range(date_start):
    end_date = date_start + timedelta(days=10)
    return (str(date_start.date()), str(end_date.date()))


def request_sentinel_api(date_start : tuple, date_end : tuple, localisation:list)-> tuple:
    CLIENT_ID = '97202d5b-007e-48a1-8669-7e3e18480ceb'
    CLIENT_SECRET = 'JnWlVNGuv0A1Z1PYV9JzcGS4fDilsRDv'

    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

    evalscript_test = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B04", "B03", "B02"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [3.5 * sample.B04, 3.5 * sample.B03, 3.5 * sample.B02];
    }
    """


    zone_gps = localisation

    start_date = get_date_range(date_start)
    end_date = get_date_range(date_end)
    bbox = BBox(bbox=zone_gps, crs=CRS.WGS84)
    target_size = (256, 256)


    # Définir la requête
    request_start= SentinelHubRequest(
        evalscript = evalscript_test,
        data_folder="satellite_images",
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=start_date,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=target_size,
        config=config,
    )

    request_end = SentinelHubRequest(
        evalscript = evalscript_test,
        data_folder="satellite_images",
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=end_date,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=target_size,
        config=config,
    )

    image_forest_start = request_start.get_data()
    image_forest_start = np.asarray(image_forest_start)[0]

    image_forest_end = request_end.get_data()
    image_forest_end = np.asarray(image_forest_end)[0]

    return image_forest_start,image_forest_end


def load_sentinel_images(result:tuple,image_number:int,location:str):
    im_start = Image.fromarray(result[0])
    im_start.save(f"../data_predict/images/{location}_image_start_{image_number}.jpeg")
    im_end = Image.fromarray(result[1])
    im_end.save(f"../data_predict/images/{location}_image_end_{image_number}_.jpeg")


def final_results(filepath_mask_start:str, filepath_mask_end:str):
    mask_start = mask_preprocessing(filepath_mask_start)
    mask_end = mask_preprocessing(filepath_mask_end)

    count_forest_start = np.count_nonzero(mask_start==1)
    count_non_forest_start = np.count_nonzero(mask_start==0)
    percentage_forest_start = round(count_forest_start/(count_forest_start+count_non_forest_start) * 100,2)

    count_forest_end = np.count_nonzero(mask_end==1)
    count_non_forest_end = np.count_nonzero(mask_end==0)
    percentage_forest_end = round(count_forest_end/(count_forest_end+count_non_forest_end) * 100,2)

    percentage_evolution = round((count_forest_end - count_forest_start)/count_forest_start*100,2)

    print(f'There was {percentage_forest_start}% of forest in 2018 in the specified zone, {percentage_forest_end}% in 2022, hence a deforestation of {percentage_evolution}% ')
    return percentage_forest_start,percentage_forest_end,percentage_evolution
