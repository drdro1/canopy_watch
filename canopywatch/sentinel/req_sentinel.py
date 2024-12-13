import pandas as pd
from sentinelhub import SHConfig, BBox, bbox_to_dimensions, SentinelHubRequest, DataCollection, CRS, MimeType
import matplotlib.pyplot as plt
import numpy as np
from params import SENTINEL_CLIENT_ID, SENTINEL_CLIENT_SECRET

def first_req():
    config = SHConfig()
    config.sh_client_id = SENTINEL_CLIENT_ID
    config.sh_client_secret = SENTINEL_CLIENT_SECRET
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
        return [sample.B04, sample.B03, sample.B02];
    }
    """
    #Zone_GPS = [98.20, 2.95, 98.60, 3.35]
    Zone_GPS = [10.80, 5.38, 11, 5.50]#via le CSV
    resolution = 10 # résolution en mètres
    time_interval = ('2018-02-01','2018-02-10')
    bbox = BBox(bbox=Zone_GPS, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    print(f"Dimensions de l'image: {size} pixels")

    # Définir la requête
    request = SentinelHubRequest(
        evalscript = evalscript_test,
        data_folder="satellite_images",
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    # response = request.save_data() ###ne fonctionne pas ! all pixel à 0
    image_forest = request.get_data()
    print("Images téléchargées :", plt.imshow(image_forest[0]))
    save_test = np.save("first_image_forest", image_forest[0])

    return image_forest, save_test
