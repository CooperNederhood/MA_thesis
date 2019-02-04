import descarteslabs as dl
import process_ona 
import pandas as pd 
import numpy as np 
from PIL import Image
import os 

from torchvision import datasets, models, transforms
import torch

import datetime  

def get_descartes_metadata(pic_id, metadata_client, sat_type, NW_lat_long, SE_lat_long,
    start_time="2000-01-01", end_time="2019-01-01", cloud_pct=1.0):
    assert sat_type in {'Pleiades', 'Spot'}
    if sat_type == "Pleiades":
        product_name = "airbus:oneatlas:phr:v2"
    elif sat_type == "Spot":
        product_name = "airbus:oneatlas:spot:v2"
    else:
        print("Incorrect sat_type specified")

    aoi = {
    "type": "Polygon",
    "coordinates": [[
        [NW_lat_long[1], NW_lat_long[0]], [SE_lat_long[1], NW_lat_long[0]],
        [SE_lat_long[1], SE_lat_long[0]], [NW_lat_long[1], SE_lat_long[0]],
        [NW_lat_long[1], NW_lat_long[0]]  ]]
    }
   

    features = metadata_client.features(product_name,
        start_datetime=start_time,
        end_datetime=end_time,
        geom=aoi)

    pic_count = 0
    avg_cloud_pct = 0.0
    min_date = datetime.date(datetime.MAXYEAR, 1, 1)
    max_date = datetime.date(datetime.MINYEAR, 1, 1)

    for ft in features:
        date_str = ft['properties']['acquired']
        yr, mo, _ = date_str.split("-")
        yr = int(yr)
        mo = int(mo)
        date_ym = datetime.date(yr, mo, 1)

        pic_count += 1
        avg_cloud_pct += ft['properties']['cloud_fraction']
        min_date = date_ym if date_ym < min_date else min_date
        max_date = date_ym if date_ym > max_date else max_date 

    if pic_count > 0:
        avg_cloud_pct /= pic_count

    print("Ona ID: {} | Sat type: {} | Pic count: {} | Date range: {} to {} | Avg cloud pct: {}".format(
        cur_ona_id, sat_type, pic_count, min_date, max_date, avg_cloud_pct))

    return features

def get_descartes_images(sat_type, NW_lat_long, SE_lat_long, 
    start_time="2010-01-01", end_time="2019-01-01", cloud_pct=1.0):
    '''

    '''
    assert sat_type in {'Pleiades', 'Spot'}
    if sat_type == "Pleiades":
        product_name = "airbus:oneatlas:phr:v2"
    elif sat_type == "Spot":
        product_name = "airbus:oneatlas:spot:v2"
    else:
        print("Incorrect sat_type specified")

    aoi = {
    "type": "Polygon",
    "coordinates": [[
        [NW_lat_long[1], NW_lat_long[0]], [SE_lat_long[1], NW_lat_long[0]],
        [SE_lat_long[1], SE_lat_long[0]], [NW_lat_long[1], SE_lat_long[0]],
        [NW_lat_long[1], NW_lat_long[0]]  ]]
    }
    print(aoi)


    scenes, geoctx = dl.scenes.search(aoi,
        products=[product_name],
        start_datetime=start_time,
        end_datetime=end_time)

    return scenes, geoctx 


def get_image_for_ona(sat_type, ona_id, dataframe, output_dict, fails, method="Cloud_rank"):
    '''
    Given the Ona ID and the dataframe containing the boundaries, 
    grabs the corresponding Google Earth image for the bounding box
    and saves based on the Ona ID
    '''

    assert method in {'Cloud_rank', 'Median'}

    try:
        process_ona.process_coords(ona_id, dataframe, output_dict)
    except:
        fails.append(ona_id)
        print("ERROR - check Ona coordinates for {}".format(ona_id))
        return None, None 

    NW_lat_long = output_dict[ona_id]['top_left']
    SE_lat_long = output_dict[ona_id]['bot_right']

    scenes, geoctx = get_descartes_images(sat_type, NW_lat_long, SE_lat_long)

    if len(scenes) > 0:

        if method == "Median":
            array_stack_RGB = scenes.stack("red green blue", geoctx)
            array_stack_FULL = scenes.stack("red green blue nir", geoctx)

            # Take medians for now, and save out
            array_med_RGB = np.array(np.ma.median(array_stack_RGB, axis=0)).astype('uint8')
            array_med_FULL = np.array(np.ma.median(array_stack_FULL, axis=0))

            rgb_img = transforms.ToPILImage()(torch.from_numpy(array_med_RGB))
            rgb_file = "ona_id{}_image.png".format(cur_ona_id)
            rgb_img.save(os.path.join(DESCARTES_PATH, "RGB", "slums", rgb_file))

        else:
            # Method == "Cloud_rank"
            scenes = scenes.sorted("properties.cloud_fraction")

            # Check that the FIRST scene has the lowest cloud_fraction
            if len(scenes) >= 2:
                assert scenes[0].properties.cloud_fraction <= scenes[-1].properties.cloud_fraction

            # Take the least cloudy in the set
            array_stack_RGB = scenes[0].ndarray("red green blue", geoctx)
            array_stack_FULL = scenes[0].ndarray("red green blue nir", geoctx)

            rgb_img = transforms.ToPILImage()(torch.from_numpy(array_stack_RGB))
            rgb_file = "ona_id{}_image.png".format(cur_ona_id)
            rgb_img.save(os.path.join(DESCARTES_PATH, "RGB", "slums", rgb_file))



    else:
        print("\n\n")
        print("Image {} returns 0 scenes".format(ona_id))



if __name__ == "__main__":

    DESCARTES_PATH = "descartes"
    metadata_client = dl.Metadata()
    sat_type = "Pleiades"

    # Read in the Ona file of settlements
    df_lagos = pd.read_csv("ona/Lagos_settlements.csv")

    # Now loop down the Oan ID's, clean, and get Google Earth image
    d = {}
    f = []
    for cur_ona_id in range(df_lagos.shape[0]):
        get_image_for_ona(sat_type, cur_ona_id, df_lagos, d, f, method="Cloud_rank")
        # try:
        #     process_ona.process_coords(cur_ona_id, df_lagos, d)
        # except:
        #     f.append(cur_ona_id)
        #     print("ERROR")
        #     continue

        # NW_lat_long = d[cur_ona_id]['top_left']
        # SE_lat_long = d[cur_ona_id]['bot_right']        
        # features = get_descartes_metadata(cur_ona_id,metadata_client, sat_type, NW_lat_long, 
        #   SE_lat_long, start_time="2010-01-01", end_time="2019-01-01", cloud_pct=1.0)
        


#get_image_for_ona(sat_type, ona_id, dataframe, output_dict, fails, method="Cloud_rank")