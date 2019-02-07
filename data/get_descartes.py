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
    start_time="2000-01-01", end_time="2019-01-01", cloud_pct=1.0):
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
    #print(aoi)


    scenes, geoctx = dl.scenes.search(aoi,
        products=[product_name],
        start_datetime=start_time,
        end_datetime=end_time)

    return scenes, geoctx 

def get_non_slums(sat_type, method="median"):

    path = "./google_earth/non_slum.csv"
    df = pd.read_csv(path)

    nonslum_metadata = {'ID':[], 'Sat_type':[], 'Pic_count':[], 'Date_min':[], 'Date_max':[], 'Avg_cloud_pct':[]}
    for i in range(1, 10+1):
        cur_id = "nonslum_id{}_image".format(i)
        cur_obs = df[df.ID==i]

        top_left = [float(x) for x in cur_obs.top_left.item().split(";")]
        bot_right = [float(x) for x in cur_obs.bot_right.item().split(";")]

        scenes, geoctx = get_descartes_images(sat_type, top_left, bot_right)
        clean_metadata(cur_id, sat_type, scenes, nonslum_metadata)

        if len(scenes) > 0:
            if method == "median":
                array_stack_RGB = scenes.stack("red green blue", geoctx)
                array_stack_FULL = scenes.stack("red green blue nir", geoctx)

                # Take medians for now, and save out
                array_RGB = np.array(np.ma.median(array_stack_RGB, axis=0)).astype('uint8')
                array_FULL = np.array(np.ma.median(array_stack_FULL, axis=0))

            else:
                assert method == "min_cloud"
                # Sort default is for ascending but we want descending, as in 
                # places of overlap mask uses the LAST value
                scenes = scenes.sorted("properties.cloud_fraction", reverse=True)

                # Check that the Last scene has the lowest cloud_fraction
                if len(scenes) >= 2:
                    assert scenes[-1].properties.cloud_fraction <= scenes[0].properties.cloud_fraction

                # Mosaic the set into a single array
                array_RGB = scenes.mosaic("red green blue", geoctx)
                array_FULL = scenes.mosaic("red green blue nir", geoctx)

            # Convert the RGB array into Image
            rgb_img = transforms.ToPILImage()(torch.from_numpy(array_RGB))
            array_FULL = np.array(array_FULL)
            rgb_file = "{}.png".format(cur_id)
            full_file = "{}.npy".format(cur_id)

            # Save RGB as image and save Full 4-band as numpy array
            rgb_img.save(os.path.join(DESCARTES_PATH, "RGB", method, "not_slums", rgb_file))
            np.save(os.path.join(DESCARTES_PATH, "Four_band", method, "not_slums", full_file), array_FULL)
         
    metadata_df = pd.DataFrame(nonslum_metadata)
    metadata_df.to_csv("descartes/{}_data_review_nonslums_{}.csv".format(sat_type, method))


def get_image_for_ona(sat_type, ona_id, dataframe, output_dict, fails, metadata_dict, method="median"):
    '''
    Given the Ona ID and the dataframe containing the boundaries, 
    grabs the corresponding Google Earth image for the bounding box
    and saves based on the Ona ID
    '''

    assert method in {'min_cloud', 'median'}

    # Process the Ona list of coords from within the DataFrame
    try:
        process_ona.process_coords(ona_id, dataframe, output_dict)
    except:
        fails.append(ona_id)
        print("ERROR - check Ona coordinates for {}".format(ona_id))
        return None 

    NW_lat_long = output_dict[ona_id]['top_left']
    SE_lat_long = output_dict[ona_id]['bot_right']

    # Now query the available scenes for that AOI's boundaries
    scenes, geoctx = get_descartes_images(sat_type, NW_lat_long, SE_lat_long)

    # Now let's process the scenes we got back for MetaData
    clean_metadata(ona_id, sat_type, scenes, metadata_dict)

    if len(scenes) > 0:

        if method == "median":
            array_stack_RGB = scenes.stack("red green blue", geoctx)
            array_stack_FULL = scenes.stack("red green blue nir", geoctx)

            # Take medians for now, and save out
            array_RGB = np.array(np.ma.median(array_stack_RGB, axis=0)).astype('uint8')
            array_FULL = np.array(np.ma.median(array_stack_FULL, axis=0))

        else:
            assert method == "min_cloud"
            # Sort default is for ascending but we want descending, as in 
            # places of overlap mask uses the LAST value
            scenes = scenes.sorted("properties.cloud_fraction", reverse=True)

            # Check that the Last scene has the lowest cloud_fraction
            if len(scenes) >= 2:
                assert scenes[-1].properties.cloud_fraction <= scenes[0].properties.cloud_fraction

            # Mosaic the set into a single array
            array_RGB = scenes.mosaic("red green blue", geoctx)
            array_FULL = scenes.mosaic("red green blue nir", geoctx)

        # Convert the RGB array into Image
        rgb_img = transforms.ToPILImage()(torch.from_numpy(array_RGB))
        array_FULL = np.array(array_FULL)
        rgb_file = "ona_id{}_image.png".format(cur_ona_id)
        full_file = "ona_id{}_image.npy".format(cur_ona_id)

        # Save RGB as image and save Full 4-band as numpy array
        rgb_img.save(os.path.join(DESCARTES_PATH, "RGB", method, "slums", rgb_file))
        np.save(os.path.join(DESCARTES_PATH, "Four_band", method, "slums", full_file), array_FULL)



def clean_metadata(ona_id, sat_type, scene_collection, metadata_dict):
    '''
    Given a scene collection, gets info on date range, clouds, and count
    '''
    pic_count = 0
    avg_cloud_pct = 0.0
    min_date = datetime.date(datetime.MAXYEAR, 1, 1)
    max_date = datetime.date(datetime.MINYEAR, 1, 1)

    for scene in scene_collection:
        
        yr = scene.properties.date.year
        mo = scene.properties.date.month
        date_ym = datetime.date(yr, mo, 1)

        pic_count += 1
        avg_cloud_pct += scene.properties.cloud_fraction
        min_date = date_ym if date_ym < min_date else min_date
        max_date = date_ym if date_ym > max_date else max_date 

    if pic_count > 0:
        avg_cloud_pct /= pic_count

    # Add vals to dictionary
    metadata_dict['ID'].append(ona_id)
    metadata_dict['Sat_type'].append(sat_type)
    metadata_dict['Pic_count'].append(pic_count)
    metadata_dict['Date_min'].append(min_date)
    metadata_dict['Date_max'].append(max_date)
    metadata_dict['Avg_cloud_pct'].append(avg_cloud_pct)


    # print("Ona ID: {} | Sat type: {} | Pic count: {} | Date range: {} to {} | Avg cloud pct: {}".format(
    #     cur_ona_id, sat_type, pic_count, min_date, max_date, avg_cloud_pct))


if __name__ == "__main__":

    DESCARTES_PATH = "descartes"
    metadata_client = dl.Metadata()
    sat_type = "Pleiades"

    #method = "min_cloud"
    method = "median"

    get_non_slums(sat_type, method)

    # Read in the Ona file of settlements
    df_lagos = pd.read_csv("ona/Lagos_settlements.csv")

    # Now loop down the Oan ID's, clean, and get Google Earth image
    d = {}
    f = []
    metadata_dict = {'ID':[], 'Sat_type':[], 'Pic_count':[], 'Date_min':[], 'Date_max':[], 'Avg_cloud_pct':[]}
    for cur_ona_id in range(df_lagos.shape[0]):
        get_image_for_ona(sat_type, cur_ona_id, df_lagos, d, f, metadata_dict, method=method)
        print("Succesfully processed Ona ID: ", cur_ona_id)
     
    metadata_df = pd.DataFrame(metadata_dict)
    metadata_df.to_csv("descartes/{}_data_review_slums_{}.csv".format(sat_type, method))

