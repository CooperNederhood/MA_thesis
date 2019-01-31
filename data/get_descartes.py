import descarteslabs as dl
import process_ona 
import pandas as pd 

def get_descartes_images(sat_type, NW_lat_long, SE_lat_long, 
    start_time="2010-01-01", end_time="2019-01-01", cloud_pct=0.0):
    '''

    '''
    assert sat_type in {'Pleiades', 'Spot'}
    if sat_type == "Pleiades":
        product_name = "airbus:oneatlas:phr:v2"
    elif sat_type == "Spot":
        product_name = "airbus:oneatlas:spot:v2"
    else:
        print("Incorrect sat_type specified")

    #product_name = "landsat:LC08:PRE:TOAR"

    # print("Original coords")
    # print(NW_lat_long)
    # print(SE_lat_long)

    # print("\nManual coords")
    # NW_lat_long = [6.9198897, 2.95470709999995]
    # SE_lat_long = [6.3989958, 3.57934279999995]

    # aoi = {
    # "type": "Polygon",
    # "coordinates": [[
    #     [NW_lat_long[0], NW_lat_long[1]], [SE_lat_long[0], NW_lat_long[1]],
    #     [SE_lat_long[0], SE_lat_long[1]], [NW_lat_long[0], SE_lat_long[1]],
    #     [NW_lat_long[0], NW_lat_long[1]]  ]]
    # }
    # print(aoi)
    aoi = {
         'type': 'Polygon',
         'coordinates': ((
             (-93.52300099792355, 41.241436141055345),
             (-93.7138666, 40.703737),
             (-94.37053769704536, 40.83098709945576),
             (-94.2036617, 41.3717716),
             (-93.52300099792355, 41.241436141055345)
         ),)
     }
    print(aoi)

    scenes, geoctx = dl.scenes.search(aoi,
        products=[product_name],
        start_datetime=start_time,
        end_datetime=end_time,
        limit=10)

    return scenes, geoctx 


def get_image_for_ona(sat_type, ona_id, dataframe, output_dict, fails):
    '''
    Given the Ona ID and the dataframe containing the boundaries, 
    grabs the corresponding Google Earth image for the bounding box
    and saves based on the Ona ID
    '''

    try:
        process_ona.process_coords(ona_id, dataframe, output_dict)
    except:
        fails.append(ona_id)
        print("ERROR")
        return None 

    NW_lat_long = output_dict[ona_id]['top_left']
    SE_lat_long = output_dict[ona_id]['bot_right']

    scenes, geoctx = get_descartes_images(sat_type, NW_lat_long, SE_lat_long)
    return scenes, geoctx
    #result.save("google_earth/ona_id{}_image.png".format(ona_id))



if __name__ == "__main__":

    # Read in the Ona file of settlements
    df_lagos = pd.read_csv("ona/Lagos_settlements.csv")

    # Now loop down the Oan ID's, clean, and get Google Earth image
    d = {}
    f = []
    # for cur_ona_id in range(df_lagos.shape[0]):
    #     scenes, geoctx = get_image_for_ona('Spot', cur_ona_id, df_lagos, d, f)
    #     print(scenes)
    #     print(geoctx)        
    scenes, geoctx = get_image_for_ona('Spot', 33, df_lagos, d, f)
    print(scenes)
    print(geoctx)