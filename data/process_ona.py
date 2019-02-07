import pandas as pd 
import numpy as np 
import get_google_earth 

file = "sdi_boundaries_2019_01_08_14_21_48_287773.csv"
CITY = 'section_B/B5_City'
BOUNDS = 'section_C/C2_Boundary'
AREA = 'section_C/C3_Area_Calculate'
COMM = 'section_B/B7_Settlement_Name_Community'
MUNI = 'section_B/B7_Settlement_Name_Community'
POINT = 'section_C/C1_GPS'

columns = [CITY, AREA, COMM, MUNI, POINT, BOUNDS]

# circumference/radius
tau = 6.283185307179586
# One degree in radians, i.e. in the units the machine uses to store angle,
# which is always radians. For converting to and from degrees. See code for
# usage demonstration.
DEGREE = tau/360

ZOOM_OFFSET = 8

# Max width or height of a single image grabbed from Google.
MAXSIZE = 640
# For cutting off the logos at the bottom of each of the grabbed images.  The
# logo height in pixels is assumed to be less than this amount.
LOGO_CUTOFF = 32


def process_coords(ona_id, dataframe, output_dict):
    '''
    Processes dataframe and returns dictionary of coords
    '''

    current_obs = dataframe.loc[ona_id,:]
    point = current_obs[POINT].replace(' 0 0', '')
    bounds = current_obs[BOUNDS].replace(' 0 0', '')
    lats = []
    lons = []
    bounds_list = bounds.split(";")

    for coord_pair in bounds_list:
        lat, lon = map(float, coord_pair.split(" "))
        lats.append(lat)
        lons.append(lon)

    max_lat = np.max(lats)
    max_lon = np.max(lons)
    min_lat = np.min(lats)
    min_lon = np.min(lons)
    center = [float(x) for x in point.split(" ")]
    center = tuple(center)

    results = {'top_left': (max_lat, min_lon), 'bot_right': (min_lat, max_lon), 
        'bounds': bounds_list, 'lats': lats, 'lons': lons, 'center': center }

    output_dict[ona_id] = results

    return output_dict 

def get_image_for_ona(ona_id, dataframe, output_dict, fails):
    '''
    Given the Ona ID and the dataframe containing the boundaries, 
    grabs the corresponding Google Earth image for the bounding box
    and saves based on the Ona ID
    '''

    try:
        process_coords(ona_id, dataframe, output_dict)
    except:
        fails.append(ona_id)
        return "ERROR"

    NW_lat, NW_long = output_dict[ona_id]['top_left']
    SE_lat, SE_long = output_dict[ona_id]['bot_right']

    NW_lat_long =  (NW_lat*DEGREE, NW_long*DEGREE)
    SE_lat_long = (SE_lat*DEGREE, SE_long*DEGREE)

    zoom = 18   # be careful not to get too many images!

    result = get_google_earth.get_maps_image(NW_lat_long, SE_lat_long, zoom=zoom)
    result.save("google_earth/ona_id{}_image.png".format(ona_id))



if __name__ == "__main__":

    # Just clean the raw Ona file, keep Lagos only, save out for posterity
    df = pd.read_csv("ona/"+file)
    df = df[df[CITY] == "Lagos"]

    df_lagos = df[columns].copy()
    df_lagos.sort_values(AREA, ascending=True, inplace=True)
    df_lagos.reset_index(drop=True, inplace=True)
    df_lagos.index.name = "ID"
    df_lagos.to_csv("ona/Lagos_settlements.csv")

    # Now loop down the Oan ID's, clean, and get Google Earth image
    d = {}
    f = []
    for cur_ona_id in range(df_lagos.shape[0]):
        get_image_for_ona(cur_ona_id, df_lagos, d, f)

