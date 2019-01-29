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

GOOGLE_MAPS_API_KEY = 'AIzaSyDnpPsSCMi3F8Y2xXHk8P3USo45V2Wc99I'\

# circumference/radius
tau = 6.283185307179586
# One degree in radians, i.e. in the units the machine uses to store angle,
# which is always radians. For converting to and from degrees. See code for
# usage demonstration.
DEGREE = tau/360

ZOOM_OFFSET = 8
#GOOGLE_MAPS_API_KEY = None  # set to 'your_API_key'

# Max width or height of a single image grabbed from Google.
MAXSIZE = 640
# For cutting off the logos at the bottom of each of the grabbed images.  The
# logo height in pixels is assumed to be less than this amount.
LOGO_CUTOFF = 32

def get_non_slum_image(non_slum_id, dataframe, fails):
    '''
    Given the Ona ID and the dataframe containing the boundaries, 
    grabs the corresponding Google Earth image for the bounding box
    and saves based on the Ona ID
    '''

    cur_obs = dataframe.loc[non_slum_id]
    NW_lat, NW_long = cur_obs['top_left'].split("; ")
    SE_lat, SE_long = cur_obs['bot_right'].split("; ")

    NW_lat_long =  (float(NW_lat)*DEGREE, float(NW_long)*DEGREE)
    SE_lat_long = (float(SE_lat)*DEGREE, float(SE_long)*DEGREE)

    zoom = 18   # be careful not to get too many images!

    result = get_google_earth.get_maps_image(NW_lat_long, SE_lat_long, zoom=zoom)
    result.save("google_earth/nonslum_id{}_image.png".format(non_slum_id))


if __name__ == "__main__":

    file = "non_slum.csv"

    df = pd.read_csv("google_earth/"+file, index_col="ID")

    # # Now loop down the Oan ID's, clean, and get Google Earth image
    f = []
    for non_slum_id in range(6, df.shape[0]+1):
        get_non_slum_image(non_slum_id, df, f)
