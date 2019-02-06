'''
Just a simple script to pull the ENTIRE AOI for Lagos
'''

import pandas as pd 
import numpy as np 
import get_google_earth 
import api_keys 

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
#GOOGLE_MAPS_API_KEY = None  # set to 'your_API_key'

# Max width or height of a single image grabbed from Google.
MAXSIZE = 640
# For cutting off the logos at the bottom of each of the grabbed images.  The
# logo height in pixels is assumed to be less than this amount.
LOGO_CUTOFF = 32

# These coords represent a large area but not all of Lagos
NW_lat, NW_long = (6.515149, 3.325531)
SE_lat, SE_long = (6.419642, 3.425078)


NW_lat_long =  (NW_lat*DEGREE, NW_long*DEGREE)
SE_lat_long = (SE_lat*DEGREE, SE_long*DEGREE)

zoom = 18   # be careful not to get too many images!

result = get_google_earth.get_maps_image(NW_lat_long, SE_lat_long, zoom=zoom)
result.save("google_earth/total_AOI.png")
