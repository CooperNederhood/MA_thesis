import pandas as pd 
import ast 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
import os 

root_google_earth = "google_earth/zoom_18/slums"
root_pleiades = "descartes/RGB/min_cloud/slums"

ge_files = os.listdir(root_google_earth)
pl_files = os.listdir(root_pleiades)

# Do Google Earth to Descartes Pleiades scaling
df_dict = {'id_num': [], 'ID': [], 'x_scale': [], 'y_scale': []}
for ge_f in ge_files:

	id_numeric = int(ge_f.replace("ona_id", "").replace("_image.png", ""))
	df_dict['ID'].append(ge_f)
	df_dict['id_num'].append(id_numeric)

	if ge_f not in pl_files:
		df_dict['x_scale'].append("NA")
		df_dict['y_scale'].append("NA")

	else:
		ge_img = Image.open(os.path.join(root_google_earth, ge_f))
		pl_img = Image.open(os.path.join(root_pleiades, ge_f))

		x_scale = pl_img.size[0] / ge_img.size[0] 
		y_scale = pl_img.size[1] / ge_img.size[1] 
		df_dict['x_scale'].append(x_scale)
		df_dict['y_scale'].append(y_scale)

df = pd.DataFrame(df_dict)
df.to_csv("Pleiades_scaling.csv")


