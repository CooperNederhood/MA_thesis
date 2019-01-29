import pandas as pd 
import ast 
import numpy as np 
from PIL import Image 
import cv2 
import matplotlib.pyplot as plt 


def load_labelbox_labels(labelbox_file):
    '''
    Loads the csv file output from Labelbox which contains the mask data.
    Parses it and returns dictionary with the filename as key
    '''

    path = "labelbox/"
    df = pd.read_csv(path+labelbox_file)

    rv_d = {}
    orig_pic_ids = df['External ID'].values

    for pic_id in orig_pic_ids:
        obs = df[df['External ID']==pic_id]

        if obs['Label'].item() == "Skip":
            rv_d[pic_id] = "Skip"

        else:
            obs_label = ast.literal_eval(obs['Label'].item())

            has_slum = 'slum' in obs_label.keys()
            has_nonslum = 'non_slum' in obs_label.keys()
            
            slum_coords = obs_label['slum'][0]['geometry'] if has_slum else None 
            non_slum_coords = obs_label['non_slum'][0]['geometry'] if has_nonslum else None 

            rv_d[pic_id] = {'slum_coords':slum_coords, 'non_slum_coords':non_slum_coords}

    return rv_d 

def make_mask(pic_id, truth_mask_dict):
    '''
    If you specify a target ID and a truth_mask_dict contianing
    the coords, loads the original iamge and builds the corresponding
    slum/not-slum mask

    Inputs:
        pic_id: (string) of Ona id ex. 'ona_id74_image.png'
        truth_mask_dict: (dict) of the ground truth labels
    Returns:
        final_mask: (np array) of binary ground truth mask. 1 denotes slum
    '''

    # Load the Ona Image
    orig_img = Image.open(ONA_IMAGE_PATH+pic_id)
    im_array = np.array(orig_img)

    # BUild the coords list for the slum coords
    slum_xy_list = labels[ona_id]['slum_coords']

    # Check whether this image has a mask for that type
    if slum_xy_list is not None:
        slum_coords = np.empty((1,len(slum_xy_list),2), dtype=np.int32)
        for i, xy_pair in enumerate(slum_xy_list):
            x = xy_pair['x']
            y = xy_pair['y']

            slum_coords[0,i,0] = x
            slum_coords[0,i,1] = y
        is_slum_mask = np.zeros(im_array.shape[0:2])
        is_slum_mask = cv2.fillPoly(is_slum_mask, slum_coords, 1)
        

    # BUild the coords list for the non-slum coords
    nonslum_xy_list = labels[ona_id]['non_slum_coords']
    if nonslum_xy_list is not None:
        nonslum_coords = np.empty((1,len(nonslum_xy_list),2), dtype=np.int32)
        for i, xy_pair in enumerate(nonslum_xy_list):
            x = xy_pair['x']
            y = xy_pair['y']

            nonslum_coords[0,i,0] = x
            nonslum_coords[0,i,1] = y
        not_slum_mask = cv2.fillPoly(np.ones(im_array.shape[0:2]), nonslum_coords, 0)

    if nonslum_xy_list != None and slum_xy_list != None:
        final_mask = is_slum_mask * not_slum_mask 
    elif nonslum_xy_list != None:
        final_mask = not_slum_mask
    elif slum_xy_list != None:
        final_mask = is_slum_mask
    else:
        print("Check img ID ", pic_id)
        

    #return im_array, is_slum_mask, not_slum_mask, final_mask
    return final_mask


f = "download_1_13_2019.csv"
labels = load_labelbox_labels(f)

ONA_IMAGE_PATH = "google_earth/zoom_18/slums/"
ona_id = 'ona_id74_image.png'

# Loop over all ID's in the ground-truth thus far
for ona_id in labels.keys():
    if labels[ona_id] != 'Skip':
        mask = make_mask(ona_id, labels)

        mask_img = Image.fromarray(mask*255)
        mask_img = mask_img.convert('RGB')
        p = "labelbox/masks/"+ona_id
        print("Saving image mask for: ", ona_id)
        mask_img.save("labelbox/masks/"+ona_id)
        

# s = labels[ona_id]['slum_coords']
# coords = np.empty((1,len(s),2), dtype=np.int32)
# for i, xy_pair in enumerate(s):
#     print(xy_pair)
#     x = xy_pair['x']
#     y = xy_pair['y']

#     coords[0,i,0] = x
#     coords[0,i,1] = y 
