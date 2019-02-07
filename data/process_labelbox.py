import pandas as pd 
import ast 
import numpy as np 
from PIL import Image 
import cv2 
import matplotlib.pyplot as plt 
import os 


def load_labelbox_labels(labelbox_file):
    '''
    Loads the csv file output from Labelbox which contains the mask data.
    Parses it and returns dictionary with the filename as key
    '''

    path = "labelbox/"
    df = pd.read_csv(os.path.join(path, labelbox_file))

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
            
            # Get slum coords, if necessary
            if has_slum:
                slum_polygon_count = len(obs_label['slum'])
                print("Ona ID {} has {} slum polygons".format(pic_id, slum_polygon_count))

                slum_coords = []
                for i in range(slum_polygon_count):
                    slum_coords.append(obs_label['slum'][i]['geometry'])
            else:
                slum_coords = None


            # Get nonslum coords, if necessary
            if has_nonslum:
                nonslum_polygon_count = len(obs_label['non_slum'])
                print("Ona ID {} has {} non-slum polygons".format(pic_id, nonslum_polygon_count))

                non_slum_coords = []
                for i in range(nonslum_polygon_count):
                    non_slum_coords.append(obs_label['non_slum'][i]['geometry'])
            else:
                non_slum_coords = None

            rv_d[pic_id] = {'slum_coords':slum_coords, 'non_slum_coords':non_slum_coords}

    return rv_d 

def make_mask(pic_id, truth_mask_dict, x_scale=1, y_scale=1):
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
    orig_img = Image.open(os.path.join(ONA_IMAGE_PATH, pic_id))
    im_array = np.array(orig_img)

    # BUild the coords list for the slum coords
    slum_xy_list = labels[ona_id]['slum_coords']

    # Check whether this image has a mask for that type
    if slum_xy_list is not None:
        is_slum_mask = np.zeros(im_array.shape[0:2])

        for slum_xy in slum_xy_list:
            slum_coords = np.empty((1,len(slum_xy),2), dtype=np.int32)
            for i, xy_pair in enumerate(slum_xy):
                x = xy_pair['x']*x_scale
                y = xy_pair['y']*y_scale

                slum_coords[0,i,0] = x
                slum_coords[0,i,1] = y
        
            is_slum_mask = cv2.fillPoly(is_slum_mask, slum_coords, 1)

    # BUild the coords list for the non-slum coords
    nonslum_xy_list = labels[ona_id]['non_slum_coords']
    if nonslum_xy_list is not None:
        is_nonslum_mask = np.ones(im_array.shape[0:2])

        for nonslum_xy in nonslum_xy_list:
            nonslum_coords = np.empty((1,len(nonslum_xy),2), dtype=np.int32)
            for i, xy_pair in enumerate(nonslum_xy):
                x = xy_pair['x']*x_scale
                y = xy_pair['y']*y_scale

                nonslum_coords[0,i,0] = x
                nonslum_coords[0,i,1] = y
        
            is_nonslum_mask = cv2.fillPoly(is_nonslum_mask, nonslum_coords, 0)
     
    if nonslum_xy_list != None and slum_xy_list != None:
        final_mask = is_slum_mask * is_nonslum_mask 
    elif nonslum_xy_list != None and slum_xy_list == None:
        final_mask = is_nonslum_mask
    elif nonslum_xy_list == None and slum_xy_list != None:
        final_mask = is_slum_mask
    else:
        print("Check img ID ", pic_id)
        

    #return im_array, is_slum_mask, not_slum_mask, final_mask
    return final_mask
# xjwa577N5zq:tc

if __name__ == "__main__":

    f = "download_1_13_2019.csv"

    scale_df = pd.read_csv("Pleiades_scaling.csv", index_col="ID")
    labels = load_labelbox_labels(f)

    # Just adjust the ONA_IMAGE_PATH to wherever the images are located, 
    #       be they in google_earth, descartes, or elsewhere
    mask_type = "pleiades"
    ONA_IMAGE_PATH = "descartes/RGB/min_cloud/slums"

    # Loop over all ID's in the ground-truth thus far
    for ona_id in labels.keys():
        if labels[ona_id] != 'Skip':

            if scale_df is not None:
                if ona_id in scale_df.index:
                    x_scale = scale_df.loc[ona_id]['x_scale']
                    y_scale = scale_df.loc[ona_id]['y_scale']

                    if pd.isna(x_scale):
                        print("Ona ID ", ona_id, " is NA in the scale dataframe")
                        continue
                    else:
                        mask = make_mask(ona_id, labels, x_scale, y_scale)
                else:
                    print("Ona ID ", ona_id, "is not in scale dataframe")
                    continue 

            else:    
                mask = make_mask(ona_id, labels)

            mask_img = Image.fromarray(mask*255)
            mask_img = mask_img.convert('RGB')
            p = os.path.join("labelbox", mask_type, ona_id)
            print("Saving image mask for: ", ona_id)
            mask_img.save(p)
            


