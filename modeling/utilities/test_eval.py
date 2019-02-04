import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torchvision import datasets, models, transforms
from torch import utils, optim 
import os 
import time 
import copy 
import pandas as pd 
import numpy as np 

from PIL import Image 
from collections import OrderedDict
import matplotlib.pyplot as plt 
import json 
import datetime 

# Import other module
import sys 
sys.path.append('../')
#from utilities import cnn_utils, transform_utils

def load_weights(raw_net, model_name, is_gpu):
    '''
    The raw_net is a type of deep neural net and 
    we fetch the state_dict contained in the appropriate
    subfolder and update the weights of raw_net accordingly

    Returns: raw_net with the trained_weights
    '''

    location = 'cpu' if not is_gpu else 'gpu'
    raw_net.load_state_dict(torch.load(os.path.join(model_name, 
        "model.pt"), map_location=location))

    return raw_net 


def plot_training_dict(model_name):

    json_data = open(os.path.join(model_name, "training_hist.json")).read()
    training_dict = json.loads(json_data)

    epoch_list = range(len(training_dict['train']['loss']))

    plt.plot(epoch_list, training_dict['train']['loss'], label='train')
    plt.plot(epoch_list, training_dict['val']['loss'], label='val')
    plt.legend()
    plt.savefig('Loss.png')

    plt.clf()
    plt.plot(epoch_list, training_dict['train']['acc'], label='train')
    plt.plot(epoch_list, training_dict['val']['acc'], label='val')
    plt.legend()
    plt.savefig('Acc.png')
        

def make_pred_map_classification(test_img, net, pic_size, step_size=None):
    '''
    Inputs:
        test_img: (np array) of image we're evaluating on
        net: (pytorch network)
        pic_size: (int) dimensions of thumnails to extract and pass into net
        step_size: (int) step size between pics
    '''

    if step_size is None:
        step_size = pic_size

    # Output prediction map starts with array of zeros
    pred_map = np.zeros(test_img.shape[1:3])
    pred_map_cat = np.zeros(test_img.shape[1:3])
    print("Prediction map shape: ", pred_map.shape)

    i = pic_size
    j = pic_size

    _, max_i, max_j = test_img.shape
    pic_count = 0

    # Putting model in eval mode sets layers like dropout to prediction
    # Wrapping in no_grad() lets us not record gradient
    net.eval()
    with torch.no_grad():
        while i < max_i:
            while j < max_j:
                sub_pic = test_img[:, i-pic_size:i, j-pic_size:j] 
                sub_pic = sub_pic.unsqueeze(0)
                
                pred = net(sub_pic).item()
                assert pred < 1 and pred > 0
                pred_cat = np.round(pred).item()
                assert pred_cat in {0,1}

                #print("Assigning pred val = {} to cat = {}".format(pred, pred_cat))

                # Update the prediction matrix accordingly
                sub_mat_cat = np.full((pic_size,pic_size), pred_cat)
                sub_mat = np.full((pic_size, pic_size), pred)

                pred_map_cat[i-pic_size:i, j-pic_size:j] += sub_mat_cat
                pred_map[i-pic_size:i, j-pic_size:j] += sub_mat

                pic_count += 1
                j += step_size
            j = pic_size
            i += step_size

    return pred_map_cat, pred_map, pic_count

def make_pred_map_segmentation(test_img, net, pic_size, step_size=None):
    '''
    Inputs:
        test_img: (np array) of image we're evaluating on
        net: (pytorch network)
        pic_size: (int) dimensions of thumnails to extract and pass into net
        step_size: (int) step size between pics
    '''

    if step_size is None:
        step_size = pic_size

    # Output prediction map starts with array of zeros
    pred_map = torch.zeros(test_img.shape[1:3])
    pred_map_cat = torch.zeros(test_img.shape[1:3])
    print("Prediction map shape: ", pred_map.shape)

    i = pic_size
    j = pic_size

    _, max_i, max_j = test_img.shape
    pic_count = 0

    # Putting model in eval mode sets layers like dropout to prediction
    # Wrapping in no_grad() lets us not record gradient
    net.eval()
    with torch.no_grad():
        while i < max_i:
            while j < max_j:
                sub_pic = test_img[:, i-pic_size:i, j-pic_size:j] 
                sub_pic = sub_pic.unsqueeze(0)
                
                # Should get 4D mask back, squeeze the batch and channel dimensions
                pred_mask = net(sub_pic)
                assert pred_mask.dim() == 4

                assert pred_mask.min().item() > 0 and pred_mask.max().item() < 1
                pred_mask = pred_mask.squeeze(dim=0).squeeze(dim=0)
                assert pred_mask.dim() == 2

                # Make classification mask
                pred_mask_cat = torch.round(pred_mask)
                assert pred_mask_cat.shape == pred_mask.shape 

                # Update the prediction matrix accordingly
                pred_map_cat[i-pic_size:i, j-pic_size:j] += pred_mask_cat
                pred_map[i-pic_size:i, j-pic_size:j] += pred_mask

                pic_count += 1
                j += step_size
            j = pic_size
            i += step_size

    pred_img = Image.fromarray(pred_map.numpy()*255).convert("RGB")
    return pred_img, pic_count


if __name__ == "__main__":
    #########################################################################################
    # MODEL SPECIFIC:
    from modeling.segmentation import base_segmentation_model
    #from modeling.classification import base_classification_model

    CHANNELS = 3 
    img_size = 128
    model_name = "seg_base"
    #model_name = "class_base"
    #########################################################################################
    THESIS_ROOT = "../../"
    DATA_ROOT = os.path.join(THESIS_ROOT, "data")

    IN_SAMPLE_ROOT = "test_images/in_sample"
    OUT_SAMPLE_ROOT = os.path.join(DATA_DIR, "aoi")

    # Initialize model and the update to trained weights
    net = base_segmentation_model.segNet(img_size)
    net = load_weights(net, model_name, is_gpu=False)

    aoi_files = os.listdir(OUT_SAMPLE_ROOT)

    # for aoi in aoi_files:
    #     img = Image.open(os.path.join(OUT_SAMPLE_ROOT, aoi))
    #     img_array = transforms.ToTensor()(img)

    #     pred_map_cat, pred_map, pic_count = make_pred_map(img_array, net, 128, 128)

    #     pred_img = Image.fromarray(pred_map*255).convert('RGB')
    #     save_out = os.path.join("aoi_test", aoi)

    #     pred_img.save(save_out)

    #     print("Predicted file {} tiled in {} subimages".format(aoi, pic_count))

    aoi = '1_4.png'
    img = Image.open(os.path.join(OUT_SAMPLE_ROOT, aoi))
    img_array = transforms.ToTensor()(img)

    pred_map_cat, pred_map, pic_count = make_pred_map(img_array, net, 128, 128)

    pred_img = Image.fromarray(pred_map*255).convert('RGB')
    save_out = os.path.join("aoi_test", aoi)

    pred_img.save(save_out)

    print("Predicted file {} tiled in {} subimages".format(aoi, pic_count))



    # #f = "ona_id54_image.png"
    # f = "ona_id50_image.png"
    # img = Image.open(os.path.join(IN_SAMPLE_ROOT, f))
    # img_array = transforms.ToTensor()(img)

    # pred_map_cat, pred_map, pic_count = make_pred_map(img_array, net, 128, 128)

    # pred_img = Image.fromarray(pred_map*255).convert('RGB')
    # pred_img.show()
    # img.show()

