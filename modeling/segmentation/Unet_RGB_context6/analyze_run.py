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
import sys
import json 

sys.path.append('../../')
sys.path.append('../../utilities')
from utilities import cnn_utils, transform_utils, test_eval, context_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

####################################################################################
####################################################################################
MODEL_NAME = "Unet_rgb_context4"

FRONT_END_TYPE = "Unet" 
PATH_TO_FRONT_END_WEIGHTS = "../Unet_rand_RGB/Unet_rand_rgb" 
IS_GPU = device == "cuda:0" 
CONTEXT_LAYER_COUNT = 4
OUTPUT_CHANNELS = 1

EPOCH_COUNT = 1 
BATCH_SIZE = 4
img_size = 256
input_channels = 3

THESIS_ROOT = "../../../"
#OUT_SAMPLE_ROOT = os.path.join(THESIS_ROOT, "data", "google_earth", "aoi")
IN_SAMPLE_ROOT = os.path.join(THESIS_ROOT, "data", "descartes", "RGB", "min_cloud")
####################################################################################
####################################################################################

# Plot the training history
test_eval.plot_training_dict(MODEL_NAME, "batch")
test_eval.plot_training_dict(MODEL_NAME, "epoch")

# Load the model weights
net = context_models.FrontEnd_ContextModel(FRONT_END_TYPE, PATH_TO_FRONT_END_WEIGHTS, IS_GPU, 
        input_channels, img_size, CONTEXT_LAYER_COUNT, OUTPUT_CHANNELS)
net = test_eval.load_weights(net, MODEL_NAME, is_gpu=device=="cuda:0")

test_eval.do_in_sample_tests(net, "../../../", tile=True)

# # Do in-sample evaluations on slum and not_slums 
# if not os.path.isdir("in_sample_test"):
#     os.mkdir("in_sample_test")
# for t in ["slums", "not_slums"]:
#     if not os.path.isdir(os.path.join("in_sample_test", t)):
#         os.mkdir(os.path.join("in_sample_test", t))

#     for s in ["pct", "binary"]:
#     	if not os.path.isdir(os.path.join("in_sample_test", t, s)):
#     		os.mkdir(os.path.join("in_sample_test", t, s))
#     #if not os.path.isdir(os.path.join("in_sample_test", t, ""))

#     files = os.listdir(os.path.join(IN_SAMPLE_ROOT, t))
#     for f in files:
#         img = Image.open(os.path.join(IN_SAMPLE_ROOT, t, f))

#         array = transforms.ToTensor()(img)
#         pred_img, pred_cat = test_eval.make_pred_map_segmentation(array, net, img_size, img_size)   
#         pred_img.save(os.path.join("in_sample_test", t, s, f))





