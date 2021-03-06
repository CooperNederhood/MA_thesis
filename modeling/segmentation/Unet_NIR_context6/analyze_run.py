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

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../utilities')
from utilities import cnn_utils, transform_utils, test_eval, context_models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

####################################################################################
####################################################################################
MODEL_NAME = "Unet_rand_nir"

FRONT_END_TYPE = "Unet" 
PATH_TO_FRONT_END_WEIGHTS = "../Unet_rand_NIR/Unet_rand_nir" 
IS_GPU = device == "cuda:0" 
CONTEXT_LAYER_COUNT = 6
OUTPUT_CHANNELS = 1

EPOCH_COUNT = 1 
BATCH_SIZE = 4
img_size = 256
input_channels = 4

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

test_eval.do_in_sample_tests(net, "../../../", tile=True, dtype="Four_band")
test_eval.do_out_sample_tests(net, "../../../", tile=True, dtype="Four_band")



