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
CONTEXT_LAYER_COUNT = 4
OUTPUT_CHANNELS = 1

EPOCH_COUNT = 16
BATCH_SIZE = 4
img_size = 256
input_channels = 4

THESIS_ROOT = "../../../"
#OUT_SAMPLE_ROOT = os.path.join(THESIS_ROOT, "data", "google_earth", "aoi")
IN_SAMPLE_ROOT = os.path.join(THESIS_ROOT, "data", "descartes", "RGB", "min_cloud")
####################################################################################
####################################################################################

# Load the model weights
net = model_def.Unet(input_channels, img_size)
net = net.to(device)
net = test_eval.load_weights(net, MODEL_NAME, is_gpu=device=="cuda:0")

# Build the val dataloader

# Root directory for dataset
if input_channels == 3:
    root = '../../../data/training_data/descartes/RGB'
else:
    root = '../../../data/training_data/descartes/Four_band'

data_root = os.path.join(root, "segmentation/size_{}".format(img_size))

val_root = os.path.join(data_root, "val")

val_dset = model_def.SegmentationDataset(val_root, f_type="Numpy_array")
val_dset_loader = utils.data.DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=True)

thresholds = [0.0, 0.25, 0.45, 0.5, 0.55, 0.75, 1.0]
#thresholds = [0.25, 0.5]

test_eval.do_ROC_curve(net, val_dset_loader, thresholds, device)


