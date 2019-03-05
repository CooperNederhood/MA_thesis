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
from utilities import cnn_utils, transform_utils, test_eval
MODEL_ZOO_PATH = "../../../model_zoo"

class FrontEnd(nn.Module):

    def __init__(self, input_channels, img_size, padding=0, classify=True):
        super(FrontEnd, self).__init__()

        # Define layers for encoding
        self.input_channels = input_channels
        self.my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.padding = padding 
        self.classify = classify
        self.trainable = None 

        # Front-end model based on VGG-16

        # Padding layer
        #self.padding_layer = nn.ReflectionPad2d(self.padding)

        # CONV BLOCK (1):
        self.conv1a = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # CONV BLOCK (2):
        # max pool
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # CONV BLOCK (3):
        # max pool
        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3c = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        # CONV BLOCK (4):
        # max pool -- REPLACED WITH DILATIONS
        self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, dilation=2, padding=2)
        self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2)
        self.conv4c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2)

        # CONV BLOCK (5):
        # max pool -- REPLACED WITH DILATIONS
        self.conv5a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=4, padding=4)
        self.conv5b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=4, padding=4)
        self.conv5c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=4, padding=4)

        # 1x1 convolutional layer for classification
        if self.classify:
        	self.conv_final = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)

        # CONVERTED CONV BLOCK (6):
        # max pool -- REPLACED WITH DILATIONS
        # self.conv6a = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, dilation=4)
        # self.dropout6a = nn.Dropout2d()
        # self.conv6b = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, dilation=4)
        # self.dropout6b = nn.Dropout2d()
        # self.conv6c = nn.Conv2d(in_channels=4096, out_channels=1, kernel_size=1, dilation=4)

    def load_vgg_weights(self, trainable=True):
    	'''
    	Loads VGG state dict stored at MODEL_ZOO_PATH 
    	and initiates the current model's weights

    	NOTE: we load the VGG16 state dict, then create a list of 
    	only the feature (ie convolutional) weights. Then, we create a corresponding
    	list of the current model, so get rid of just the last 2 params which 
    	reflect the conv_final layer
    	'''
    	self.trainable = trainable 

    	# So extract the convolutional weights only from VGG16
    	vgg_state_dict = torch.load(os.path.join(MODEL_ZOO_PATH, "vgg16.pt"))
    	vgg_feature_weights = [v for k,v in vgg_state_dict.items() if "features" in k]

    	# If our current model contains the conv_final then we want to not set that
    	param_list = list(self.parameters())
    	if self.classify:
    		assert len(param_list) - 2 == len(vgg_feature_weights)
    	else:
    		assert len(param_list) == len(vgg_feature_weights)

    	# Loop over the named parameters, updating weights and fixing, if necessary
    	for i, (name, param) in enumerate(self.named_parameters()):
    		if "conv_final" in name:
    			break    		

    		assert param.data.shape == vgg_feature_weights[i].shape
    		param.data = vgg_feature_weights[i]
    		if not self.trainable:
    			param.requires_grad_(False)



    def forward(self, x):

        # CONV BLOCK (1)
        #x1 = self.padding_layer(x)
        x1 = F.relu(self.conv1a(x))
        x1 = F.relu(self.conv1b(x1))

        # CONV BLOCK (2)
        x2 = F.max_pool2d(x1, (2,2))
        x2 = F.relu(self.conv2a(x2))
        x2 = F.relu(self.conv2b(x2))

        # CONV BLOCK (3)
        x3 = F.max_pool2d(x2, (2,2))
        x3 = F.relu(self.conv3a(x3))
        x3 = F.relu(self.conv3b(x3))
        x3 = F.relu(self.conv3c(x3))

        # CONV BLOCK (4)
        #x4 = F.max_pool2d(x3, (2,2))
        x4 = F.relu(self.conv4a(x3))
        x4 = F.relu(self.conv4b(x4))
        x4 = F.relu(self.conv4c(x4))

        # CONV BLOCK (5)
        #x5 = F.max_pool2d(x4, (2,2)) -- REPLACED WITH DILATIONS
        x5 = F.relu(self.conv5a(x4))
        x5 = F.relu(self.conv5b(x5))
        x5 = F.relu(self.conv5c(x5))

        # CONVERTED CONV BLOCK (6):
        #x6 = F.max_pool2d(x5, (2,2)) -- REPLACED WITH DILATIONS        
        # x6 = self.dropout6a(F.relu(self.conv6a(x5)))
        # x6 = self.dropout6b(F.relu(self.conv6b(x6)))
        # x6 = F.relu(self.conv6c(x6))

        if self.classify:
        	x5 = torch.sigmoid(self.conv_final(x5))

        return x5

class SegmentationDataset(utils.data.Dataset):
    '''
    Class defines segmentation (ie image with mask) dataset
    '''

    def __init__(self, root_path, list_common_trans=None, list_img_trans=None, f_type="PIL", trim=0):
        '''NOTE: transforms should be TENSOR transforms, and the PIL->Tensor transform is 
        already included
        '''
        self.image_root = os.path.join(root_path, "image")
        self.mask_root = os.path.join(root_path, "mask")
        self.files = self._build_file_list(self.image_root, self.mask_root)
        self.list_common_trans = list_common_trans
        self.list_img_trans = list_img_trans
        self.f_type = f_type
        self.trim = trim 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        f = self.files[index]

        # Load image and mask, convert to Tensors
        if self.f_type == "PIL":
            image = transforms.ToTensor()(Image.open(os.path.join(self.image_root, f)))
            

        else:
            assert self.f_type == "Numpy_array"
            f_np = f.replace(".png", ".npy")
            image = torch.from_numpy(np.load(os.path.join(self.image_root, f_np))).type(torch.float32)
            image = image.permute(2,0,1)

        mask = transforms.ToTensor()(Image.open(os.path.join(self.mask_root, f)))

        # Join the image and mask, so random transforms can be applied consistently
        if self.list_common_trans is not None:
            image_channels = image.shape[0]
            mask_channels = mask.shape[0]

            # Stack along the channel dimension
            #mask = mask.type_as(image)
            image_mask_combined = torch.cat((image, mask), 0)

            # Apply common tranforms
            for t in self.list_common_trans:
                image_mask_combined = t(image_mask_combined)

            #image_mask_combined = torch.Tensor(image_mask_combined)
            assert image_mask_combined.shape[1:] == image.shape[1:]
            
            # Split back out
            image = image_mask_combined[0:image_channels,:,:]
            mask = image_mask_combined[image_channels:,:,:]

        # Apply the image only transforms
        if self.list_img_trans is not None:
            image = transforms.Compose(self.list_img_trans)(image)

        mask = convert_img_to_2D_mask(mask)

        assert image.shape[1:] == mask.shape

        # If the model reduces the resolution, return just the center
        if self.trim > 0:
            d = mask.shape[0]
            mask = mask[self.trim:d-self.trim, self.trim:d-self.trim]

        return image, mask 


    def _build_file_list(self, image_root, mask_root):
        '''
        Helper function to make sure files are consistent 
        across masks and images
        '''

        image_files = os.listdir(image_root)
        mask_files = os.listdir(mask_root)

        image_files_noext = [x.replace(".png", "").replace(".npy", "") for x in image_files]
        mask_files_noext = [x.replace(".png", "").replace(".npy", "") for x in mask_files]

        s0 = set(image_files_noext)
        s1 = set(mask_files_noext)

        assert len(s0.difference(s1)) == 0
        
        return mask_files  


def convert_img_to_2D_mask(tensor_img):
    '''
    Helper function which converts a 3D mask stored as a Tensor
    to a 2D binary Tensor mask

    Inputs:
        tensor_img: 3D Tensor
    Returns:
        tensor_mask: 2D binary mask
    '''

    # Check that masks are consistent across the channel dimension
    assert tensor_img.std(dim=0).max().item() == 0  # Channels are redundant
    assert tensor_img.max() in (0,1)                # Vals are always either 0 or 1
    assert tensor_img.min() in (0,1)                # Vals are always either 0 or 1

    return tensor_img[0,:,:]


