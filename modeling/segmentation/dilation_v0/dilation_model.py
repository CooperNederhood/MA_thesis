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


MODEL_NAME = "dilation_v1"

class DilationNet_v1(nn.Module):

    def __init__(self, input_channels, img_size):
        super(DilationNet_v1, self).__init__()

        self.input_channels = input_channels
        self.img_size = img_size
        self.my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.c1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, dilation=2)
        self.c4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, dilation=2)

        self.c5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, dilation=3)
        self.c6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=3)
        self.c7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=3)

        self.c8 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=7, dilation=3)
        self.c9 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.c10 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1)


    def forward(self, x):

        # Use ReLU activation for all Conv except final
        for l in self.children()[0:-1]:
            x = F.relu(l(x))

        # Use sigmoid for final Conv
        x = torch.sigmoid(self.c10(x))

        return x 

class SegmentationDataset(utils.data.Dataset):
    '''
    Class defines segmentation (ie image with mask) dataset
    '''

    def __init__(self, root_path, list_common_trans=None, list_img_trans=None, trim=0):
        '''NOTE: transforms should be TENSOR transforms, and the PIL->Tensor transform is 
        already included
        '''
        self.image_root = os.path.join(root_path, "image")
        self.mask_root = os.path.join(root_path, "mask")
        self.files = self._build_file_list(self.image_root, self.mask_root)
        self.list_common_trans = list_common_trans
        self.list_img_trans = list_img_trans
        self.trim = trim 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        f = self.files[index]

        # Load image and mask, convert to Tensors
        image = transforms.ToTensor()(Image.open(os.path.join(self.image_root, f)))
        mask = transforms.ToTensor()(Image.open(os.path.join(self.mask_root, f)))

        # Join the image and mask, so random transforms can be applied consistently
        if self.list_common_trans is not None:
            image_channels = image.shape[0]
            mask_channels = mask.shape[0]

            # Stack along the channel dimension
            image_mask_combined = torch.cat((image, mask), 0)

            # Apply common tranforms
            for t in self.list_common_trans:
                image_mask_combined = t(image_mask_combined)

            image_mask_combined = torch.Tensor(image_mask_combined)
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
        assert image_files == mask_files

        return image_files 


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

