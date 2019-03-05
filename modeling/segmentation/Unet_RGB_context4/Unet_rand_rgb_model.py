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

class SegmentationDataset(utils.data.Dataset):
    '''
    Class defines segmentation (ie image with mask) dataset
    '''

    def __init__(self, root_path, list_common_trans=None, list_img_trans=None, f_type="PIL"):
        '''NOTE: transforms should be TENSOR transforms, and the PIL->Tensor transform is 
        already included
        '''
        self.image_root = os.path.join(root_path, "image")
        self.mask_root = os.path.join(root_path, "mask")
        self.files = self._build_file_list(self.image_root, self.mask_root)
        self.list_common_trans = list_common_trans
        self.list_img_trans = list_img_trans
        self.f_type = f_type

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



