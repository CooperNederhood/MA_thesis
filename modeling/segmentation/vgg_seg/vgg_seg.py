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

MODEL_NAME = "vgg_seg"

class segNetVGG(nn.Module):

    def __init__(self, img_size):
        super(segNetVGG, self).__init__()

        # Define layers for encoding
        self.my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        # self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        # self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        # self.conv5a = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        # self.conv5b = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)

        # Define layers for decoding
        self.tconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv1a = nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1)
        self.up_conv1b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.tconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv2a = nn.Conv2d(in_channels=64*2, out_channels=64, kernel_size=3, padding=1)
        self.up_conv2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        #self.tconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=2)

        # Layer for classification
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)


    def encoder_pass(self, x):

        # Encode 1
        x = F.relu(self.conv1a(x))
        x1 = F.relu(self.conv1b(x))

        # Encode 2
        x2 = F.max_pool2d(F.relu(self.conv2a(x1)), 2)
        x2 = F.relu(self.conv2b(x2))

        # Encode 3 -- no downsampling here
        x3 = F.max_pool2d(F.relu(self.conv3a(x2)), 2)
        #x3 = F.max_pool2d(F.relu(self.conv3b(x3)), 2)
        x3 = F.relu(self.conv3b(x3))

        # # Encode 4
        # x = F.relu(self.conv4a(x))
        # x = F.max_pool2d(F.relu(self.conv4b(x)), 2)

        # # Encode 5
        # x = F.relu(self.conv5a(x))
        # x = F.relu(self.conv5b(x))

        return (x1, x2, x3)

    def decoder_pass(self, x_list):

        x1, x2, x3 = x_list

        # Decode 1
        x4 = self.tconv1(x3)
        x4 = F.relu(self.up_conv1a(torch.cat((x2, x4),1)))
        x4 = F.relu(self.up_conv1b(x4))

        # Decode 2 
        x5 = self.tconv2(x4)
        x5 = F.relu(self.up_conv2a(torch.cat((x1, x5),1)))
        x5 = F.relu(self.up_conv2b(x5))

        return (x4, x5)

    def initialize_weights(self, vgg):
        '''
        Initialize weights to trained VGG model
        '''

        vgg_conv = [l for i, l in enumerate(vgg.features) if i in {0, 2, 5, 7, 10, 12}]
        seg_conv = list(self.children())[0:6]

        for vgg_layer, seg_layer in zip(vgg_conv, seg_conv):
            assert vgg_layer.weight.shape == seg_layer.weight.shape

            seg_layer.weight.data = vgg_layer.weight.data 
            seg_layer.bias.data = vgg_layer.bias.data 

    def forward(self, x):

        x1, x2, x3 = self.encoder_pass(x)
        x4, x5 = self.decoder_pass((x1,x2,x3))

        segmented_image = torch.sigmoid(self.final_conv(x5))

        assert segmented_image.shape[-2:] == x.shape[-2:] 

        return segmented_image


class SegmentationDataset(utils.data.Dataset):
    '''
    Class defines segmentation (ie image with mask) dataset
    '''

    def __init__(self, root_path, list_common_trans=None, list_img_trans=None):
        '''NOTE: transforms should be TENSOR transforms, and the PIL->Tensor transform is 
        already included
        '''
        self.image_root = os.path.join(root_path, "image")
        self.mask_root = os.path.join(root_path, "mask")
        self.files = self._build_file_list(self.image_root, self.mask_root)
        self.list_common_trans = list_common_trans
        self.list_img_trans = list_img_trans

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

def get_batch(dataloader_dict, d_type):
    '''
    Just a helper function, fetches an example from the dataloader
    specified to allow for easier debugging
    '''

    for data in dataloader_dict[d_type]:
        break

    return data



