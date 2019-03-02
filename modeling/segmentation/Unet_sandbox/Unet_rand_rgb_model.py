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

class Unet(nn.Module):

    def __init__(self, input_channels, img_size, layer2_ft=64):
        super(Unet, self).__init__()

        # Define layers for encoding
        self.ft2 = layer2_ft
        self.input_channels = input_channels
        self.my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.BN1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.BN1b = nn.BatchNorm2d(64)

        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.BN2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.BN2b = nn.BatchNorm2d(128)

        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.BN3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.BN3b = nn.BatchNorm2d(256)
        
        self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.BN4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.BN4b = nn.BatchNorm2d(512)

        # Define layers for bottleneck
        self.bot_conv1a = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bot_BNa = nn.BatchNorm2d(1024)
        self.bot_conv1b = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.bot_BNb = nn.BatchNorm2d(1024)
       
        # Define layers for decoding
        self.tconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv1a = nn.Conv2d(in_channels=512*2, out_channels=512, kernel_size=3, padding=1)
        self.up_BN1a = nn.BatchNorm2d(512)
        self.up_conv1b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.up_BN1b = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv2a = nn.Conv2d(in_channels=256*2, out_channels=256, kernel_size=3, padding=1)
        self.up_BN2a = nn.BatchNorm2d(256)
        self.up_conv2b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.up_BN2b = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv3a = nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1)
        self.up_BN3a = nn.BatchNorm2d(128)
        self.up_conv3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.up_BN3b = nn.BatchNorm2d(128)

        self.tconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4a = nn.Conv2d(in_channels=64*2, out_channels=64, kernel_size=3, padding=1)
        self.up_BN4a = nn.BatchNorm2d(64)
        self.up_conv4b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.up_BN4b = nn.BatchNorm2d(64)

        # Layer for classification
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)


    def encoder_pass(self, x):

        # Encode
        x1 = F.relu(self.BN1a(self.conv1a( x)))
        x1 = F.relu(self.BN1b(self.conv1b(x1)))

        x2 = F.max_pool2d(x1, (2,2))
        x2 = F.relu(self.BN2a(self.conv2a(x2)))
        x2 = F.relu(self.BN2b(self.conv2b(x2)))

        x3 = F.max_pool2d(x2, (2,2))
        x3 = F.relu(self.BN3a(self.conv3a(x3)))
        x3 = F.relu(self.BN3b(self.conv3b(x3)))

        x4 = F.max_pool2d(x3, (2,2))
        x4 = F.relu(self.BN4a(self.conv4a(x4)))
        x4 = F.relu(self.BN4b(self.conv4b(x4)))

        return (x1, x2, x3, x4)

    def forward(self, x):

        x1, x2, x3, x4 = self.encoder_pass(x)

        # Bottleneck pass
        y = F.max_pool2d(x4, (2,2))
        y = F.relu(self.bot_BNa(self.bot_conv1a(y)))
        y = F.relu(self.bot_BNb(self.bot_conv1b(y)))

        # Decoder pass 
        y = self.tconv1(y)
        y = F.relu(self.up_BN1a(self.up_conv1a(torch.cat((y,x4),1))))
        y = F.relu(self.up_BN1b(self.up_conv1b(y)))

        y = self.tconv2(y)
        y = F.relu(self.up_BN2a(self.up_conv2a(torch.cat((y,x3),1))))
        y = F.relu(self.up_BN2b(self.up_conv2b(y)))

        y = self.tconv3(y)
        y = F.relu(self.up_BN3a(self.up_conv3a(torch.cat((y,x2),1))))
        y = F.relu(self.up_BN3b(self.up_conv3b(y)))

        y = self.tconv4(y)
        y = F.relu(self.up_BN4a(self.up_conv4a(torch.cat((y,x1),1))))
        y = F.relu(self.up_BN4b(self.up_conv4b(y)))

        y = torch.sigmoid(self.final_conv(y))
        assert y.shape[-2:] == x.shape[-2:] 

        return y


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



