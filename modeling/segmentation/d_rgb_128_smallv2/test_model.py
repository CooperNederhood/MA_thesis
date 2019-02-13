
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

MODEL_NAME = "d_rgb_128_small"

EPOCH_COUNT = 25
BATCH_SIZE = 3 
CHANNELS = 3
img_size = 128
input_channels = 3


# Root directory for dataset
root = '../../../data/training_data/descartes/RGB'
data_root = os.path.join(root, "segmentation/size_{}".format(img_size))

SAVE_ROOT = "./" 

# Check whether GPU is enabled
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class TestNet(nn.Module):

    def __init__(self, input_channels, img_size):
        super(TestNet, self).__init__()

        # Define layers for encoding
        self.input_channels = input_channels
        self.my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        #self.conv3a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        # Define layers for bottleneck
        self.conv3a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        
        # Define layers for decoding
        self.tconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.up_conv1a = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.tconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
        self.up_conv2a = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        #self.tconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        #self.up_conv2a = nn.Conv2d(in_channels=32*2, out_channels=32, kernel_size=5, padding=2)

        # Layer for classification
        #self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):

        # encode
        x = F.relu(self.conv1a(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2a(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3a(x))
    

        x = self.tconv1(x)
        x = F.relu(self.up_conv1a(x))
        x = self.tconv2(x)
        x = F.sigmoid(self.up_conv2a(x))

        return x 


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

        # LIMIT TO JUST "ona_id20_image"
        rv_limited = [x for x in image_files if "ona_id20_image" in x]

        return rv_limited 


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = TestNet(3, 128).to(device)
root = '../../../data/training_data/descartes/RGB'
data_root = os.path.join(root, "segmentation/size_{}".format(img_size))

train_root = os.path.join(data_root, "train")
val_root = os.path.join(data_root, "val")

train_dset = SegmentationDataset(train_root)
img, mask = train_dset[0]
img_b = img.unsqueeze(0)
mask_b = mask.unsqueeze(0)

criterion_loss = nn.BCELoss()

optimizer = optim.Adam(net.parameters()).to(device)

ITER_COUNT = 30
train_dset_sub = [train_dset[0], train_dset[7]]
for i in range(ITER_COUNT):

    iter_loss = 0
    for img, mask in train_dset_sub:
        optimizer.zero_grad()

        img_b = img.unsqueeze(0)
        mask_b = mask.unsqueeze(0)

        out = net(img_b).squeeze(0)

        out = out.view(mask_b.shape)

        loss = criterion_loss(out, mask_b)

        loss.backward()
        optimizer.step()

        iter_loss += loss 

    print("[{}][{}] loss = {}".format(i, ITER_COUNT, iter_loss/len(train_dset)))
