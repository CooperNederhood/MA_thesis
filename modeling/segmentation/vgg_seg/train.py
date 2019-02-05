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


MODEL_NAME = "seg_vgg"
MODEL_DETAILS = '''Segmentation model using weights from VGG. Contains random hor/vert flips 
for data augmentation and add some additional color shifts.
EPOCH_COUNT = 15; BATCH_SIZE=16; img_size=128
'''

import vgg_seg as model_def

EPOCH_COUNT = 1
BATCH_SIZE = 16
CHANNELS = 3
img_size = 128
input_channels = 3


# Root directory for dataset
root = '../../../data/training_data'
data_root = os.path.join(root, "segmentation/size_128")

SAVE_ROOT = "./" 

# Check whether GPU is enabled
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train_segmentation(model, num_epochs, dataloader_dict, criterion, optimizer, verbose=False, detailed_time=False):
    print("Starting training loop...\n")
    print("Model's device = {}".format(model.my_device))

    device = model.my_device

    # Track best model
    best_model_wts = copy.deepcopy(model.state_dict())
    current_best_loss = 1000

    # Initialize loss dict to record training, figures are per epoch
    epoch_loss_dict = {'train': {'acc': [], 'loss':[], 'time':[]}, 
                         'val': {'acc': [], 'loss':[], 'time':[]}}

    # For each epoch
    for epoch in range(num_epochs):
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            begin_epoch_time = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            #running_corrects = 0
            total_obs = 0

            # For each mini-batch in the dataloader
            total_data_fetch = 0
            total_pass = 0
            b_data_fetch = time.time()
            for i, data in enumerate(dataloader_dict[phase]):

                if detailed_time: 
                    total_data_fetch += (time.time() - b_data_fetch)
                    b_pass = time.time()

                images, target = data
                target = torch.tensor(target, dtype=torch.float32, device=device)
                images = images.to(device)

                batch_size = target.shape[0]
                total_obs += batch_size
                
                # Zero the gradients
                model.zero_grad()
                # Forward pass -- reshape output to be like the target (has extra 1D dimension)
                output = model(images)
                output = output.view(target.shape)

                #preds = torch.round(output)
                
                # Calculate loss
                error = criterion(output, target)

                # Make detailed output if verbose
                verbose_str = ""

                # Training steps
                if phase == 'train':
                    # Backpropogate the error
                    error.backward()
                    # Take optimizer step
                    optimizer.step()
                if detailed_time: 
                    total_pass += (time.time() - b_pass)

                # The error is divided by the batch size, so reverse this
                running_loss += error.item() * batch_size
                #running_corrects += correct_count 

                if detailed_time: 
                    b_data_fetch = time.time()

            epoch_loss = running_loss / total_obs
            epoch_acc = 0.5
            #epoch_acc = (running_corrects.double() / total_obs).item()

            if epoch_loss < current_best_loss:
                current_best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            t = time.time()-begin_epoch_time
            # Add to our epoch_loss_dict
            epoch_loss_dict[phase]['acc'].append(epoch_acc)
            epoch_loss_dict[phase]['loss'].append(epoch_loss)
            epoch_loss_dict[phase]['time'].append(t)
            if detailed_time:
                epoch_loss_dict[phase]['backward_pass_time'].append(total_pass)
                epoch_loss_dict[phase]['data_fetch_time'].append(total_data_fetch)

            print("PHASE={} EPOCH={} TIME={} LOSS={} ACC={}".format(phase, 
                epoch, t, epoch_loss, epoch_acc))


    return model, best_model_wts, epoch_loss_dict 
    
# Define trasnforms
common_transforms = [transform_utils.RandomHorizontalFlip(0.5), 
                     transform_utils.RandomVerticalFlip(0.5)]
#img_transforms = [transforms.ColorJitter()]

# Define network
model_zoo = "../../../model_zoo"
net = model_def.segNetVGG(img_size)
vgg = models.vgg16(pretrained=False)
vgg.load_state_dict(torch.load(os.path.join(model_zoo, "vgg16.pt")), map_location='gpu')
net.initialize_weights(vgg)

# # Define dataloaders
train_root = os.path.join(data_root, "train")
val_root = os.path.join(data_root, "val")

train_dset = model_def.SegmentationDataset(train_root, list_common_trans=common_transforms,
                                 list_img_trans=None)
val_dset = model_def.SegmentationDataset(val_root)

train_dset_loader = utils.data.DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
val_dset_loader = utils.data.DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=True)

dset_loader_dict = {'train':train_dset_loader, 'val':val_dset_loader}

criterion_loss = nn.BCELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
optimizer = optim.Adam(net.parameters())


trained_net, best_model_wts, training_hist = train_segmentation(net, EPOCH_COUNT, dset_loader_dict, criterion_loss, optimizer)

cnn_utils.save_model(net, MODEL_NAME, best_model_wts, training_hist, MODEL_DETAILS, SAVE_ROOT)
