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
sys.path.append('../')
from utilities import cnn_utils, transform_utils, test_eval


MODEL_NAME = "dil_net_front"
MODEL_DETAILS = '''This is an exact copy of the corresponding dil_net model.
We simply change the VGG weights st they are trainable
'''

VGG_TRAIN = True 

from dil_net0 import dil_net as model_def

EPOCH_COUNT = 1 
BATCH_SIZE = 4
img_size = 256
input_channels = 3
TRIM = int((img_size-64)/2)
PADDING = 0


# Root directory for dataset
if input_channels == 3:
    root = '../../../data/training_data/descartes/RGB'
else:
    root = '../../../data/training_data/descartes/Four_band'

data_root = os.path.join(root, "segmentation/size_{}".format(img_size))

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
    epoch_loss_dict = {'train': {'acc': [], 'loss':[], 'IoU':[], 'time':[]}, 
                         'val': {'acc': [], 'loss':[], 'IoU':[], 'time':[]}}
    batch_loss_dict = {'train': {'acc': [], 'loss':[], 'IoU':[]}, 
                         'val': {'acc': [], 'loss':[], 'IoU':[]}}

    if detailed_time:
        epoch_loss_dict['train']['backward_pass_time'] = []
        epoch_loss_dict['train']['data_fetch_time'] = []
        epoch_loss_dict['val']['backward_pass_time'] = []
        epoch_loss_dict['val']['data_fetch_time'] = []

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
            running_corrects = 0
            running_IoU = 0.0
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
                img_size = target.shape[-1]
                #assert img_size == 128

                target = torch.tensor(target, dtype=torch.float32, device=device)
                images = images.to(device)

                batch_size = target.shape[0]
                total_obs += batch_size
                
                # Zero the gradients
                model.zero_grad()
                # Forward pass -- reshape output to be like the target (has extra 1D dimension)
                output = model(images)
                output = output.view(target.shape)

                # Just round since we have binary classification
                preds = torch.round(output)
                correct_count = (preds == target).sum()
                
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
                batch_loss_dict[phase]['acc'].append(correct_count.item()/batch_size)
                batch_loss_dict[phase]['loss'].append(error.item())
                batch_loss_dict[phase]['IoU'].append(test_eval.inter_over_union(preds, target))

                running_loss += error.item() * batch_size
                running_corrects += correct_count.item()
                running_IoU += test_eval.inter_over_union(preds, target) * batch_size

                if detailed_time: 
                    b_data_fetch = time.time()

            epoch_loss = running_loss / total_obs
            epoch_acc = running_corrects / (total_obs*img_size*img_size) 
            
            epoch_IoU = running_IoU / total_obs 

            if epoch_loss < current_best_loss:
                current_best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            t = time.time()-begin_epoch_time
            # Add to our epoch_loss_dict
            epoch_loss_dict[phase]['acc'].append(epoch_acc)
            epoch_loss_dict[phase]['loss'].append(epoch_loss)
            epoch_loss_dict[phase]['IoU'].append(epoch_IoU)
            epoch_loss_dict[phase]['time'].append(t)
            if detailed_time:
                epoch_loss_dict[phase]['backward_pass_time'].append(total_pass)
                epoch_loss_dict[phase]['data_fetch_time'].append(total_data_fetch)

            print("PHASE={} EPOCH={} TIME={} LOSS={} ACC={}".format(phase, 
                epoch, t, epoch_loss, epoch_acc))


    return model, best_model_wts, epoch_loss_dict, batch_loss_dict
   
# Define trasnforms
common_transforms = [transform_utils.RandomHorizontalFlip(0.5), 
                     transform_utils.RandomVerticalFlip(0.5)]
#img_transforms = [transforms.ColorJitter()]

# Define network
net = model_def.FrontEnd(input_channels, img_size, PADDING, classify=True)
net.load_vgg_weights(VGG_TRAIN)

# Define dataloaders
train_root = os.path.join(data_root, "train")
val_root = os.path.join(data_root, "val")

train_dset = model_def.SegmentationDataset(train_root, list_common_trans=common_transforms,
                                  list_img_trans=None, f_type = "PIL", trim=TRIM)
val_dset = model_def.SegmentationDataset(val_root, f_type="PIL", trim=TRIM)

train_dset_loader = utils.data.DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
val_dset_loader = utils.data.DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=True)

dset_loader_dict = {'train':train_dset_loader, 'val':val_dset_loader}

criterion_loss = nn.BCELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
optimizer = optim.Adam(net.parameters())


# trained_net, best_model_wts, epoch_loss_dict, batch_loss_dict = train_segmentation(net, EPOCH_COUNT, dset_loader_dict, criterion_loss, optimizer)

# cnn_utils.save_model(net, MODEL_NAME, best_model_wts, epoch_loss_dict, batch_loss_dict, MODEL_DETAILS, SAVE_ROOT)
