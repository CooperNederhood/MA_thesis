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
from collections import OrderedDict
import matplotlib.pyplot as plt 

import json


# Root directory for dataset
root = '../../data/training_data'
data_root = os.path.join(root, "classification/size_128")

# Check whether GPU is enabled
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""# Begin script"""
class ConvPassClassifyNet(nn.Module):

    '''
    Implement a basic sequential forward pass
    '''

    def __init__(self, input_channels, img_size):
        super(ConvPassClassifyNet, self).__init__()

        self.input_channels = input_channels
        self.my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((2,2))

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x        

    def get_layer_output(self, x, layer_name):
        '''
        Exits forward pass early and returns the 
        specified layer
        '''

        for name, layer in self.named_children():
            x = layer(x)
            if name == layer_name:
                return x

        print("Desired error not found -- forward pass completed")
        return x 

class FullyConnectedClassifyNet(nn.Module):
    '''
    Implement a basic full-connected classification model
    '''

    def __init__(self, begin_len):
        super(FullyConnectedClassifyNet, self).__init__()

        self.dout_rate = 0.5

        self.fc1 = nn.Linear(begin_len, 400)
        self.activ1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dout_rate)

        self.fc2 = nn.Linear(400, 1)
        self.activ2 = nn.Sigmoid()

    def forward(self, x):

        for layer in self.children():
            x = layer(x)
        return x 

class classifyNet(nn.Module):

    def __init__(self, input_channels, img_size):
        super(classifyNet, self).__init__()

         # Define the forward pass
        self.conv_net = ConvPass(input_channelsS, img_size)

        # Determine dimension of flattened input to fully-connected net
        self.conv_end_ft_count = self._determine_ft_count(img_size)
        print(self._determine_ft_count(img_size))

        # Define FC layers
        self.class_net = FullyConnectedClassifier(self.conv_end_ft_count)

    def _determine_ft_count(self, img_size):
        '''
        Calculate the output ft count of what comes out of 
        the end of our convolutional base
        '''

        example = torch.randn(1, self.INPUT_CHANNELS, img_size, img_size)
        conv_example = self.conv_net(example)
        
        return self.num_flat_features(conv_example)

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.class_net(x)

        return x 


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def build_dataloader_dict(data_root, batch_size, list_of_transforms=None):
    '''
    Once you specify the data_root, will return a dictionary
    containing train and val dataloaders
    '''
    if list_of_transforms is None:
        trans = transforms.ToTensor()
    else:
        #assert transforms[-1] == transforms.ToTensor()
        trans = transforms.Compose(list_of_transforms)

    train_data = datasets.ImageFolder(os.path.join(data_root, "train"), transform=trans)
    val_data = datasets.ImageFolder(os.path.join(data_root, "val"), transform=transforms.ToTensor())

    train_dataloader = utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    dataloader_dict = {'val': val_dataloader, 'train': train_dataloader}

    return dataloader_dict

def get_batch(dataloader_dict, d_type):
    '''
    Just a helper function, fetches an example from the dataloader
    specified to allow for easier debugging
    '''

    for data in dataloader_dict[d_type]:
        break

    return data


def train_model(model, num_epochs, dataloader_dict, criterion, optimizer, verbose=False):
    print("Starting training loop...\n")
    print("Model's device = {}".format(model.my_device))

    device = model.my_device

    # Save a copy of the weights, before we start training
    #begin_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts = copy.deepcopy(model.state_dict())
    current_best_acc = 1000

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
            running_corrects = 0
            total_obs = 0

            # For each mini-batch in the dataloader
            for i, data in enumerate(dataloader_dict[phase]):

                images, target = data
                target = torch.tensor(target, dtype=torch.float32, device=device)
                images = images.to(device)

                batch_size = target.shape[0]
                total_obs += batch_size
                
                # Zero the gradients
                model.zero_grad()
                # Forward pass
                output = model(images)
                output = output.view(batch_size)

                preds = torch.round(output)
                
                # Calculate loss
                error = criterion(output, target)
                correct = preds==target
                incorrect = preds!=target
                correct_count = torch.sum(correct)

                # Make detailed output if verbose
                verbose_str = ""
                if verbose:
                    np_output = output.detach()
                    np_preds = preds.detach()
                    np_target = target.detach()
                    df = pd.DataFrame({'output': np_output, 'preds': np_preds, 'target':np_target})
                    df_c = df.groupby(by=['preds', 'target']).count().reset_index()
                    df_m = df.groupby(by=['preds', 'target']).mean().reset_index()
                    print(df_c)
                    print(df_m)

                # Training steps
                if phase == 'train':
                    # Backpropogate the error
                    error.backward()
                    # Take optimizer step
                    optimizer.step()

                # The error is divided by the batch size, so reverse this
                running_loss += error.item() * batch_size
                running_corrects += correct_count 

                print('%s - [%d/%d][%d/%d]\tError: %.4f\tAccuracy: %d/%d\t%s' % 
                    (phase, epoch, num_epochs, i, len(dataloader_dict[phase]), 
                        error.item(), correct_count, batch_size, verbose_str))
            epoch_loss = running_loss / total_obs
            epoch_acc = (running_corrects.double() / total_obs).item()
            
            if epoch_acc < current_best_acc:
              current_best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())

            # Add to our epoch_loss_dict
            epoch_loss_dict[phase]['acc'].append(epoch_acc)
            epoch_loss_dict[phase]['loss'].append(epoch_loss)

            print("\nPHASE={} EPOCH={} TIME={} LOSS={} ACC={}\n".format(phase, 
                epoch, time.time()-begin_epoch_time, epoch_loss, epoch_acc))
            

    return model, best_model_wts, epoch_loss_dict

EPOCH_COUNT = 25
img_size = 128

# Define network
net = classifyNet(img_size)

# Define dataloaders
#data_root =  "/home/cooper/Documents/MA_thesis/data/training_data/classification/size_128"
trans_list = [transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5), transforms.ToTensor()]
dataloader_dict = build_dataloader_dict(data_root, 16, trans_list)

# Define criterion function
criterion = nn.BCELoss()

# Define optimizer
net = net.to(device)
optimizer = optim.Adam(net.parameters())

print(net)
print(device)

net.my_device

trained_model, best_wts, epoch_loss_dict = train_model(net, EPOCH_COUNT, dataloader_dict, criterion, optimizer)

# Save out
f = 'model.pt'
training_f = 'training_hist.json'

torch.save(best_wts, os.path.join(root, f))
with open(os.path.join(root,training_f), 'w') as fp:
  json.dump(epoch_loss_dict, fp)


