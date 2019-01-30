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

def save_model(model, model_name, state_dict, training_hist, model_details, root_path):

    # Store everything in folder of the model name
    path = os.path.join(root_path, model_name)
    if not os.path.isdir(path):
        os.mkdir(path)

    # Save state dict
    torch.save(state_dict, os.path.join(path, 'model.pt') )

    # Save training history
    with open(os.path.join(path, 'training_hist.json'), 'w') as fp:
        json.dump(training_hist, fp)

    # Write out details of the model
    with open(os.path.join(path, 'Model_specs.txt'), 'w') as fp:
        fp.write("*****"*10)
        fp.write("\n")
        fp.write("MODEL NOTES:")
        fp.write(model_details)
        fp.write("\n\n")
        fp.write("*****"*10)
        fp.write("\n")
        fp.write("MODEL STRUCTURE:")
        fp.write(model.__str__())




def plot_training_dict(training_dict):
    epoch_list = range(25)
    plt.plot(epoch_list, epoch_loss_dict['train']['loss'], label='train')
    plt.plot(epoch_list, epoch_loss_dict['val']['loss'], label='val')
    plt.legend()

    plt.clf()
    plt.plot(epoch_list, epoch_loss_dict['train']['acc'], label='train')
    plt.plot(epoch_list, epoch_loss_dict['val']['acc'], label='val')
    plt.legend()
        

class ConvPass(nn.Module):

    '''
    Implement a basic sequential forward pass
    '''

    def __init__(self, input_channels, img_size=28):
        super(ConvPass, self).__init__()

        self.INPUT_CHANNELS = input_channels

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.INPUT_CHANNELS, out_channels=6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2,2))

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

class FullyConnectedClassifier(nn.Module):
    '''
    Implement a basic full-connected classification model
    '''

    def __init__(self, begin_len):
        super(FullyConnectedClassifier, self).__init__()

        self.fc1 = nn.Linear(begin_len, begin_len//2)
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(begin_len//2, 10)
        self.activ2 = nn.Sigmoid()

    def forward(self, x):

        for layer in self.children():
            x = layer(x)
        return x 

def get_max_activating_image(init_img, net, layer_name, ft_num, iter_count):
    '''
    Inputs:
        init_img: initial guess for max activating image
        net: neural network, should have net.conv_net
        layer_name: name of layer within net.conv_net
        ft_num: int denoting which feature map to analzyze
        iter_count: how many gradient steps to take
    Returns:
        max_img: image that maximizes the output feature activations
    '''
    assert len(init_img.shape) == 4
    assert init_img.shape[0] == 1

    max_img = init_img.requires_grad_(True)

    for i in range(iter_count):
        print("Taking step #",i)
        out = net.conv_net.get_layer_output(max_img, layer_name)[0,ft_num,:,:]
        out = torch.norm(out)
        print("Act norm = ", out)

        out.backward()

        # Now max_img has gradient
        step_size = 1.0
        norm_grad = max_img.grad / torch.norm(max_img.grad)

        new_img = (max_img + step_size*norm_grad).detach()
        new_out = net.conv_net.get_layer_output(new_img, layer_name)[0,ft_num,:,:]

        i = 0
        while torch.norm(new_out).item() < out.item():
            print("Cutting step size in half - ", i)
            step_size *= 0.5
            new_img = (max_img + step_size*norm_grad)
            new_out = net.conv_net.get_layer_output(new_img, layer_name)[0,ft_num,:,:]
            i += 1

        assert new_img.requires_grad == False

        max_img = new_img.requires_grad_(True)

    return max_img 
