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


sys.path.append("../")
from segmentation.Unet_rand_RGB import Unet_rand_rgb_model as Unet 
from segmentation.dil_net0 import dil_net as dilation_vgg
import test_eval 

def identity_weights(layer):

    # Only update convolution layers with kernel=3, the final convol
    #       layer decreases the layer count to 2D
    if type(layer) == nn.Conv2d:
        if layer.weight.shape[-1] == 3:
            
            mid = 1

            layer.bias.data.fill_(0.0)
            layer.weight.data.fill_(0.0)

            in_channels = layer.weight.shape[1]
            num_filters = layer.weight.shape[0]



            for i in range(num_filters):
                layer.weight.data[i, i, mid, mid] = 1




class FrontEnd_ContextModel(nn.Module):

    def __init__(self, front_end_type, path_to_front_end_weights, is_gpu, input_channels, img_size, 
                    context_layer_count, output_channels, load_weights=True, include_activ=True):
        '''
        Inupts:
            front_end_type: (str) specifying one of (Unet, vgg_orig, vgg_tuned)
            path_to_front_end_weights: (str) to the folder containing the model.pt weights to load for front end
            is_gpu: (bool) when loading the weights we need to know whether the target is cpu or gpu
            
        '''
        super(FrontEnd_ContextModel, self).__init__()

        self.front_end_type = front_end_type
        self.path_to_front_end_weights = path_to_front_end_weights
        self.is_gpu = is_gpu 
        self.input_channels = input_channels
        self.img_size = img_size
        self.context_layer_count = context_layer_count
        self.output_channels = output_channels
        self.load_weights = load_weights
        self.include_activ = include_activ

        self.my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the front-end model and load the weights
        # Note, the 'vgg_orig' and 'vgg_tuned' have the same Front End model, they differ
        #     in where they initialize the weights from
        if front_end_type == "Unet":
            self.FE_model = Unet.Unet(input_channels, img_size, include_final_conv=False)
            context_channels = 64

        elif front_end_type == "vgg_orig":
            # get weights from dil_net0
            self.FE_model = dilation_vgg.FrontEnd(input_channels, img_size, classify=False)
            context_channels = 512

        elif front_end_type == "vgg_tuned":
            # get weights from dil_alt0
            self.FE_model = dilation_vgg.FrontEnd(input_channels, img_size, classify=False)
            context_channels = 512
        else:
            print("Mispecified front-end model")

        if self.load_weights:
            self.FE_model = test_eval.load_weights(self.FE_model, self.path_to_front_end_weights, self.is_gpu)
        self.Context_model = ContextModel(context_channels, self.context_layer_count, self.output_channels, self.include_activ)

        self._init_context_weights()

    def _init_context_weights(self):
        self.Context_model.init_weights_to_identity()

    def fix_front_end_weights(self):
        for p in self.FE_model.parameters():
            p.requires_grad_(False)

    def forward(self, x):

        x = self.FE_model(x)
        x = self.Context_model(x)

        return x 

class ContextModel(nn.Module):

    def __init__(self, input_channels, context_layer_count, output_channels, include_activ=True):
        super(ContextModel, self).__init__()

        self.c = input_channels
        self.depth = context_layer_count
        self.output_channels = output_channels
        self.include_activ = include_activ

        assert self.depth in [1, 2, 3, 4, 5, 6]

        if self.depth >= 1:
            self.context1 = nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, dilation=1)
        if self.depth >= 2:
            self.context2 = nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, dilation=1)
        if self.depth >= 3:
            self.context3 = nn.Conv2d(self.c, self.c, kernel_size=3, padding=2, dilation=2)
        if self.depth >= 4:
            self.context4 = nn.Conv2d(self.c, self.c, kernel_size=3, padding=4, dilation=4)
        if self.depth >= 5:
            self.context5 = nn.Conv2d(self.c, self.c, kernel_size=3, padding=8, dilation=8)
        if self.depth >= 6:
            self.context6 = nn.Conv2d(self.c, self.c, kernel_size=3, padding=16, dilation=16)

        self.prefinal_conv = nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, dilation=1)
        self.final_conv = nn.Conv2d(self.c, self.output_channels, kernel_size=1)

    def init_weights_to_identity(self):
        '''
        The weights are initialized st convolutions act as the identity
        '''
        self.apply(identity_weights) 


    def forward(self, x):

        input_width = x.shape[-1]

        for name, layer in self.named_children():
            if name == "final_conv":
                x = layer(x)
                if self.include_activ:
                    x = torch.sigmoid(x)

            else:
                x = layer(x)
                if self.include_activ:
                    x = F.relu(x)

        assert x.shape[-1] == input_width

        return x 


