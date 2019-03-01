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
import datetime 

from functools import reduce 
#import data_vis

def save_model(model, model_name, state_dict, training_hist, batch_hist, model_details, root_path):

    # Store everything in folder of the model name
    path = os.path.join(root_path, model_name)
    if not os.path.isdir(path):
        os.mkdir(path)

    # Save state dict
    torch.save(state_dict, os.path.join(path, 'model.pt') )

    # Save training history - EPOCH
    with open(os.path.join(path, 'training_hist.json'), 'w') as fp:
        json.dump(training_hist, fp)

    # Save training history - BATCH
    with open(os.path.join(path, 'batch_hist.json'), 'w') as fp:
        json.dump(batch_hist, fp)

    # Write out details of the model
    with open(os.path.join(path, 'Model_specs.txt'), 'w') as fp:
        currentDT = datetime.datetime.now()
        fp.write(currentDT.__str__())
        fp.write("\n")
        fp.write("*****"*10)
        fp.write("\n")
        fp.write("MODEL NOTES:")
        fp.write(model_details)
        fp.write("\n\n")
        fp.write("*****"*10)
        fp.write("\n")
        fp.write("MODEL STRUCTURE:")
        fp.write(model.__str__())
        fp.write("\n\n")
        fp.write("PARAMETER COUNT BY LAYER:\n")
        # print Parameter counts
        total_params = 0
        for name, layer in model.named_children():
            layer_total = 0
            for param in layer.parameters():
                layer_total += reduce(lambda x,y: x*y, param.shape)
            fp.write("Layer {} has {} parameters\n".format(name, layer_total)) 
            total_params += layer_total
        fp.write("\nModel has {} total parameters".format(total_params))

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


def get_layer_output(input_img, net, layer_name, ft_num):
    '''
    Helper function, will make forward pass on input image until
    target layer is reached. Returns the corresponding feature number
    '''

    out = input_img 
    # make forward pass until were at our layer
    for name, layer in net.named_children():
        out = layer(out)
        if name == layer_name:
            break 

    return out[0,ft_num,:,:]


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
        #print("Taking step #",i)

        out = get_layer_output(max_img, net, layer_name, ft_num)

        out = torch.mean(out)
        #print("Activation norm for ft num {} in layer {} current = {}".format(ft_num, layer_name, out))

        out.backward()

        # Now max_img has gradient
        step_size = 1.0
        norm_grad = max_img.grad / torch.norm(max_img.grad)

        new_img = (max_img + step_size*norm_grad).detach()
        new_out = get_layer_output(new_img, net, layer_name, ft_num)

        i = 0
        while torch.norm(new_out).item() < out.item():
            print("Cutting step size in half - ", i)
            step_size *= 0.5
            new_img = (max_img + step_size*norm_grad)
            new_out = get_layer_output(new_img, net, layer_name, ft_num)
            i += 1

        assert new_img.requires_grad == False

        max_img = new_img.requires_grad_(True)

    print("Activation norm for ft num {} in layer {} current = {}".format(ft_num, layer_name, out))

    return max_img 

# path_model_zoo = "../../model_zoo"

# vgg16 = models.vgg16(pretrained=False)
# vgg16.load_state_dict(torch.load(os.path.join(path_model_zoo, "vgg16.pt")))
# for param in vgg16.features.parameters():
#     param.requires_grad = False 

# vgg16_features = vgg16.features 

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# to_pil = transforms.ToPILImage()

# # Iintiailze images
# init_zero = torch.zeros(1,3,256,256)
# init_gray_noise = np.random.random((1, 3, 256, 256)) * 0.01 + 128
# init_gray_noise = torch.from_numpy(init_gray_noise).type(torch.float32)
# init_noise = normalize(torch.randn(3,256,256)).unsqueeze(0)

# image_zero = to_pil(init_zero.squeeze(0))
# image_gray_noise = to_pil(init_gray_noise.squeeze(0))
# image_noise = to_pil(init_noise.squeeze(0))


def gen_max_act_image_map(init_val, net, layer_name, ft_count, iters):

    d = int(np.sqrt(ft_count))
    
    vis_list = []
    for cur_num in range(ft_count):
        max_img = get_max_activating_image(init_val, net, layer_name, cur_num, iters).squeeze(0)
        vis_list.append(to_pil(max_img))

    total_pic = data_vis.join_as_grid(vis_list, (d,d)) 
    return total_pic

# # First 3 convolutional layers of VGG-16
# layers = ['0', '2', '5']

# ft_count = 16
# steps = 1000

# zero_maps = {}
# print("DOING ZERO MAPS")
# for l in layers:
#     zero_maps[l] = gen_max_act_image_map(init_zero, vgg16_features, l, ft_count, steps)
#     zero_maps[l].save('layer_'+l+'_zero.png')

# gray_maps = {}
# print("DOING GRAY MAPS")
# for l in layers:
#     gray_maps[l] = gen_max_act_image_map(init_gray_noise, vgg16_features, l, ft_count, steps)
#     gray_maps[l].save('layer_'+l+'_gray.png')

# noise_maps = {}
# print("DOING NOISE MAPS")
# for l in layers:
#     noise_maps[l] = gen_max_act_image_map(init_noise, vgg16_features, l, ft_count, steps)
#     noise_maps[l].save('layer_'+l+'_noise.png')
