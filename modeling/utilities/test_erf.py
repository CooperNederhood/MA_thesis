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
import matplotlib.pyplot as plt 

from PIL import Image
#import Unet_rand_rgb_model as model 

def make_alpha(under_img, mask):

	black_img = Image.new('RGB', under_img.size)
	mask = mask.convert("L")

	compos = Image.composite(under_img, black_img, mask)

	return compos 


def RF_once(net, img):
    '''
    Given an input image and a network, calculates the 
    effective receptive field from the final layer
    '''

    img.requires_grad_(True)
    img_size = img.shape[-1]

    out = net(img)
    out_size = out.shape[-1]
    grad_input = torch.zeros_like(out)
    grad_input[0,0,int(out_size/2), int(out_size/2)] = 1

    out.backward(grad_input)

    g = img.grad.squeeze(0)

    return g 


def test_RF(net, img_iters, img_size, channels):

    # Generate tensor to store output
    output = torch.zeros(img_iters, channels, img_size, img_size)
    
    # Initialize to uniform weights
    net.apply(init_weights)

    i = 0

    for _ in range(img_iters):
        img = torch.randn(1, channels, img_size, img_size)
        
        g = RF_once(net, img)

        output[i,:,:,:] = g 
        net.zero_grad()
        i += 1

    print("Through {} iterations".format(i))
    return output

def output_ERF_analysis(net, img_iters, img_size, 
                        channels, save_to_path, model_name="", uniform_weights=True):
    '''
    Calls test_rf, then processes and saves out the output
    ''' 

    if uniform_weights:
        net.apply(init_weights)

    test_output = test_RF(net, img_iters, img_size, channels)    

    mean_output = test_output.mean(dim=0).squeeze(0)

    if len(mean_output.shape) == 3:
        print("Working on 3D VGG output")
        mean_output = mean_output.permute(1,2,0)

    print(mean_output.shape)

    #mean_output.squeeze(0)

    # Use PIL rather than pyplot
    full_erf_array = mean_output.numpy()
    full_erf_img = Image.fromarray(full_erf_array).convert('RGB')
    full_erf_img.save(os.path.join(save_to_path, "new_ERF_full_{}.png".format(model_name)))

    top_left, bot_right = get_bounding_box(mean_output, zero_floor=1e-8)
    theor_rf = mean_output[top_left[0]:bot_right[0], top_left[1]:bot_right[1]]
    theor_rf_array = theor_rf.numpy()
    theor_rf_array = Image.fromarray(theor_rf_array).convert('RGB')
    theor_rf_array.save(os.path.join(save_to_path, "new_ERF_zoom_{}.png".format(model_name)))
    
    # plt.clf()
    # x_plot = plt.imshow(mean_output, cmap="binary_r", interpolation='nearest')
    # plt.axis('off')
    # plt.savefig(os.path.join(save_to_path, "ERF_full_{}.png".format(model_name)), bbox_inches='tight')
    
    # plt.clf()
    # top_left, bot_right = get_bounding_box(mean_output, zero_floor=1e-8)
    # theor_rf = mean_output[top_left[0]:bot_right[0], top_left[1]:bot_right[1]]
    # x_plot = plt.imshow(theor_rf, cmap="binary_r", interpolation='nearest')
    # plt.axis('off')
    # plt.savefig(os.path.join(save_to_path, "ERF_zoom_{}.png".format(model_name)), bbox_inches='tight')




def init_weights(m):
    
    if type(m) == nn.Conv2d:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)


def get_bounding_box(img, zero_floor=1e-08):
    
    img_size = img.shape[-1]

    mean_zerod = img
    mean_zerod[mean_zerod<zero_floor] = 0
    nonzero_coords = torch.nonzero(mean_zerod)

    min_x = img_size
    max_x = 0
    min_y = img_size 
    max_y = 0

    for coord in nonzero_coords:
        if coord[0] < min_x:
            min_x = coord[0]
        if coord[0] > max_x:
            max_x = coord[0]

        if coord[1] < min_y:
            min_y = coord[1]
        if coord[1] > max_y:
            max_y = coord[1] 

    return [(min_x.item(), min_y.item()), (max_x.item(), max_y.item())]   

# to_pil = transforms.ToPILImage()

# # Test uniform initialization
# #net = RFnet()
# channels = 1
# img_size = 128

# # Test receptive field for U-Net
# net = model.Unet(channels, img_size, layer2_ft=64)
# net.apply(init_weights)

# test_output = test_RF(net, 1, 6, img_size, channels)    

# mean_output = test_output.mean(dim=0).squeeze(0)
# x_plot = plt.imshow(mean_output, cmap="binary_r", interpolation='nearest')
# plt.axis('off')
# plt.savefig('test.png', bbox_inches='tight')
# plt.clf()

# top_left, bot_right = get_bounding_box(mean_output, zero_floor=1e-8)
# theor_rf = mean_output[top_left[0]:bot_right[0], top_left[1]:bot_right[1]]
# x_plot = plt.imshow(theor_rf, cmap="binary_r", interpolation='nearest')
# plt.axis('off')
# plt.savefig('ERF_Unet.png', bbox_inches='tight')

# Test receptive field for Dilated Net
# dil_net = dilation_model.DilationNet_v1(channels, img_size)
# dil_net.apply(init_weights)

# test_output = test_RF(dil_net, 1, 6, img_size, channels)    

# mean_output = test_output.mean(dim=0).squeeze(0)
# x_plot = plt.imshow(mean_output, cmap="binary_r", interpolation='nearest')
# plt.axis('off')
# plt.savefig('Dilation_test.png', bbox_inches='tight')
# plt.clf()

# top_left, bot_right = get_bounding_box(mean_output, zero_floor=1e-8)
# theor_rf = mean_output[top_left[0]:bot_right[0], top_left[1]:bot_right[1]]
# x_plot = plt.imshow(theor_rf, cmap="binary_r", interpolation='nearest')
# plt.axis('off')
# plt.savefig('ERF_Dilation.png', bbox_inches='tight')

