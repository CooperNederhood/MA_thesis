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

# Import other module
import sys 
sys.path.append('../')


def concat_horizontal(pic_list, border=0):
    '''
    Given a list of pics, concatenates them side-by-side
    '''

    num_imgs = len(pic_list)

    pic_mode = pic_list[0].mode 

    widths, heights = zip(*(i.size for i in pic_list))
    
    total_width = sum(widths) + (num_imgs-1)*border
    height = heights[0]

    new_img = Image.new(pic_mode, (total_width,height), color='white')

    x_offset = 0
    for pic in pic_list:
        w, h = pic.size
        assert h == height 
        new_img.paste(pic, (x_offset,0))
        x_offset += (w + border)

    return new_img 

    
def concat_vertical(pic_list):
    '''
    Given a list of pics, concatenates them verticlaly
    '''

    pic_mode = pic_list[0].mode 

    widths, heights = zip(*(i.size for i in pic_list))
    
    width = widths[0]
    total_height = sum(heights)

    new_img = Image.new(pic_mode, (width,total_height))

    y_offset = 0
    for pic in pic_list:
        w, h = pic.size
        assert w == width 
        new_img.paste(pic, (0,y))
        y_offset += h 

    return new_img 

def join_as_grid(pic_list, grid_size):
    '''
    Given a list of images, join them in a grid fashion
    with grid dimension of grid_size. Pics should be of 
    a consistent size and will fill with empty if grid_size
    is too big for the current pic_list

    Inputs:
        grid_size (tuple)

    '''

    x, y = pic_list[0].size 
    mode = pic_list[0].mode 

    out_img = Image.new("RGB", (x*grid_size[0], y*grid_size[1]), (255,255,255) )
    out_img = out_img.convert(mode)

    for pic_num, pic in enumerate(pic_list):
        x_offset = pic_num % grid_size[0]
        y_offset = pic_num // grid_size[0]

        out_img.paste(pic, (x*x_offset, y*y_offset))
        #out_img.paste(pic, (y*y_offset, x*x_offset))
        #print("Putting pic_num={} at grid_coords=({},{})".format(pic_num, x_offset, y_offset))

        #out_img.save(os.path.join(path2, "tiled", "grid{}.png".format(pic_num)))
    return out_img 


# path = "../segmentation/d_rgb_256_small/inter_layers/Layer_0/"
# path2 = "../segmentation/d_rgb_256_small/inter_layers/"

# file_list = ["ft_map{}.png".format(x) for x in range(32)]
# img_list = []
# for f in file_list:
#     img = Image.open(os.path.join(path, f))
#     img.thumbnail((256, 256))
#     img_list.append(img)

# grid0 = join_as_grid(img_list, (8,4) )
# #grid0.save(os.path.join(path2, "tiled", "grid.png"))
