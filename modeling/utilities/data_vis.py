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


def make_side_by_side(left_pic, right_pic, output_filename, output_path):
	'''
	Given two file paths, makes a side-by-side image
	and saves to output_path
	'''

	left_img = Image.open(left_pic)
	right_img = Image.open(right_pic)

	assert left_img.shape == right_img.shape 

	

