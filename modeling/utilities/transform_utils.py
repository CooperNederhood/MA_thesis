import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
from torchvision import datasets, models, transforms
from torch import utils, optim 

import numpy as np 

from PIL import Image 
import matplotlib.pyplot as plt 
import scipy.ndimage 


class RandomHorizontalFlip(object):
	'''
	Randomly rotate Tensor or Array
	'''

	def __init__(self, p, random_state=np.random):

		self.p = p 
		self.random_state = random_state


	def __call__(self, tensor_img):

		b = True if self.random_state.uniform(0,1) < self.p else False
		print("Flipping = ", b)

		if b:
			rv = torch.as_tensor(np.flip(tensor_img, axis=1))
			assert rv.shape == tensor_img.shape
			assert isinstance(rv, torch.Tensor)
			return rv
		else:
			return tensor_img 

class RandomVerticalFlip(object):
	'''
	Randomly rotate Tensor or Array
	'''

	def __init__(self, p, random_state=np.random):

		self.p = p 
		self.random_state = random_state


	def __call__(self, tensor_img):

		b = True if self.random_state.uniform(0,1) < self.p else False
		print("Flipping = ", b)

		if b:
			rv = torch.as_tensor(np.flip(tensor_img, axis=2))
			assert rv.shape == tensor_img.shape
			assert isinstance(rv, torch.Tensor)
			return rv
		else:
			return tensor_img 
