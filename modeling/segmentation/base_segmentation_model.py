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

MODEL_NAME = "seg_base"
MODEL_DETAILS = '''Base segmentation model, using truncated U-Net structure. Adding random hor/ver flips to training images
EPOCH_COUNT = 15; BATCH_SIZE=16; img_size=128
'''

# Import our utilities module
sys.path.append('../')
from utilities import cnn_utils, transform_utils

EPOCH_COUNT = 15
BATCH_SIZE = 16
CHANNELS = 3
img_size = 128


# Root directory for dataset
root = '../../data/training_data'
data_root = os.path.join(root, "segmentation/size_128")

SAVE_ROOT = root 

# Check whether GPU is enabled
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class segNet(nn.Module):

    def __init__(self, img_size):
        super(segNet, self).__init__()

        # Define layers for encoding
        self.my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        # self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        # self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        # self.conv5a = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        # self.conv5b = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)

        # Define layers for decoding
        self.tconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv1a = nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=3, padding=1)
        self.up_conv1b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.tconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv2a = nn.Conv2d(in_channels=64*2, out_channels=64, kernel_size=3, padding=1)
        self.up_conv2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        #self.tconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=2)

        # Layer for classification
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)


    def encoder_pass(self, x):

        # Encode 1
        x = F.relu(self.conv1a(x))
        x1 = F.relu(self.conv1b(x))

        # Encode 2
        x2 = F.max_pool2d(F.relu(self.conv2a(x1)), 2)
        x2 = F.relu(self.conv2b(x2))

        # Encode 3 -- no downsampling here
        x3 = F.max_pool2d(F.relu(self.conv3a(x2)), 2)
        #x3 = F.max_pool2d(F.relu(self.conv3b(x3)), 2)
        x3 = F.relu(self.conv3b(x3))

        # # Encode 4
        # x = F.relu(self.conv4a(x))
        # x = F.max_pool2d(F.relu(self.conv4b(x)), 2)

        # # Encode 5
        # x = F.relu(self.conv5a(x))
        # x = F.relu(self.conv5b(x))

        return (x1, x2, x3)

    def decoder_pass(self, x_list):

        x1, x2, x3 = x_list

        # Decode 1
        x4 = self.tconv1(x3)
        x4 = F.relu(self.up_conv1a(torch.cat((x2, x4),1)))
        x4 = F.relu(self.up_conv1b(x4))

        # Decode 2 
        x5 = self.tconv2(x4)
        x5 = F.relu(self.up_conv2a(torch.cat((x1, x5),1)))
        x5 = F.relu(self.up_conv2b(x5))

        return (x4, x5)

    def forward(self, x):

        x1, x2, x3 = self.encoder_pass(x)
        x4, x5 = self.decoder_pass((x1,x2,x3))

        segmented_image = torch.sigmoid(self.final_conv(x5))

        assert segmented_image.shape[-2:] == x.shape[-2:] 

        return segmented_image


class SegmentationDataset(utils.data.Dataset):
    '''
    Class defines segmentation (ie image with mask) dataset
    '''

    def __init__(self, root_path, list_common_trans=None, list_img_trans=None):
        '''NOTE: transforms should be TENSOR transforms, and the PIL->Tensor transform is 
        already included
        '''
        self.image_root = os.path.join(root_path, "image")
        self.mask_root = os.path.join(root_path, "mask")
        self.files = self._build_file_list(self.image_root, self.mask_root)
        self.list_common_trans = list_common_trans
        self.list_img_trans = list_img_trans

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        f = self.files[index]

        # Load image and mask, convert to Tensors
        image = transforms.ToTensor()(Image.open(os.path.join(self.image_root, f)))
        mask = transforms.ToTensor()(Image.open(os.path.join(self.mask_root, f)))

        # Join the image and mask, so random transforms can be applied consistently
        if self.list_common_trans is not None:
            image_channels = image.shape[0]
            mask_channels = mask.shape[0]

            # Stack along the channel dimension
            image_mask_combined = torch.cat((image, mask), 0)

            # Apply common tranforms
            for t in self.list_common_trans:
                image_mask_combined = t(image_mask_combined)

            image_mask_combined = torch.Tensor(image_mask_combined)
            assert image_mask_combined.shape[1:] == image.shape[1:]
            
            # Split back out
            image = image_mask_combined[0:image_channels,:,:]
            mask = image_mask_combined[image_channels:,:,:]

        # Apply the image only transforms
        if self.list_img_trans is not None:
            image = transforms.Compose(self.list_img_trans)(image)

        mask = convert_img_to_2D_mask(mask)

        assert image.shape[1:] == mask.shape

        return image, mask 


    def _build_file_list(self, image_root, mask_root):
        '''
        Helper function to make sure files are consistent 
        across masks and images
        '''

        image_files = os.listdir(image_root)
        mask_files = os.listdir(mask_root)
        assert image_files == mask_files

        return image_files 


def convert_img_to_2D_mask(tensor_img):
    '''
    Helper function which converts a 3D mask stored as a Tensor
    to a 2D binary Tensor mask

    Inputs:
        tensor_img: 3D Tensor
    Returns:
        tensor_mask: 2D binary mask
    '''

    # Check that masks are consistent across the channel dimension
    assert tensor_img.std(dim=0).max().item() == 0  # Channels are redundant
    assert tensor_img.max() in (0,1)                # Vals are always either 0 or 1
    assert tensor_img.min() in (0,1)                # Vals are always either 0 or 1

    return tensor_img[0,:,:]

def get_batch(dataloader_dict, d_type):
    '''
    Just a helper function, fetches an example from the dataloader
    specified to allow for easier debugging
    '''

    for data in dataloader_dict[d_type]:
        break

    return data


def train_segmentation(model, num_epochs, dataloader_dict, criterion, optimizer, verbose=False):
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
            for i, data in enumerate(dataloader_dict[phase]):

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
                
                #print("avg output={}".format(output.sum()/output.shape[0]))

                # Calculate loss
                error = criterion(output, target)
                # correct = preds==target
                # incorrect = preds!=target
                # correct_count = torch.sum(correct)

                # Make detailed output if verbose
                verbose_str = ""
                if verbose:
                    pass 

                # Training steps
                if phase == 'train':
                    # Backpropogate the error
                    error.backward()
                    # Take optimizer step
                    optimizer.step()

                # The error is divided by the batch size, so reverse this
                running_loss += error.item() * batch_size
                #running_corrects += correct_count 

                #print('%s - [%d/%d][%d/%d]\tError: %.4f\t' % 
                #    (phase, epoch, num_epochs, i, len(dataloader_dict[phase]), error.item()))
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

            print("PHASE={} EPOCH={} TIME={} LOSS={} ACC={}".format(phase, 
                epoch, t, epoch_loss, epoch_acc))


    return model, best_model_wts, epoch_loss_dict 

# Define trasnforms
common_transforms = [transform_utils.RandomHorizontalFlip(0.5), 
                     transform_utils.RandomVerticalFlip(0.5)]
#img_transforms = [transforms.ColorJitter()]

# Define network
net = segNet(img_size)

# Define dataloaders
train_root = os.path.join(data_root, "train")
val_root = os.path.join(data_root, "val")

train_dset = SegmentationDataset(train_root, list_common_trans=common_transforms,
                                 list_img_trans=None)
val_dset = SegmentationDataset(val_root)

train_dset_loader = utils.data.DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
val_dset_loader = utils.data.DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=True)

dset_loader_dict = {'train':train_dset_loader, 'val':val_dset_loader}

criterion_loss = nn.BCELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
optimizer = optim.Adam(net.parameters())

# for i, (x,y) in enumerate(dset_loader_dict['train']):
#     print("{} / {}".format(i, len(dset_loader_dict['train'])))


trained_net, best_model_wts, training_hist = train_segmentation(net, EPOCH_COUNT, dset_loader_dict, criterion_loss, optimizer)

cnn_utils.save_model(net, MODEL_NAME, best_model_wts, training_hist, MODEL_DETAILS, SAVE_ROOT)


# Save out

# f = 'seg_model.pt'
# training_f = 'seg_training_hist.json'

# torch.save(best_wts, os.path.join(root, f))
# with open(os.path.join(root,training_f), 'w') as fp:
#     json.dump(epoch_loss_dict, fp)

