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

class classifyNet(nn.Module):

    def __init__(self, img_size):
        super(classifyNet, self).__init__()

        self.my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        #self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        # To correct initialize the fc1 layer, we need to know the output size
        #        of the last convolutional layer
        self.conv_end_ft_count = self._determine_ft_count(img_size)

        self.fc1 = nn.Linear(self.conv_end_ft_count, 1)
        #self.fc2 = nn.Linear(400, 200)
        #self.fc3 = nn.Linear(200, 1)


    def _determine_ft_count(self, img_size):
        '''
        Calculate the output ft count of what comes out of 
        the end of our convolutional base
        '''

        example = torch.randn(1, 3, img_size, img_size)
        conv_example = self.forward_convolutional(example)
        return self.num_flat_features(conv_example)


    def forward_convolutional(self, x):
        '''
        Does the forward pass of our convolutional base
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        #x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))

        return x 

    def forward_fully_connected(self, x):
        '''
        Does the forward pass of our fully-connected 
        part of the model
        '''     
        x = F.sigmoid(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.sigmoid(self.fc3(x))

        return x    


    def forward(self, x):
        x = self.forward_convolutional(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.forward_fully_connected(x)

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

    train_dataloader = utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

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

img_size = 128
x1 = torch.randn(1, 3, img_size, img_size)

# Define network
net = classifyNet(img_size)

# Define dataloaders
data_root =  "/home/cooper/Documents/MA_thesis/data/training_data/classification/size_128"
dataloader_dict = build_dataloader_dict(data_root, 16)

# Define criterion function
criterion = nn.BCELoss()

# Define optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
optimizer = optim.Adam(net.parameters())

def train_model(model, num_epochs, dataloader_dict, criterion, optimizer, verbose=False):
    print("Starting training loop...\n")
    print("Model's device = {}".format(model.device))

    device = model.my_device

    # Save a copy of the weights, before we start training
    begin_model_wts = copy.deepcopy(model.state_dict())

    # Initialize loss dict to record training, figures are per epoch
    epoch_loss_dict = {'train': {'acc': [], 'loss':[]}, 'val': {'acc': [], 'loss':[]}}

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
                
                #print("avg output={}".format(output.sum()/output.shape[0]))

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
                    # p0t0_c = df_c[(df_c.preds==0)&(df_c.target==0)].output.item()
                    # p1t1_c = df_c[(df_c.preds==1)&(df_c.target==1)].output.item()
                    # p0t1_c = df_c[(df_c.preds==0)&(df_c.target==1)].output.item()
                    # p1t0_c = df_c[(df_c.preds==1)&(df_c.target==0)].output.item()

                    # p0t0_m = df_m[(df_m.preds==0)&(df_m.target==0)].output.item()
                    # p1t1_m = df_m[(df_m.preds==1)&(df_m.target==1)].output.item()
                    # p0t1_m = df_m[(df_m.preds==0)&(df_m.target==1)].output.item()
                    # p1t0_m = df_m[(df_m.preds==1)&(df_m.target==0)].output.item()

                    # verbose_str = "Counts: p0t0={};p1t1={};p0t1={};p1t0={}\tMeans:p0t0={};p1t1={};p0t1={};p1t0={}".format(
                    #     p0t0_c, p1t1_c, p0t1_c, p1t0_c, p0t0_m, p1t1_m, p0t1_m, p1t0_m)


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

            # Add to our epoch_loss_dict
            epoch_loss_dict[phase]['acc'].append(epoch_acc)
            epoch_loss_dict[phase]['loss'].append(epoch_loss)

            print("\nPHASE={} EPOCH={} TIME={} LOSS={} ACC={}\n".format(phase, 
                epoch, time.time()-begin_epoch_time, epoch_loss, epoch_acc))
            

    return model, begin_model_wts, epoch_loss_dict 


#trained_model, begin_wts, epoch_loss_dict = train_model(net, 1, dataloader_dict, criterion, optimizer, verbose=True)



# output = net(x1)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()

