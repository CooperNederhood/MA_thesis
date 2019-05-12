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

from sklearn.metrics import confusion_matrix
import pickle 

# Import other module
import sys 
sys.path.append('../')
#from utilities import cnn_utils, transform_utils

plt.style.use('ggplot')

def do_ROC_curve(net, val_dset_loader, thresholds, device):
    '''
    Do an ROC curve evaluation for the given model
    '''

    net.eval()

    total_pixels = 0
    total_slum = 0
    running_conf_matrix = np.zeros((2, 2))

    conf_matrices = {}
    for t in thresholds:
        conf_matrices[t] = running_conf_matrix.copy()

    with torch.no_grad():
        for i, data in enumerate(val_dset_loader):

            images, target = data
            img_size = target.shape[-1]
            #assert img_size == 128

            target = torch.tensor(target, dtype=torch.float32, device=device)
            images = images.to(device)

            batch_size = target.shape[0]
            total_pixels += batch_size*img_size*img_size
            total_slum += target.sum().item()
            
            # Forward pass -- reshape output to be like the target (has extra 1D dimension)
            output = net(images)
            output = output.view(-1)
            target = target.view(-1)

            # Do predictions based on each of our thresholds
            for t in thresholds:
                preds = output > t 
                conf_matrices[t] += confusion_matrix(preds, target)

    print("There are {} pixels and {} are slum\n".format(total_pixels, total_slum/total_pixels))

    for t in thresholds:
        print("Conf mat for threshold={}".format(t))
        print(conf_matrices[t]/total_pixels)
        print()

    # Save 
    with open("confusion_matrices.p", 'w') as f:

        pickle.dump(conf_matrices, f)

    print("\n\nDONE!")




def do_in_sample_tests(net, THESIS_ROOT, tile=False, dtype="RGB"):
    '''
    Evaluate the model over the in-sample images, both slum 
    and non-slum
    '''
    assert dtype in ["RGB", "Four_band"]
    IN_SAMPLE_ROOT = os.path.join(THESIS_ROOT, "data", "descartes", dtype, "min_cloud")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Do in-sample evaluations on slum and not_slums 
    if not os.path.isdir("in_sample_test"):
        os.mkdir("in_sample_test")
    for t in ["slums", "not_slums"]:
        if not os.path.isdir(os.path.join("in_sample_test", t)):
            os.mkdir(os.path.join("in_sample_test", t))

        for s in ["pct", "binary"]:
            if not os.path.isdir(os.path.join("in_sample_test", t, s)):
                os.mkdir(os.path.join("in_sample_test", t, s))

        files = os.listdir(os.path.join(IN_SAMPLE_ROOT, t))
        for f in files:
            print("Processing file: ", f)

            if dtype == "RGB":
                img = Image.open(os.path.join(IN_SAMPLE_ROOT, t, f))
                array = transforms.ToTensor()(img)
            else:
                assert dtype == "Four_band"
                img = np.load(os.path.join(IN_SAMPLE_ROOT, t, f))
                array = torch.from_numpy(img).type(torch.float32)
                f = f.replace(".npy", ".png")

            array = array.to(device)

            pred_img, pred_cat = make_pred_map_segmentation(array, net, pic_size=256, tile=tile)
            print("Saving at: ", os.path.join("in_sample_test", t, "pct"))   
            pred_img.save(os.path.join("in_sample_test", t, "pct", f))    
            pred_cat.save(os.path.join("in_sample_test", t, "binary", f))    

def do_out_sample_tests(net, THESIS_ROOT, tile=False, dtype="RGB"):
    '''
    Evaluate the model over the out-of-sample images
    '''
    assert dtype in ["RGB", "Four_band"]
    IN_SAMPLE_ROOT = os.path.join(THESIS_ROOT, "data", "descartes", dtype, "min_cloud")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Do out-of-sample evaluation
    if not os.path.isdir("out_sample_test"):
        os.mkdir("out_sample_test")

    t = "aoi"

    if not os.path.isdir(os.path.join("out_sample_test", t)):
        os.mkdir(os.path.join("out_sample_test", t))

    for s in ["pct", "binary"]:
        if not os.path.isdir(os.path.join("out_sample_test", t, s)):
            os.mkdir(os.path.join("out_sample_test", t, s))

    files = os.listdir(os.path.join(IN_SAMPLE_ROOT, t))
    for f in files:
        print("Processing file: ", f)

        if dtype == "RGB":
            img = Image.open(os.path.join(IN_SAMPLE_ROOT, t, f))
            array = transforms.ToTensor()(img)
        else:
            assert dtype == "Four_band"
            img = np.load(os.path.join(IN_SAMPLE_ROOT, t, f))
            array = torch.from_numpy(img).type(torch.float32)
            f = f.replace(".npy", ".png")

        array = array.to(device)

        pred_img, pred_cat = make_pred_map_segmentation(array, net, pic_size=256, tile=tile)
        print("Saving at: ", os.path.join("out_sample_test", t, "pct"))   
        pred_img.save(os.path.join("out_sample_test", t, "pct", f))    
        pred_cat.save(os.path.join("out_sample_test", t, "binary", f))    


def inter_over_union(pred, target):

    log_or = torch.max(pred, target)
    log_and = pred*target 

    if log_or.sum() == 0:
        rv = 1
    else:
        rv = log_and.sum().item() / log_or.sum().item()

    return rv


def load_weights(raw_net, model_name, is_gpu):
    '''
    The raw_net is a type of deep neural net and 
    we fetch the state_dict contained in the appropriate
    subfolder and update the weights of raw_net accordingly

    Returns: raw_net with the trained_weights
    '''

    location = 'cpu' if not is_gpu else 'gpu'
    raw_net.load_state_dict(torch.load(os.path.join(model_name, 
        "model.pt"), map_location=location))

    return raw_net 


def plot_training_dict(model_name, dict_type="epoch"):
    assert dict_type in ["batch", "epoch"]

    if dict_type == "epoch":
        l = "Epoch avg"
        history_file = "training_hist.json"
    else:
        l = "Batch"
        history_file = "batch_hist.json"

    json_data = open(os.path.join(model_name, history_file)).read()
    training_dict = json.loads(json_data)

    if dict_type == "batch":
        smooth = 10
        training_dict['val']['loss'] = np.convolve(training_dict['val']['loss'], np.ones(smooth)/smooth)
        training_dict['val']['acc'] = np.convolve(training_dict['val']['acc'], np.ones(smooth)/smooth)
        training_dict['val']['IoU'] = np.convolve(training_dict['val']['IoU'], np.ones(smooth)/smooth)

    epoch_list = range(1, len(training_dict['val']['loss'])+1)

    plt.clf()
    if dict_type == "epoch":
        plt.plot(epoch_list, training_dict['train']['loss'], label="Training phase")
    plt.plot(epoch_list, training_dict['val']['loss'], label="Validation phase")
    plt.legend()
    plt.title("{} mean-squared-error".format(l))
    plt.ylim(0,1)
    plt.savefig('{} Loss.png'.format(dict_type))

    plt.clf()
    if dict_type == "epoch":
        plt.plot(epoch_list, training_dict['train']['acc'], label="Training phase")
    plt.plot(epoch_list, training_dict['val']['acc'], label="Validation phase")
    plt.legend()
    plt.title("{} pixel-level prediction accuracy".format(l))
    plt.ylim(0,1)
    plt.savefig('{} Acc.png'.format(dict_type))
        
    plt.clf()
    if dict_type == "epoch":
        plt.plot(epoch_list, training_dict['train']['IoU'], label="Training phase")
    plt.plot(epoch_list, training_dict['val']['IoU'], label="Validation phase")
    plt.legend()
    plt.title("{} intersection-over-union score".format(l))
    plt.ylim(0,1)
    plt.savefig('{} IoU.png'.format(dict_type))

    if dict_type == "epoch":
        val_loss = np.min(training_dict['val']['loss'])
        min_epoch = np.argmin(training_dict['val']['loss'])

        val_IoU = training_dict['val']['IoU'][min_epoch]
        val_acc = training_dict['val']['acc'][min_epoch]
        mean_train_time = round(np.mean(training_dict['train']['time'])/60)

        with open("perf_summary.csv", 'w') as f:
            f.write(", Model Name\n")

            f.write("Min loss, ")
            f.write(str(val_loss))
            f.write("\n")

            f.write("IoU, ")
            f.write(str(val_IoU))
            f.write("\n")

            f.write("Acc., ")
            f.write(str(val_acc))
            f.write("\n")

            f.write("Avg train time, ")
            f.write(str(mean_train_time))
            f.write("\n")

def make_pred_map_classification(test_img, net, pic_size, step_size=None):
    '''
    Inputs:
        test_img: (np array) of image we're evaluating on
        net: (pytorch network)
        pic_size: (int) dimensions of thumnails to extract and pass into net
        step_size: (int) step size between pics
    '''

    if step_size is None:
        step_size = pic_size

    # Output prediction map starts with array of zeros
    pred_map = np.zeros(test_img.shape[1:3])
    pred_map_cat = np.zeros(test_img.shape[1:3])
    print("Prediction map shape: ", pred_map.shape)

    i = pic_size
    j = pic_size

    _, max_i, max_j = test_img.shape
    pic_count = 0

    # Putting model in eval mode sets layers like dropout to prediction
    # Wrapping in no_grad() lets us not record gradient
    net.eval()
    with torch.no_grad():
        while i < max_i:
            while j < max_j:
                sub_pic = test_img[:, i-pic_size:i, j-pic_size:j] 
                sub_pic = sub_pic.unsqueeze(0)
                
                pred = net(sub_pic).item()
                assert pred < 1 and pred > 0
                pred_cat = np.round(pred).item()
                assert pred_cat in {0,1}

                #print("Assigning pred val = {} to cat = {}".format(pred, pred_cat))

                # Update the prediction matrix accordingly
                sub_mat_cat = np.full((pic_size,pic_size), pred_cat)
                sub_mat = np.full((pic_size, pic_size), pred)

                pred_map_cat[i-pic_size:i, j-pic_size:j] += sub_mat_cat
                pred_map[i-pic_size:i, j-pic_size:j] += sub_mat

                pic_count += 1
                j += step_size
            j = pic_size
            i += step_size

    return pred_map_cat, pred_map, pic_count

def calc_buffer_needed(test_img, pic_size, step_size):
    '''
    To make predictions on test_img, given a network that accepts input
    of size=pic_size and a step_size>1, there will almost certainly NOT
    be a perfect alignment, resulting in a prediction image smaller than
    the input test_img. Thus, calculate a buffer needed to add to the test_img
    resulting in a perfect fit
    '''

    # Check that size gives a 4 dimensional size
    assert len(test_img.size()) == 4

    buffer_dict = {}
    for i, img_size in enumerate(test_img.size()[2:]):
        remainder = (img_size - pic_size) % step_size
        added_buffer = step_size - remainder

        assert (img_size + added_buffer - pic_size) % step_size == 0

        # Need to split to two sides
        if added_buffer % 2 == 1:
            added_buffer0 = (added_buffer-1)/2
            added_buffer1 = (added_buffer-1)/2 + 1
        else:
            added_buffer0 = added_buffer/2
            added_buffer1 = added_buffer/2

        if i==0:
            buffer_dict['L'] = int(added_buffer0)
            buffer_dict['R'] = int(added_buffer1)

        else:
            buffer_dict['T'] = int(added_buffer0)
            buffer_dict['B'] = int(added_buffer1)

    return buffer_dict  




def make_pred_map_segmentation(test_img, net, pic_size, step_size=None, tile=False):
    '''
    Inputs:
        test_img: (np array) of image we're evaluating on
        net: (pytorch network)
        pic_size: (int) dimensions of thumnails to extract and pass into net
        step_size: (int) step size between pics
        tile: (bool) of whether to break image into tiles
    '''

    if step_size is None:
        step_size = pic_size

    if tile:
        pass 
        # NEW: add reflection padding to have no cut off edges
        # test_img.unsqueeze_(0)
        # print("Original dim = ", test_img.shape)
        # pad = calc_buffer_needed(test_img, pic_size, step_size)
        # padding_layer = nn.ReplicationPad2d((pad['T'], pad['B'], pad['L'], pad['R']))
        # test_img = padding_layer(test_img)
        # print("Padded dim = ", test_img.shape)
        # test_img.squeeze_(0)

    # Output prediction map starts with array of zeros
    pred_map = torch.zeros(test_img.shape[1:3])
    pred_map_cat = torch.zeros(test_img.shape[1:3])
    print("Prediction map shape: ", pred_map.shape)

    i = pic_size
    j = pic_size

    _, max_i, max_j = test_img.shape
    pic_count = 0

    if max_i * max_j > 640000:
        tile = True 

    # Putting model in eval mode sets layers like dropout to prediction
    # Wrapping in no_grad() lets us not record gradient
    if tile:

        net.eval()
        with torch.no_grad():
            while i < max_i:
                while j < max_j:
                    sub_pic = test_img[:, i-pic_size:i, j-pic_size:j] 
                    sub_pic = sub_pic.unsqueeze(0)
                    
                    # Should get 4D mask back, squeeze the batch and channel dimensions
                    pred_mask = net(sub_pic)
                    assert pred_mask.dim() == 4

                    assert pred_mask.min().item() > 0 and pred_mask.max().item() < 1
                    pred_mask = pred_mask.squeeze(dim=0).squeeze(dim=0)
                    assert pred_mask.dim() == 2

                    # Make classification mask
                    pred_mask_cat = torch.round(pred_mask)
                    assert pred_mask_cat.shape == pred_mask.shape 

                    # Update the prediction matrix accordingly
                    print("\tAt {},{} of max {},{}".format(i,j,max_i, max_j))
                    
                    pred_map[i-pic_size:i, j-pic_size:j] += pred_mask

                    pic_count += 1
                    j += step_size
                j = pic_size
                i += step_size
            #pred_map = pred_map[pad['R']:-pad['L'], pad['B']:-pad['T']]
            pred_map_cat = torch.round(pred_map)

    else:
        net.eval()
        with torch.no_grad():
            test_img.unsqueeze_(0)
            assert test_img.dim() == 4
            
            pred_mask = net(test_img)
            assert pred_mask.dim() == 4
            assert pred_mask.min().item() > 0 and pred_mask.max().item() < 1

            pred_mask.squeeze_(0).squeeze_(0)
            assert pred_mask.dim() == 2

            # Make classification map
            pred_mask_cat = torch.round(pred_mask)
            assert pred_mask_cat.shape == pred_mask.shape 

    #pred_img = Image.fromarray(pred_mask.cpu().numpy()).convert("RGB")
    #pred_img_cat = Image.fromarray(pred_mask_cat.cpu().numpy()).convert("RGB")
    pred_img = Image.fromarray(pred_map.cpu().numpy()*255).convert("RGB")
    pred_img_cat = Image.fromarray(pred_map_cat.cpu().numpy()*255).convert("RGB")
    print("Final shape = ", pred_map.shape)
    return pred_img, pred_img_cat

def plot_encoder_pass_layers(net, image, thumbnail_size=None):
    '''
    Inputs:
        net: trained FNN with encoder_pass method
        image: batch-style image, shape = (1,BANDS,SIZE_X,SIZE_Y)
        thumbnail_size: tuple denoting output size, ex. (256, 256). Maintains 
                            the proper x,y proportions subject to size constraint
    '''

    net.eval()
    with torch.no_grad():
        inter_output = net.encoder_pass(image)

        rv_dict = {}

        for layer_num, layer in enumerate(inter_output):
            ft_count = layer.shape[1]

            ft_map_list = []
            for ft_num in range(ft_count):
                ft_map = layer[0,ft_num,:,:].numpy()
                ft_map = Image.fromarray(ft_map*255, mode="L")
                if thumbnail_size is not None:
                    ft_map.thumbnail(thumbnail_size)
                ft_map_list.append(ft_map)

            rv_dict['Layer_{}'.format(layer_num)] = ft_map_list

    return rv_dict


if __name__ == "__main__":
    #########################################################################################
    # MODEL SPECIFIC:
    #from modeling.segmentation import base_segmentation_model
    #from modeling.classification import base_classification_model

    CHANNELS = 3 
    img_size = 128
    step_size = img_size
    model_name = "seg_base"
    #model_name = "class_base"
    #########################################################################################
    THESIS_ROOT = "../../"
    DATA_ROOT = os.path.join(THESIS_ROOT, "data")

    #IN_SAMPLE_ROOT = "test_images/in_sample"
    #OUT_SAMPLE_ROOT = os.path.join(DATA_DIR, "aoi")

    ge_path = os.path.join(DATA_ROOT, "google_earth", "zoom_18", "slums")

    #f = "ona_id54_image.png"
    f = "ona_id14_image.png"
    img = Image.open(os.path.join(ge_path, f))
    
    buffer_dict = calc_buffer_needed(img, img_size, step_size)
    

