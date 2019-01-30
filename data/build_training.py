'''
https://www.jeremyjordan.me/semantic-segmentation/
We need to split the raw slum/non-slum images into smaller, standard pieces
that can actually go into a classifier. It's ambiguous which image size is best,
so build a function that can accept different image sizes. For reference, most ImageNet
applications are 256x256
'''

import numpy as np 
from PIL import Image 
import os 
import shutil

def split_image(image, image_id, zoom, output_size, step_size, class_type):
    '''
    Input:
        image: large np array image to partition into smaller images
        image_id: ID of large image
        zoom: zoom level of large image
        output_size: size of output images to generate
        step_size: how much should the frame step over after grabbing a sub-image
        class_type: either slum or non_slum

    Returns:
        Saves out images accordingly
    '''

    output_path = "training_data/zoom_{}/pic{}/{}/".format(zoom, output_size, class_type)

    i = output_size
    j = output_size

    max_i, max_j, _ = image.shape
    pic_num = 0

    while i < max_i:
        while j < max_j:
            sub_pic = image[i-output_size:i, j-output_size:j, :] 
            im = Image.fromarray(sub_pic)
            im.save("{}pic_{}_{}.png".format(output_path, image_id, pic_num))
            # print("{}-{},{}-{}".format(i-output_size, i, j-output_size, j))
            # print("subpic shape = ", sub_pic.shape)
            # print()
            pic_num += 1
            j += step_size
        j = output_size
        i += step_size

# zoom = 18
# class_type = "slums"
# source_path = "google_earth/zoom_{}/{}/".format(zoom, class_type)
# file = "ona_id47_image.png"

# im = Image.open(source_path+file)
# im_array = np.array(im)

# split_image(im_array, "47", "18", 256, 256, class_type)
# split_image(im_array, "47", "18", 128, 128, class_type)

def determine_image_class(mask, threshold):
    '''
    Given a pic's mask and a threshold, returns whether that %
    of pixels that are from a slum is beyond the threshold
    '''

    assert mask.ndim == 3
    m = mask / mask.max()
    m = m.max(axis=2,  keepdims=True)

    pct_slum = m.mean()

    pic_class = "slum" if pct_slum > threshold else "nonslum"
    return pic_class




def build_training_data(zoom_level, pic_size, step_size, class_threshold, 
    mask_path, slum_image_path, nonslum_image_path):
    '''
    Given a desired picture output size, a threshold to determine what % of pixels
    need to be a slum for the image to be classified a slum, and paths to the raw images
    and their corresponding image masks - builds both segmentation and classification datasets

    Inputs:
        pic_size: (int) size out output training image size
        step_size: (int) how much to slide across when partitioning images
        class_threshold: (float) 0-1 determining threhold for classificaiton of image
        mask_path: (str) path to slum/non-slum image masks
        slum_image_path: (str) path to Ona slum images
        nonslum_image_path: (str) path to additional non-slum images we need
    '''

    output_path = "training_data"
    segmentation_output = os.path.join(os.path.join(output_path, "segmentation"), "size_{}".format(pic_size))
    classification_output = os.path.join(os.path.join(output_path, "classification"), "size_{}".format(pic_size))

    # First, process all the Ona slum images and their corresponding masks
    ona_images = os.listdir(mask_path)

    for img_id in ona_images:
        ona_img = np.array(Image.open(os.path.join(slum_image_path, img_id)))
        ona_mask = np.array(Image.open(os.path.join(mask_path, img_id)))

        assert ona_img.shape == ona_mask.shape

        i = pic_size
        j = pic_size
        max_i, max_j, _ = ona_img.shape

        # Loop over our current image and partition it into subimages
        pic_num = 0 
        while i < max_i:
            while j < max_j:
                sub_img = Image.fromarray(ona_img[i-pic_size:i, j-pic_size:j, :])
                sub_mask = Image.fromarray(ona_mask[i-pic_size:i, j-pic_size:j, :])

                output_file = img_id.replace(".png", "{}.png".format(pic_num))

                # Save out the sub-image and the sub-mask in the segmentation folder
                sub_img.save(os.path.join(os.path.join(segmentation_output, "image"), output_file))
                sub_mask.save(os.path.join(os.path.join(segmentation_output, "mask"), output_file))

                # Save out the sub-image in the classification folder
                pic_class = determine_image_class(ona_mask[i-pic_size:i, j-pic_size:j, :], class_threshold)

                print("Ona ID={}; pic_num={}; class={}".format(img_id, pic_num, pic_class))

                sub_img.save(os.path.join(os.path.join(classification_output, pic_class), output_file))

                pic_num += 1

                j += step_size
            j = pic_size
            i += step_size
            
    print("DONE WITH SLUM IMAGES\n\n")

    # Second, process all the non-slum images we added
    nonslum_images = os.listdir(nonslum_image_path)

    for img_id in nonslum_images:
        img = np.array(Image.open(os.path.join(nonslum_image_path, img_id)))
        mask = np.zeros(img.shape, dtype=np.uint8)

        img.dtype

        assert img.shape == mask.shape

        i = pic_size
        j = pic_size
        max_i, max_j, _ = img.shape 

        # Loop over the current non-slum image and partition
        pic_num = 0
        while i < max_i:
            while j < max_j:
                sub_img = Image.fromarray(img[i-pic_size:i,j-pic_size:j, :])
                sub_mask = Image.fromarray(mask[i-pic_size:i, j-pic_size:j, :])

                output_file = img_id.replace(".png", "{}.png".format(pic_num))

                # Save out the sub-image and the sub-mask in the segmentation folder
                sub_img.save(os.path.join(os.path.join(segmentation_output, "image"), output_file))
                sub_mask.save(os.path.join(os.path.join(segmentation_output, "mask"), output_file))

                # Save out the sub-image in the classification folder
                pic_class = "nonslum"

                print("Nonslum ID={}; pic_num={}; class={}".format(img_id, pic_num, pic_class))

                sub_img.save(os.path.join(os.path.join(classification_output, pic_class), output_file))

                pic_num += 1

                j += step_size
            j = pic_size
            i += step_size

def train_val_split_segmentation(train_pct):
    '''
    The function build_training_data just creates the two broad classes
    but does not do a training/validation split. Thus, import this function,
    then change to the directory where the data is and point using the target_folder
    to the data you want to split and it will put it into train/val folders
    '''


    all_files = np.array(os.listdir("mask"))
    is_train = np.random.binomial(1, train_pct, size=len(all_files))

    train_files = all_files[is_train==1]
    val_files = all_files[is_train==0]

    print("{}pct files in training".format(is_train.mean()))

    # Copy training files from original home to the training folder
    for f in train_files:
        src = os.path.join("mask", f)
        dst = os.path.join("train", "mask", f)
        shutil.copy(src, dst)

        src = os.path.join("image", f)
        dst = os.path.join("train", "image", f)
        shutil.copy(src, dst)

    for f in val_files:
        src = os.path.join("mask", f)
        dst = os.path.join("val", "mask", f)
        shutil.copy(src, dst)

        src = os.path.join("image", f)
        dst = os.path.join("val", "image", f)
        shutil.copy(src, dst)

def train_val_split_classification(target_folder, train_pct):
    '''
    NOTE: using 80% for training data
    NOTE: needs to be called twice, once targeted at 'slum' and once at 'nonslum'
    The function build_training_data just creates the two broad classes
    but does not do a training/validation split. Thus, import this function,
    then change to the directory where the data is and point using the target_folder
    to the data you want to split and it will put it into train/val folders
    '''


    all_files = np.array(os.listdir(target_folder))
    is_train = np.random.binomial(1, train_pct, size=len(all_files))

    train_files = all_files[is_train==1]
    val_files = all_files[is_train==0]

    print("{}pct files in training".format(is_train.mean()))

    # Copy training files from original home to the training folder
    for f in train_files:
        src = os.path.join(target_folder, f)
        dst = os.path.join("train", target_folder, f)
        shutil.copy(src, dst)

    for f in val_files:
        src = os.path.join(target_folder, f)
        dst = os.path.join("val", target_folder, f)
        shutil.copy(src, dst)


if __name__ == "__main__":
    pic_size = 128
    step_size = 128 
    zoom_level = 18
    class_threshold = .1
    mask_path = "labelbox/masks"
    slum_image_path = "google_earth/zoom_18/slums"
    nonslum_image_path = "google_earth/zoom_18/not_slums"

    build_training_data(zoom_level, pic_size, step_size, class_threshold, 
        mask_path, slum_image_path, nonslum_image_path)





