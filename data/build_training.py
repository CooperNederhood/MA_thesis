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
    mask_path, slum_image_path, nonslum_image_path, output_path, img_type="PIL"):
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

    f_ext = "png" if img_type=="PIL" else "npy"

    # Make folders, if needed
    if not os.path.isdir(os.path.join(output_path, "segmentation")):
        os.mkdir(os.path.join(output_path, "segmentation"))
    if not os.path.isdir(os.path.join(output_path, "classification")):
        os.mkdir(os.path.join(output_path, "classification"))

    if not os.path.isdir(os.path.join(output_path, "segmentation", "size_{}".format(pic_size))):
        os.mkdir(os.path.join(output_path, "segmentation", "size_{}".format(pic_size)))
    if not os.path.isdir(os.path.join(output_path, "classification", "size_{}".format(pic_size))):
        os.mkdir(os.path.join(output_path, "classification", "size_{}".format(pic_size)))

    segmentation_output = os.path.join(output_path, "segmentation", "size_{}".format(pic_size))
    classification_output = os.path.join(output_path, "classification", "size_{}".format(pic_size))

    # Make final sub-folders, if needed
    if not os.path.isdir(os.path.join(segmentation_output, "image")):
        os.mkdir(os.path.join(segmentation_output, "image"))
        os.mkdir(os.path.join(segmentation_output, "mask"))
    if not os.path.isdir(os.path.join(classification_output, "slum")):
        os.mkdir(os.path.join(classification_output, "slum"))
        os.mkdir(os.path.join(classification_output, "nonslum"))

    # First, process all the Ona slum images and their corresponding masks
    ona_images = [x.replace(".png", "") for x in os.listdir(mask_path)]

    for img_id in ona_images:

        img_id = img_id + "." + f_ext 

        # For RGB images, the file is a PIL image
        if img_type == "PIL":
            ona_img = np.array(Image.open(os.path.join(slum_image_path, img_id)))
           
        # But for multiband images, the file is a numpy array 
        # The channel will be FIRST, but it should be put to last
        # NOTE: it looks like channel 3 is the NIR band 
        else:
            assert img_type == "Numpy_array"
            ona_img = np.load(os.path.join(slum_image_path, img_id))
            ona_img = np.moveaxis(ona_img, 0, 2)

        ona_mask = np.array(Image.open(os.path.join(mask_path, img_id.replace(".npy", ".png"))))

        if img_type == "PIL":
            assert ona_img.shape == ona_mask.shape
        else:
            assert ona_img.shape[0:2] == ona_mask.shape[0:2]
            assert ona_img.shape[2] > 3

        i = pic_size
        j = pic_size
        max_i, max_j, _ = ona_img.shape

        # Loop over our current image and partition it into subimages
        pic_num = 0 
        while i < max_i:
            while j < max_j:
                #sub_img = Image.fromarray(ona_img[i-pic_size:i, j-pic_size:j, :])
                sub_mask = Image.fromarray(ona_mask[i-pic_size:i, j-pic_size:j, :])
                sub_img = ona_img[i-pic_size:i, j-pic_size:j, :]

                output_file = img_id.replace(".{}".format(f_ext), "{}.{}".format(pic_num, f_ext))
                mask_output_file = img_id.replace(".npy", ".png")
                mask_output_file = mask_output_file.replace(".png", "{}.png".format(pic_num))

                # Save out the sub-image in the segmentation folder
                if img_type == "PIL":
                    sub_img = Image.fromarray(sub_img)     
                    sub_img.save(os.path.join(os.path.join(segmentation_output, "image"), output_file))
                else:
                    assert img_type == "Numpy_array"
                    np.save(os.path.join(segmentation_output, "image", output_file), sub_img)
                
                # Save out the sub-mask the same way regardless of whether RGB or 4-Band
                sub_mask.save(os.path.join(segmentation_output, "mask", mask_output_file))

                # Save out the sub-image in the classification folder
                pic_class = determine_image_class(ona_mask[i-pic_size:i, j-pic_size:j, :], class_threshold)

                print("Ona ID={}; pic_num={}; class={}".format(img_id, pic_num, pic_class))

                # we're not really using the classification model so just skip it for the time being
                #sub_img.save(os.path.join(os.path.join(classification_output, pic_class), output_file))

                pic_num += 1

                j += step_size
            j = pic_size
            i += step_size
            
    print("DONE WITH SLUM IMAGES\n\n")

    # Second, process all the non-slum images we added
    nonslum_images = os.listdir(nonslum_image_path)

    for img_id in nonslum_images:

        # For RGB images, the file is a PIL image
        if img_type == "PIL":
            img = np.array(Image.open(os.path.join(nonslum_images, img_id)))
           
        # But for multiband images, the file is a numpy array 
        # The channel will be FIRST, but it should be put to last
        # NOTE: it looks like channel 3 is the NIR band 
        else:
            assert img_type == "Numpy_array"
            img = np.load(os.path.join(nonslum_image_path, img_id))
            img = np.moveaxis(img, 0, 2)

        # Make mask same for both
        x, y = img.shape[0:2]
        mask = np.zeros([x,y,3], dtype=np.uint8)

        if img_type == "PIL":
            assert ona_img.shape == ona_mask.shape
        else:
            assert ona_img.shape[0:2] == ona_mask.shape[0:2]
            assert ona_img.shape[2] > 3

        i = pic_size
        j = pic_size
        max_i, max_j, _ = img.shape 

        # Loop over the current non-slum image and partition
        pic_num = 0
        while i < max_i:
            while j < max_j:
                #sub_img = Image.fromarray(ona_img[i-pic_size:i, j-pic_size:j, :])
                sub_mask = Image.fromarray(mask[i-pic_size:i, j-pic_size:j, :])
                sub_img = img[i-pic_size:i, j-pic_size:j, :]

                output_file = img_id.replace(".{}".format(f_ext), "{}.{}".format(pic_num, f_ext))
                mask_output_file = img_id.replace(".npy", ".png")
                mask_output_file = mask_output_file.replace(".png", "{}.png".format(pic_num))

                # Save out the sub-image in the segmentation folder
                if img_type == "PIL":
                    sub_img = Image.fromarray(sub_img)     
                    sub_img.save(os.path.join(os.path.join(segmentation_output, "image"), output_file))
                else:
                    assert img_type == "Numpy_array"
                    np.save(os.path.join(os.path.join(segmentation_output, "image"), output_file), sub_img)
                
                # Save out the sub-mask the same way regardless of whether RGB or 4-Band
                sub_mask.save(os.path.join(os.path.join(segmentation_output, "mask"), mask_output_file))

                # Save out the sub-image in the classification folder
                pic_class = "nonslum"

                print("Nonslum ID={}; pic_num={}; class={}".format(img_id, pic_num, pic_class))

                #sub_img.save(os.path.join(os.path.join(classification_output, pic_class), output_file))

                pic_num += 1

                j += step_size
            j = pic_size
            i += step_size

def train_val_split_segmentation(train_pct, img_type="PIL"):
    '''
    The function build_training_data just creates the two broad classes
    but does not do a training/validation split. Thus, import this function,
    then change to the directory where the data is and point using the target_folder
    to the data you want to split and it will put it into train/val folders
    '''

    f_ext = ".png" if img_type=="PIL" else ".npy"

    if not os.path.isdir("train"):
        os.mkdir("train")
    if not os.path.isdir("val"):
        os.mkdir("val")

    if not os.path.isdir(os.path.join("train", "image")):
        os.mkdir(os.path.join("train", "image"))
    if not os.path.isdir(os.path.join("val", "image")):
        os.mkdir(os.path.join("val", "image"))

    if not os.path.isdir(os.path.join("train", "mask")):
        os.mkdir(os.path.join("train", "mask"))
    if not os.path.isdir(os.path.join("val", "mask")):
        os.mkdir(os.path.join("val", "mask"))


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

        src = os.path.join("image", f.replace(".png", f_ext))
        dst = os.path.join("train", "image", f.replace(".png", f_ext))
        shutil.copy(src, dst)

    for f in val_files:
        src = os.path.join("mask", f)
        dst = os.path.join("val", "mask", f)
        shutil.copy(src, dst)

        src = os.path.join("image", f.replace(".png", f_ext))
        dst = os.path.join("val", "image", f.replace(".png", f_ext))
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
    #pic_size = 128
    pic_size = 256
    step_size = 128 
    zoom_level = 18
    class_threshold = .5

    # The below 4 paths need to be set depending on the dataset being analyzed
 
    # # USED FOR GOOGLE EARTH IMAGES:   
    # mask_path = "labelbox/google_earth_masks"
    # slum_image_path = "google_earth/zoom_18/slums"
    # nonslum_image_path = "google_earth/zoom_18/not_slums"
    # output_path = "training_data/google_earth"


    # USED FOR PLEIADES IMAGES -- RGB
    # mask_path = "labelbox/pleiades"
    # slum_image_path = "descartes/RGB/min_cloud/slums"
    # nonslum_image_path = "descartes/RGB/min_cloud/not_slums"
    # output_path = "training_data/descartes/RGB"

    # USED FOR PLEIADES IMAGES -- 4Band
    mask_path = "labelbox/pleiades"
    slum_image_path = "descartes/Four_band/min_cloud/slums"
    nonslum_image_path = "descartes/Four_band/min_cloud/not_slums"
    output_path = "training_data/descartes/Four_band"

    build_training_data(zoom_level, pic_size, step_size, class_threshold, 
        mask_path, slum_image_path, nonslum_image_path, output_path, img_type="Numpy_array")





