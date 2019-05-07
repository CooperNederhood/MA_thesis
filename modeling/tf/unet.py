from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd 
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.keras import layers, models, callbacks, losses 

import os 
import matplotlib.pyplot as plt 
import functools 

#tf.enable_eager_execution()

print("GPU is available?: ", tf.test.is_gpu_available())

ROOT = "../../"
MODELING = os.path.join(ROOT, "modeling")
DATA = os.path.join(ROOT, "data")

descartes = os.path.join(DATA, "training_data", "descartes", "RGB", "segmentation", "size_256")
train_path = os.path.join(descartes, "train")
val_path = os.path.join(descartes, "val")

# Parameters
BATCH_SIZE = 8
THREADS = 4
EPOCHS = 1


# Build the data pipeline
def build_image_mask_paths(path):
    '''
    Given a path to the image and mask subfolders, returns two lists
    which contain the full paths
    '''

    files = os.listdir(os.path.join(path, "mask"))
    image_files = [os.path.join(path, "image", f) for f in files]
    mask_files = [os.path.join(path, "mask", f) for f in files]

    return (image_files, mask_files)

def _process_pathnames(image_file, mask_file):

    img_str = tf.read_file(image_file)
    img = tf.image.decode_png(img_str, channels=3)

    mask_str = tf.read_file(mask_file)
    mask = tf.image.decode_png(mask_str, channels=3)
    mask = mask[:,:,0]
    mask = tf.expand_dims(mask, axis=-1)

    return img, mask

def shift_img(img, mask, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""

    img_shape = (256, 256, 3)

    if width_shift_range or height_shift_range:
        if width_shift_range:
          width_shift_range = tf.random_uniform([], 
                                              -width_shift_range * img_shape[1],
                                              width_shift_range * img_shape[1])
        if height_shift_range:
          height_shift_range = tf.random_uniform([],
                                               -height_shift_range * img_shape[0],
                                               height_shift_range * img_shape[0])
      # Translate both 
        img = tfcontrib.image.translate(img,
                                             [width_shift_range, height_shift_range])
        mask = tfcontrib.image.translate(mask,
                                             [width_shift_range, height_shift_range])
    return img, mask


def flip_img(img, mask, flip_rate):
    if flip_rate:

        # Do horizontal flip
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        img, mask = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(mask)),
                                lambda: (img, mask))

        # Do vertical flip
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        img, mask = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_up_down(img), tf.image.flip_up_down(mask)),
                                lambda: (img, mask))

    return img, mask

train_image_files, train_mask_files = build_image_mask_paths(train_path)
val_image_files, val_mask_files = build_image_mask_paths(val_path)

# Define our data augmentation function
def _augment(img, mask, shift_range, flip_rate):

    img, mask = shift_img(img, mask, shift_range, shift_range)
    img, mask = flip_img(img, mask, flip_rate)

    img = tf.to_float(img) * (1/255.) # rescale to 0-1

    return img, mask 

FLIP_RATE = 0.5
SHIFT_PCT = 0

augment_function = functools.partial(_augment, shift_range=SHIFT_PCT, flip_rate=FLIP_RATE)

# Make a dataset with augmentation
def make_dataset(data_path, shift_pct, flip_rate, batch_size, threads):
    image_files, mask_files = build_image_mask_paths(data_path)
    augment_function = functools.partial(_augment, shift_range=shift_pct, flip_rate=flip_rate)

    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.shuffle(len(image_files))
    dataset = dataset.repeat()
    dataset = dataset.map(_process_pathnames, num_parallel_calls = threads)
    dataset = dataset.map(augment_function, num_parallel_calls = threads)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset

train_data = make_dataset(train_path, 0.0, 0.5, BATCH_SIZE, THREADS)
val_data = make_dataset(train_path, 0.0, 0.0, BATCH_SIZE, THREADS)

# Plot a batch of our training data
def plot_data(data):
    x,y = next(iter(data))

    n = x.shape[0].value 
    for i in range(n):
        j = 2*(i+1)
        plt.subplot(n, 2, j-1)
        plt.imshow(x[i,:,:,:])

        plt.subplot(n, 2, j)
        #plt.imshow(y[i,:,:,:])

    plt.show()

#plot_data(train_data)

# Define helper functions for our U-Net model

def conv_block(input_tensor, ftrs, is_batchnorm=True):
    
    out = layers.Conv2D(ftrs, (3,3), padding='same')(input_tensor)
    if is_batchnorm:
        out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)

    out = layers.Conv2D(ftrs, (3,3), padding='same')(out)
    if is_batchnorm:
        out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)

    return out 

def encoder_block(input_tensor, ftrs, is_batchnorm=True):

    out = conv_block(input_tensor, ftrs, is_batchnorm)
    out_downsampled = layers.MaxPool2D((2,2))(out)

    return out_downsampled, out

def decoder_block(input_tensor, concat_tensor, ftrs, is_batchnorm=True):

    out = layers.Conv2DTranspose(ftrs, kernel_size=(2,2), 
                strides=(2,2), padding='same')(input_tensor)
    out = layers.concatenate([out,concat_tensor], axis=-1)
    if is_batchnorm:
        out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    out = conv_block(out, ftrs, is_batchnorm)

    return out 

#f = 64
f = 32

inputs = layers.Input(shape=(256,256,3))

encoder0_pooled, encoder0 = encoder_block(inputs, f)
encoder1_pooled, encoder1 = encoder_block(encoder0_pooled, f*2)
encoder2_pooled, encoder2 = encoder_block(encoder1_pooled, f*4)
encoder3_pooled, encoder3 = encoder_block(encoder2_pooled, f*8)

bottleneck = conv_block(encoder3_pooled, f*16)

decoder0 = decoder_block(bottleneck, encoder3, f*8)
decoder1 = decoder_block(decoder0, encoder2, f*4)
decoder2 = decoder_block(decoder1, encoder1, f*2)
decoder3 = decoder_block(decoder2, encoder0, f)

outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(decoder3)

unet = models.Model(inputs=[inputs], outputs=[outputs])


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

# Callbacks
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_dice_loss', factor=0.2,
                        patience=2, min_lr=0.001)
if not os.path.isdir("model"):
    os.mkdir("model")

cp = callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor="val_dice_loss",
                        save_best_only=True, verbose=1)

unet.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])

unet.summary()

history = unet.fit(train_data, 
    epochs=EPOCHS, 
    steps_per_epoch=int(np.ceil(len(train_image_files)/BATCH_SIZE)),
    validation_data=val_data,
    validation_steps=validation_steps_per_epoch=int(np.ceil(len(val_image_files)/BATCH_SIZE)),
    callbacks=[reduce_lr, cp])

df_history = pd.DataFrame(history.history)
df_history.to_csv("model/train_history.csv")


