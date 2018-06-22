# Functions for pre-processing steps
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import skimage.io as io
from keras import backend as K
import skimage.transform as trans
from check_data import *

# Clean up data - normalize image intensity and make mask binary
def adjustData(img, mask):
    if (np.max(img) > 1):
       img = preprocess_image_for_model(img)
       img = img / 255
       mask = mask / 255
       mask[mask > 0.5] = 1
       mask[mask <= 0.5] = 0
    return (img, mask)

# Generate training and mask set together (customize the batch_size and target size)
# Change save_to_dir = PATH if you want to see the generator results
def trainGenerator(batch_size, train_path, image_folder,
                   mask_folder, aug_dict, image_color_mode = 'rgb',
                   mask_color_mode = 'rgb', image_save_prefix = 'img',
                   mask_save_prefix = 'mask', save_to_dir = None,
                   target_size = (256, 256), seed = 1):
    image_datagen = ImageDataGenerator(*aug_dict)
    mask_datagen = ImageDataGenerator(*aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path, 
        classes = [image_folder], 
        class_mode = None, 
        color_mode = image_color_mode, 
        target_size = target_size, 
        batch_size = batch_size, 
        save_to_dir = save_to_dir, 
        save_prefix = image_save_prefix, 
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path, 
        classes = [mask_folder], 
        class_mode = None, 
        color_mode = mask_color_mode, 
        target_size = target_size, 
        batch_size = batch_size, 
        save_to_dir = save_to_dir, 
        save_prefix = mask_save_prefix, 
        seed = seed)
    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)

# Generate validation set
def valGenerator(batch_size, val_path, image_folder,
                 mask_folder, aug_dict, image_color_mode = 'rgb',
                 mask_color_mode = 'rgb', image_save_prefix = 'img',
                 mask_save_prefix = 'mask', save_to_dir = None,
                 target_size = (256, 256), seed = 1):
    image_datagen = ImageDataGenerator(*aug_dict)
    mask_datagen = ImageDataGenerator(*aug_dict)
    valimage_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = mask_save_prefix,
        seed = seed)
    val_generator = zip(valimage_generator, mask_generator)
    
    for (val_img, val_mask) in val_generator:
        val_img, val_mask = adjustData(val_img, val_mask)
        yield (val_img, val_mask)

# Intersection area over union as the loss function
def IoU_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    IoU = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(IoU)

def IoU_coef_int(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    IoU = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(IoU)

def IoU_loss(y_true, y_pred):
    return -K.log(IoU_coef(y_true, y_pred)) + K.binary_crossentropy(y_pred, y_true)

# Generate test set altogether (customize target_size)
def testGenerator(test_path, num_image = 54, target_size = (256, 256), as_gray = False):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = preprocess_image_for_model(img)
        img = img / 255
        img = trans.resize(img,target_size)
        if img.shape != (256, 256, 3):
            img = img[:, :, :3]
#        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img



