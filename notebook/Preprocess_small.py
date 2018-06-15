from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import cv2
from keras import backend as keras
import skimage.transform as trans

def adjustData(img, mask):
    if (np.max(img) <= 255 and np.min(img) >= 0):
       img = img / 255
       mask = mask / 255
       mask[mask > 0.5] = 1
       mask[mask <= 0.5] = 0
    else:
        print ('not between 0 and 255!')
    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder,
                   mask_folder, aug_dict, image_color_mode = 'grayscale',
                   mask_color_mode = 'grayscale', image_save_prefix = 'img',
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

#def IOU_calc(y_true, y_pred):
#   y_true_f = keras.flatten(y_true)
#   y_pred_f = keras.flatten(y_pred)
#   intersection = keras.sum(y_true_f * y_pred_f)
    
#   return 2*(intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)


def testGenerator(test_path, num_image = 1, target_size = (256, 256), as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d_01_01.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img

def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, '%d_predict.png'%i), img)

