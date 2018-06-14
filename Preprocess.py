
# coding: utf-8

# In[1]:


from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
#from keras import backend as keras
#from skimage.color import rgb2gray

# In[3]:


def adjustData(img, mask, flag_multi_class, num_class):
   if (flag_multi_class):
       img = img / 255
       mask = mask[:, :, :, :, 0] if(len(maskshape) == 4) else mask[:, :, 0]
       new_mask = np.zeros(mask.shape + (num_class,))
       for i in range(num_class):
           new_mask[mask == i, i] =1
       new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
       mask = new_mask
   elif(np.max(img) > 1):
       img = img / 255
       mask = mask / 255
       mask[mask > 0.5] = 1
       mask[mask <= 0.5] = 0
   return (img, mask)


# In[4]:


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = 'grayscale', 
                  mask_color_mode = 'grayscale', image_save_prefix = 'img', mask_save_prefix = 'mask',
                  flag_multi_class = False, num_class = 1, save_to_dir = None, target_size = (500, 500), seed = 1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
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
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)

#def IOU_calc(y_true, y_pred):
#   y_true_f = keras.flatten(y_true)
#   y_pred_f = keras.flatten(y_pred)
#   intersection = keras.sum(y_true_f * y_pred_f)
    
#   return 2*(intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

# In[5]:

def testGenerator(test_path, num_image = 1, target_size = (500, 500), flag_multi_class = False, as_gray = True):
    #for i in range(num_image):
        img = io.imread(os.path.join(test_path, '0_01_01.png'), as_gray = as_gray)
        img = img / 255
        #img = img.astype(np.float32)
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, -1]
        print (img.shape)
        print (img)
        io.imsave(os.path.join(save_path, '%d_predictgray.png'%i), img)

