
# coding: utf-8

# In[1]:


from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
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
                  flag_multi_class = False, num_class = 1, save_to_dir = 'notebook/ex', target_size = (500, 500), seed = 1):
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


# In[5]:

def testGenerator(test_path, num_image = 1, target_size = (500, 500), flag_multi_class = False, as_gray = True):
    #for i in range(num_image):
        img = io.imread(os.path.join(test_path, '0_01_01.png'), as_gray = as_gray)
        #img = cv2.imread(os.path.join(test_path, '0_01_01.png'),0)
        #img = rgb2gray(img) / 255
        img = img / 255
        #img = img.astype(np.float32)
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


# In[6]:


#def genTrainNpy(image_path, mask_path, flag_multi_class = False, num_class = 2, image_prefix = None, mask_prefix = None, image_as_gray = True, mask_as_gray = True):
#image_name_arr = glob.glob(os.path.join(image_path, '%s*.png'%image_prefix))
#image_arr = []
# mask_arr = []
# for index, item in enumerate(image_name_arr):
#       img = io.imread(item, as_gray = image_as_gray)
#       img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
#    mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray = mask_as_gray)
#       mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
#       img, mask = adjustData(img, mask, flag_multi_class, num_class)
#       image_arr.append(img)
#       mask_arr.append(mask)
#   image_arr = np.array(image_arr)
#   mask_arr = np.array(mask_arr)
#   return image_arr, mask_arr


def saveResult(save_path, npyfile, flag_multi_class = False, num_class = 1):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, '%d_predictgray256.png'%i), img)

