"""This python file contains all the preprocessing steps before feed into Unet
"""
from keras.preprocessing.image import ImageDataGenerator
import glob
import math
import numpy as np
import os
import shutil
import skimage.io as io
import skimage.transform as trans


def split_train_val(base_dir, train_img, train_mask, val_img, val_mask):
    """Randomly split training data sets into
    training set (95%) and validation set (5%)

    Args
    ----
    base_dir : base directory of all folders
    train_img : path to the images folder in training set
    train_mask : path to the mask folder in training set
    val_img : path to the images folder in validation set
    val_mask : path to the mask folder in validation set

    Returns
    -------
    tra_num : number of images in training folder
    val_num : number of images in validation folder
    """
    sourceN = base_dir + train_img
    destN = base_dir + val_img
    sourceP = base_dir + train_mask
    destP = base_dir + val_mask
    filesN = os.listdir(sourceN)

    for f in filesN:
        if np.random.rand(1) < 0.05:
            shutil.move(sourceN + '/' + f, destN + '/' + f)
            shutil.move(sourceP + '/' + f, destP + '/' + f)
    print(len(os.listdir(sourceN)))
    print(len(os.listdir(sourceP)))
    print(len(os.listdir(destN)))
    print(len(os.listdir(destP)))
    tra_num = len(os.listdir(sourceN))
    val_num = len(os.listdir(destN))
    return tra_num, val_num


def data_is_ok(data, raise_exception=False):
    """Perform a check to ensure the image data is in the correct range

    Args
    ----
    data (np.array) : the image data
    raise_exception (bool) : raise exception if data is not ok

    Returns
    -------
    (bool) : True if data is OK, otherwise False
    """
    try:
        assert data.dtype == np.uint8
        assert data.max() <= 255
        assert data.min() >= 0

        # make sure data wasn't normalized to [0,1]
        assert data.max() > 1.0

    except AssertionError as e:
        if raise_exception:
            raise e
        else:
            _data_is_ok = False
    else:
        _data_is_ok = True
        print(_data_is_ok)
    return _data_is_ok


def image_save_preprocessor(img, report=True):
    """Normalize the image
    Procedure
    ---------
    - Convert higher bit images (16, 10, etc) to 8 bit
    - Set color channel to the last channel

    Args
    ----
    img (np array) : raw image data
    report (bool) : output a short log on the imported data

    Returns
    -------
    numpy array of cleaned image data with values [0, 255]
    """
    data = np.asarray(img)
    if data.ndim == 3:
        # set the color channel to last if in channel_first format
        if data.shape[0] <= 4:
            data = np.rollaxis(data, 0, 3)

        # remove alpha channel
        if data.shape[-1] == 4:
            data = data[..., :3]

    # if > 8 bit, shift to a 255 pixel max
    bitspersample = int(math.ceil(math.log(data.max(), 2)))
    if bitspersample > 8:
        data >>= bitspersample - 8

    # if data [0, 1), then set range to [0,255]
    if bitspersample <= 0:
        data *= 255

    data = data.astype(np.uint8)
    if report:
        print("Cleaned To:")
        print("\tShape: ", data.shape)
        print("\tdtype: ", data.dtype)

    # Make sure the data is actually in the correct format
    data_is_ok(data, raise_exception=True)
    return data


def preprocess_image_for_model(data, raise_exception=False):
    """Process data in the manner expected by Unet
    preprocessor that takes a clean image and performs final adjustments
    before it's feed into a model

    Args
    ----
    data (np.array)

    Returns
    -------
    normalized data of the same shape
    """
    try:
        data_is_ok(data, raise_exception=True)
    except AssertionError as e:
        data = image_save_preprocessor(data, report=False)
        # Doing this for debug purposes
        if raise_exception:
            raise e

    # cast as float
    data = data.astype(np.float32)
    # Normalize to the mean of the color channels
    # data[...,0] -= 103.939
    # data[...,1] -= 116.779
    # data[...,2] -= 123.78

    return data


def adjust_data(img, mask):
    """Adjust data value between [0,1]

    Args
    ----
    img, mask (np.array)

    Returns
    -------
    normalized data of the same shape
    """
    if (np.max(img) > 1):
        img = preprocess_image_for_model(img)
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def train_generator(batch_size, train_path, image_folder,
                    mask_folder, aug_dict, image_color_mode='rgb',
                    mask_color_mode='rgb', image_save_prefix='img',
                    mask_save_prefix='mask', save_to_dir=None,
                    target_size=(256, 256), seed=1):
    """Generate training and mask set together

    Args
    ----
    batch_size : number of images in a batch
    train_path : path to training folder
    image_folder : path to image folder in training set
    mask_folder : path to mask folder in training set
    aug_dict : augmentation argument (optional)
    image_color_mode : 'rgb' or 'grayscale'
    mask_color_mode : 'rgb' or 'grayscale'
    image_save_prefix : prefix for augmented images when save_to_dir = PATH
    mask_save_prefix : prefix for augmented masks when save_to_dir = PATH
    save_to_dir : set to PATH if you want to see the generator results
    target_size : desired image size for Unet
    seed : Optional random seed for shuffling and transformations

    Returns
    -------
    Tuples of (image, mask) for training set where image is a numpy array containing a batch of images, mask is a numpy array of corresponding labels
    """
    image_datagen = ImageDataGenerator(*aug_dict)
    mask_datagen = ImageDataGenerator(*aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask)
        yield (img, mask)


def valid_generator(batch_size, val_path, image_folder,
                    mask_folder, aug_dict, image_color_mode='rgb',
                    mask_color_mode='rgb', image_save_prefix='img',
                    mask_save_prefix='mask', save_to_dir=None,
                    target_size=(256, 256), seed=1):
    """Generate validation set

    Args
    ----
    batch_size : number of images in a batch
    val_path : path to validation folder
    image_folder : path to image folder in validation set
    mask_folder : path to mask folder in validation set
    aug_dict : augmentation argument (optional)
    image_color_mode : 'rgb' or 'grayscale'
    mask_color_mode : 'rgb' or 'grayscale'
    image_save_prefix : prefix for augmented images when save_to_dir = PATH
    mask_save_prefix : prefix for augmented masks when save_to_dir = PATH
    save_to_dir : set to PATH if you want to see the generator results
    target_size : desired image size for Unet
    seed : Optional random seed for shuffling and transformations

    Returns
    -------
    Tuples of (image, mask) for validation set where image is a numpy array containing a batch of images, mask is a numpy array of corresponding labels
    """
    image_datagen = ImageDataGenerator(*aug_dict)
    mask_datagen = ImageDataGenerator(*aug_dict)
    valimage_generator = image_datagen.flow_from_directory(
        val_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    val_generator = zip(valimage_generator, mask_generator)

    for (val_img, val_mask) in val_generator:
        val_img, val_mask = adjust_data(val_img, val_mask)
        yield (val_img, val_mask)


def test_generator(test_path, num_image=54, target_size=(256, 256), as_gray=False):
    """Generate test set

    Args
    ----
    test_path : path to test folder
    num_image : number of test images
    target_size : desired image size for Unet
    as_gray : True for grayscale image; vice versa

    Returns
    -------
    A numpy array containing a batch of test images
    """
    list = sorted(glob.glob('test/*.png'))
    for i in range(len(list)):
        img = io.imread(list[i])
        img = preprocess_image_for_model(img)
        img = img / 255
        img = trans.resize(img, target_size)
        if img.shape != (256, 256, 3):
            img = img[:, :, :3]
        img = np.reshape(img, (1,) + img.shape)
        yield img
