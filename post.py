"""This python file contains all the postprocessing steps after predictions
    """
import cv2
import glob
from keras import backend as K
import numpy as np
import os
from PIL import Image
import skimage.io as io


def jaccard_coef(y_true, y_pred):
    """Intersection area over union (IoU)
        Args
        ----
        y_true : mask array
        y_pred : predicted array
        Returns
        -------
        average IoU over all images
        """
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_loss(y_true, y_pred):
    """IoU as loss function
        Args
        ----
        y_true : mask array
        y_pred : predicted array
        Returns
        -------
        loss function
        """
    return -K.log(jaccard_coef(y_true, y_pred)) + K.binary_crossentropy(y_pred, y_true)


# Save loss/accuracy history as txt file
# print(history.history.keys())
def save_history(history):
    """Save training loss and validation loss as text files
        Args
        ----
        hist : loss history after each epoch
        Returns
        -------
        Saved text files for training loss and validation loss
        """
    train_array = np.array(history.history['loss'])
    val_array = np.array(history.history['val_loss'])
    np.savetxt('loss_history70.txt', train_array, delimiter=',')
    np.savetxt('val_history70.txt', val_array, delimiter=',')


def save_result(save_path, npyfile):
    """Save predicted image to png file
        Args
        ----
        save_path : desired path to save predicted png image
        npyfile : output predicted ndarray from model
        Returns
        -------
        Saved png images at desired path
        """
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, '%d_pred.png' % i), img)


def draw_boundary(pred_path, test_path, ori_size):
    """Draw contour around the building
        Args
        ----
        pred_path : path to predicted png images
        test_path : path to test images
        ori_size : size of predicted images
        Returns
        -------
        Saved png images with contour around the building on the original test images
        """
    pred_list = sorted(glob.glob('*_pred.png'))
    test_list = sorted(glob.glob('*_test.png'))
    for i in range(len(pred_list)):
        img = cv2.imread(os.path.join(pred_path, pred_list[i]))
        test = Image.open(os.path.join(test_path, test_list[i]))
        test_resize = test.resize(ori_size, Image.NEAREST)
        test_arr = np.array(test_resize)
        imbw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        (thresh, im) = cv2.threshold(imbw, 128, 255, 0)
        _, contours, _ = cv2.findContours(
            im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        val_cnt = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:
                val_cnt.append(cnt)
        cv2.drawContours(test_arr, val_cnt, -1, (0, 255, 0), 3)
        cv2.imwrite('%d_ct.png' % i, test_arr)
