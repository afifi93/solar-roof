# Functions for post-processing steps
#import cv2
import os
import skimage.io as io
#import requests
import numpy as np
import cv2

# Save the prediction image at the provided save_path
def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, '%d_pred.png'%i), img)

## Post-processing prediction image to binary image
#def to_binary(pred, th):
#    pred[pred > th] = 1
#    pred[pred <= th] = 0
#    return pred
#
## Draw the boundary; prediction results as input; resize back to original size
#def draw_bound(npyfile, test_path, ori_size):
#    for i, item in enumerate(npyfile):
#        test = io.imread(os.path.join(test_path,"%d.png"%i))
#img = io.imread('0_pred.png')
##img = item[:, :, 0]
#imbw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#gray_blur = cv2.GaussianBlur(imbw, (15, 15), 0)
#thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
#kernel = np.ones((3, 3), np.uint8)
#closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
#cont_img = closing.copy()
#contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#for cnt in contours:
#    area = cv2.contourArea(cnt)
#    if area < 30:
#        continue
#cv2.drawContours(img,cnt,-1,(0,255,0),2)
#cv2.imwrite('0_ct.png',img)
#cv2.imshow('final result', roi)

#        (thresh, im) = cv2.threshold(imbw, 0.5, 1, 0)
#        _, contours,_ = cv2.findContours(im, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#        cnt = trans.resize(contours,ori_size)
#        cv2.drawContours(img,cnt,-1,(0,255,0),2)
#        cv2.imwrite('%d_cnt.png'%i, test)

# Grab roof area from API
#r = requests('https://www.google.com/async/sclp?async=lat:{},lng:{}'.format{lat, lng})
#np.savetxt('API query.txt', r.text())
# extract "roof_good_solar_square_feet":

