# Functions for post-processing steps
import os
import skimage.io as io
import numpy as np
#import cv2
import glob
from PIL import Image
import requests
import urllib
import matplotlib.pyplot as plt
import json

# Save the prediction image at the provided save_path
def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, '%d_pred.png'%i), img)

### Draw the boundary; prediction results as input; resize back to original size
#def draw_boundary(pred_path, test_path, ori_size):
#    pred_list = sorted(glob.glob('*_pred.png'))
#    test_list = sorted(glob.glob('*_test.png'))
#    print (pred_list)
#    print (test_list)
#    for i in range (len(pred_list)):
#            img = cv2.imread(os.path.join(pred_path, pred_list[i]))
#            test = Image.open(os.path.join(test_path, test_list[i]))
#            test_resize = test.resize(ori_size, Image.NEAREST)
#            test_arr = np.array(test_resize)
#            imbw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#            (thresh, im) = cv2.threshold(imbw, 128, 255, 0)
#            _, contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#            val_cnt = []
#            for cnt in contours:
#                area = cv2.contourArea(cnt)
#                print (area)
#                if area > 400:
#                    val_cnt.append(cnt)
#            print (len(val_cnt))
#            cv2.drawContours(test_arr,val_cnt,-1,(0,255,0),3)
#            cv2.imwrite('%d_ct.png'%i,test_arr)
#
##draw_boundary('', '', (256,256))
#
## Grab roof image from googlemap API
#center_x = 37.451736
#center_y = -122.146029
#r = requests.get('https://maps.googleapis.com/maps/api/staticmap?center={},{} &zoom=19&size=200x200&maptype=satellite&'.format(center_x, center_y))
#img_array = np.array(bytearray(r.content), dtype=np.uint8)
#img = cv2.imdecode(img_array, -1)
#print (img.shape)
##cv2.imshow('URL Image', img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#plt.imshow(img)
#plt.show()
#
## request roof area from sunroof API
#url = 'https://www.google.com/async/sclp?async=lat:{},lng:{}'.format(center_x, center_y)
#
#response = requests.get(url)
#r = (response.text[4:])
#d = json.loads(r)
#
#print (d["HouseInfoResponse"]["house_assumptions"]['roof_good_solar_square_feet'])
#
