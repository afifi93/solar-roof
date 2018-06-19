# Functions for post-processing steps
#import cv2
import os
import skimage.io as io

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
#        img = item[:, :, 0]
#        imbw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#        (thresh, im) = cv2.threshold(imbw, 0.5, 1, 0)
#        _, contours,_ = cv2.findContours(im, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#        cnt = trans.resize(contours,ori_size)
#        cv2.drawContours(img,cnt,-1,(0,255,0),2)
#        cv2.imwrite('%d_cnt.png'%i, test)
