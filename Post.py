# Functions for post-processing steps
import cv2

# Post-processing prediction image to binary image
def to_binary(pred, th):
    pred[pred > th] = 1
    pred[pred <= th] = 0
    return pred

# Draw the boundary; path as input
def draw_bound(img):
#    img = cv2.imread('path/test.jpg')
    imbw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (thresh, im) = cv2.threshold(imbw, 0.5, 1, 0)
    _, contours,_ = cv2.findContours(im, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,contours,-1,(0,255,0),2)
    cv2.imwrite('cnt.png', img)

