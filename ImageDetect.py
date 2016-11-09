import numpy as np
import cv2

img = cv2.imread('001.jpg')


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)


edges = cv2.Canny(gray,100,200,1)

im2,contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,color=(0,255,0),thickness=3)

edges = cv2.resize(edges,(0,0),fx=0.25,fy=0.25)
img = cv2.resize(img,(0,0),fx=0.25,fy=0.25)

cv2.imshow('Grey',edges)
cv2.imshow('Image',img)

#Wait for key to exit
while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


