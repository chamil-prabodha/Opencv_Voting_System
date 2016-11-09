import cv2
import numpy as np

#Load images
img = cv2.imread('001.jpg')
template = cv2.imread('P1.jpg',0)

#convert to grayscale, keep the original image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

w,h = template.shape[::-1]

res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8

loc = np.where(res>=threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,255,255),2)


#------------------IMG SHOW-----------------------------

img = cv2.resize(img,(0,0),fx=0.25,fy=0.25)

cv2.imshow('image',img)
cv2.imshow('template',template)


cv2.waitKey(0)