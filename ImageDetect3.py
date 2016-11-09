import cv2
import numpy as np

img = cv2.imread('001.jpg')
template = cv2.imread('P1.jpg',0)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray,(3,3))

template = cv2.GaussianBlur(template,(5,5),0)
# thresh, template = cv2.threshold(template,127,255,cv2.THRESH_BINARY)
template_edges = cv2.Canny(template,100,200,10)

im1, temp_cnt, hierarchy = cv2.findContours(template_edges.copy(),cv2.RETR_TREE,cv2.RETR_FLOODFILL)

# temp_areas = []
# for i in temp_cnt:
#     temp_areas.append(cv2.contourArea(i))

gray = cv2.GaussianBlur(gray,(5,5),0)
cannyout = cv2.Canny(gray,100,200,10)

im2,contours,hierarchy = cv2.findContours(cannyout.copy(),cv2.RETR_TREE,cv2.RETR_FLOODFILL)
#cv2.drawContours(img,contours,-1,color=(0,255,0),thickness=3)
mu = []
for i in contours:
    mu.append(cv2.moments(i))

mc = []
#-----------------------------------------------------------------------------
# for i in mu:
#     mc.append([int(i['m10']/i['m00']),int(i['m01']/i['m00'])])
#-----------------------------------------------------------------------------


# for i in temp_cnt:
#     temp_area = cv2.contourArea(i)
#     temp_peri = cv2.arcLength(i,False)
#     for j in contours:
#         con_area = cv2.contourArea(j)
#         con_peri = cv2.arcLength(j,False)
#
#         if temp_area == con_area:
#
#             x,y,w,h = cv2.boundingRect(j)
#             cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

for cont in contours:
    area = cv2.contourArea(cont)
    for temp in temp_cnt:
        match = cv2.matchShapes(cont,temp,1,0.0)

        if match < 0.15 and area >750 :
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

#------------------IMG SHOW-------------------------
img = cv2.resize(img,(0,0),fx=0.25,fy=0.25)
gray = cv2.resize(cannyout,(0,0),fx=0.25,fy=0.25)
cv2.imshow('Image',img)
cv2.imshow('Gray',template_edges)
cv2.waitKey(0)
