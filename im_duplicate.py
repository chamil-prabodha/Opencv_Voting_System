import random
import cv2
pos = 14

# for i in range(0,2000):
#     # print('d')
#     rand = random.randrange(2,14)
#     img = cv2.imread('img/P'+str(rand)+'.jpg')
#     # cv2.imshow('adf',img)
#     cv2.imwrite('img/P'+str(pos)+'.jpg',img)
#
#     with open('bg.txt','a') as f:
#         f.write('img/P'+str(pos)+'.jpg\n')
#
#     pos += 1


for i in range(2,2014):
    with open('bg.txt','w') as f:
        f.write('C:\Users\Chamil\PycharmProjects\CV_Project\img\P'+str(i)+'.jpg')