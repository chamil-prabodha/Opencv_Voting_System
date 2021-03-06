import cv2
import numpy as np
import ImageTransform

class ImageSegmentation(object):

    def segmentPreferenceVoting(self,img_main):

        gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # ---------------------<Other blur functions>
        # gray = cv2.fastNlMeansDenoising(gray,None,18,7,21)
        # thresh, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        # ----------------------</Other blur functions>

        gray_erode = cv2.erode(gray.copy(), np.ones((5, 5), np.uint8), iterations=1)
        cannyout = cv2.Canny(gray_erode, 50, 100, 5)

        im2, contours, hierarchy = cv2.findContours(cannyout.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the contours with convexity and four points
        contours = [contour for contour in contours if len(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4]
        contours = sorted(contours, key=lambda x: (cv2.contourArea(x)), reverse=True)

        x, y, w, h = cv2.boundingRect(contours[1])

        pref_im = img_main[y:y + h, x:x + w]



        return pref_im

    def segmentPartyVoting(self,img_main):
        gray = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # ----------------------<Other functions>--------------------------------------
        # gray = cv2.fastNlMeansDenoising(gray,None,18,7,21)
        # thresh, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        # ----------------------</Other functions>-------------------------------------
        
        gray_erode = cv2.erode(gray.copy(), np.ones((5, 5), np.uint8), iterations=1)
        cannyout = cv2.Canny(gray_erode, 50, 100, 5)

        im2, contours, hierarchy = cv2.findContours(cannyout.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the contours with convexity and four points
        contours = [contour for contour in contours if not cv2.isContourConvex(contour) and len(
            cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4]
        contours = sorted(contours, key=lambda x: (cv2.contourArea(x)), reverse=True)

        transform = ImageTransform.ImageTransform()
        good_lines = transform.get_verticle_line_angles(cannyout)

        if len(good_lines) == 0:
            # Straighten image if tilted
            dest = transform.rotateImage(img_main, contours[0])
            img = dest

        else:
            x, y, w, h = cv2.boundingRect(contours[0])

            # Select the largest rectangle containing signs and voting boxes
            img = img_main[y:y + h, x:x + w]

        img_main = img
        return img_main
