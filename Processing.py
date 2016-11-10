
import cv2
import numpy as np
import threading
from Tkinter import *
import Tkconstants
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import os

class Processing(object):

    def __init__(self,panelA,panelB,topframe,prg_bar):
        self.panelA = panelA
        self.panelB = panelB
        self.topframe = topframe
        self.prg_bar = prg_bar

    def startDetectPartyImage(self,img,template):
        thread = threading.Thread(target=self.detectPartyImage,args=(img,template,),name='detectPartyImage')
        thread.start()

    def startDetectPartyVote(self,img,img_0,path,mutex,arr,pref_votes):
        thread = threading.Thread(target=self.detectPartyVote,args=(img,img_0,path,mutex,arr,pref_votes,),name='detectPartyVote')
        thread.start()

    def detectPartyImage(self,img,template):
        # panelA.grid_forget()
        img_0 = img.copy()
        # -------------------<Ballot paper>----------------------------

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges
        cannyout = cv2.Canny(gray, 50, 100, 5)

        # Find contours
        im2, contours, hierarchy = cv2.findContours(cannyout.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the contours with convexity and four points
        contours = [contour for contour in contours if not cv2.isContourConvex(contour) and len(
            cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4]

        # contours = sorted(contours, key=lambda x: (cv2.contourArea(x)), reverse=True)

        # ------------------</Ballot paper>----------------------------

        # ------------------<Party sign>-------------------------------

        # Read the image
        # template_ori = cv2.imread(path, 0)
        template_ori = template
        # template = cv2.bilateralFilter(template_ori,9,75,75)
        # template = cv2.medianBlur(template_ori,5)
        # template = cv2.fastNlMeansDenoising(template_ori,None,10,7,21)

        # Apply blur to reduce noise
        template = cv2.GaussianBlur(template_ori, (5, 5), 0)

        # thresh, template = cv2.threshold(template, 200, 255, cv2.THRESH_BINARY)

        # Detect edges
        template_edges = cv2.Canny(template, 100, 200, 10)

        # ------------------</Party sign>-------------------------------

        # ------------------<BRISK descriptors>---------------------------

        # Create the BRISK descriptor detector
        detector = cv2.BRISK_create(10, 1)

        # Compute descriptors and keypoints in image and template
        kp1, des1 = detector.detectAndCompute(gray, None)
        kp2, des2 = detector.detectAndCompute(template, None)

        # Create the matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the descriptors
        matches = bf.match(des1, des2)
        # Sort matches by distances
        matches = sorted(matches, key=lambda x: x.distance)
        # Compute the distances for matches
        distances = [match.distance for match in matches]

        # Min distance
        min_dist = min(distances)
        # Average distance
        avg_dist = sum(distances) / len(distances)

        # Define tolerance
        min_tolerance = 10

        # Compute the new min distance
        min_dist = min_dist or avg_dist * 1.0 / min_tolerance

        # Select the good matches based on the min distance
        good_matches = [match for match in matches if match.distance <= min_dist * min_tolerance]

        # Get matched points in the image and the symbol
        ballot_matched_points = np.array([kp1[match.queryIdx].pt for match in good_matches])
        party_matched_points = np.array([kp2[match.trainIdx].pt for match in good_matches])

        # ------------------<Find homography>---------------------------
        # Find homography
        homography, h_mask = cv2.findHomography(party_matched_points, ballot_matched_points, cv2.RANSAC, 2.0)

        h, w = template_ori.shape[0:2]
        sh, sw = img.shape[0:2]

        pts = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)

        # Perspective transformation using homography
        dst = cv2.perspectiveTransform(pts.reshape(1, -1, 2), homography).reshape(-1, 2)
        # print(dst)
        # Draw lines in the image
        img_0 = cv2.polylines(img_0, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # ------------------</Find homography>---------------------------

        # im3 = cv2.drawMatches(img, kp1, template, kp2, good_matches, None, flags=2)

        # ------------------</BRISK descriptors>---------------------------

        # Resize the image
        im3 = cv2.resize(img_0, (0, 0), fx=0.25, fy=0.25)

        # -------------------<Image display processing>-------------------
        # This is for Tkinter gui
        template_ori = Image.fromarray(template_ori)
        template_edges = Image.fromarray(template_edges)

        template_ori = ImageTk.PhotoImage(template_ori)
        template_edges = ImageTk.PhotoImage(template_edges)

        # -------------------</Image display processing>-------------------

        # Update panels
        if self.panelA is None or self.panelB is None:

            self.panelA = Label(self.topframe, image=template_ori)
            self.panelA.image = template_ori
            self.panelA.pack(side='left', padx=10, pady=10)
            self.panelA.grid(row=0, column=0, padx=10, pady=10)

            self.panelB = Label(self.topframe, image=template_edges)
            self.panelB.image = template_edges
            self.panelB.pack(side='left', padx=10, pady=10)
            self.panelB.grid(row=0, column=1, padx=10, pady=10)


        else:
            # panelA.grid_forget()
            # panelB.grid_forget()
            self.panelA.configure(image=template_ori)
            self.panelB.configure(image='')
            self.panelB.configure(image=template_edges)
            self.panelA.image = template_ori
            self.panelB.image = template_edges

        # Show the image
        # cv2.imshow('Win', im3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def detectPartyVote(self,img,img_0,path,mutex,arr,pref_votes):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Reduce noise
        gray_erode = cv2.fastNlMeansDenoising(gray, None, 18, 7, 21)
        # Reduce noise using Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Erode the image
        gray_erode = cv2.erode(gray_erode.copy(), np.ones((5, 5), np.uint8), iterations=1)
        # Detect edges
        cannyout = cv2.Canny(gray_erode, 50, 100, 5)

        # Find contours
        im2, contours, hierarchy = cv2.findContours(cannyout.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select contours with convexity and four point
        contours = [contour for contour in contours if not cv2.isContourConvex(contour) and len(
            cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4]

        sum_x = 0
        count = 0

        dst_centroids = []

        if len(path) > 0:
            for file in os.listdir(path):
                count += 1
                if file.endswith('.jpg'):
                    filepath = path + '/' + file
                    if len(filepath) > 0:
                        # ------------------<Party sign>-------------------------------

                        template_ori = cv2.imread(filepath, 0)
                        # template = cv2.bilateralFilter(template_ori,9,75,75)
                        # template = cv2.medianBlur(template_ori,5)
                        # template = cv2.fastNlMeansDenoising(template_ori,None,10,7,21)
                        template = cv2.GaussianBlur(template_ori, (5, 5), 0)
                        # thresh, template = cv2.threshold(template, 200, 255, cv2.THRESH_BINARY)

                        # ------------------</Party sign>-------------------------------

                        # ------------------<BRISK descriptors>---------------------------

                        detector = cv2.BRISK_create(10, 1)

                        kp1, des1 = detector.detectAndCompute(gray, None)
                        kp2, des2 = detector.detectAndCompute(template, None)
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                        matches = bf.match(des1, des2)
                        matches = sorted(matches, key=lambda x: x.distance)
                        distances = [match.distance for match in matches]

                        min_dist = min(distances)
                        avg_dist = sum(distances) / len(distances)

                        min_tolerance = 10

                        min_dist = min_dist or avg_dist * 1.0 / min_tolerance

                        good_matches = [match for match in matches if match.distance <= min_dist * min_tolerance]

                        ballot_matched_points = np.array([kp1[match.queryIdx].pt for match in good_matches])
                        party_matched_points = np.array([kp2[match.trainIdx].pt for match in good_matches])

                        # ------------------<Find homography>---------------------------
                        homography, h_mask = cv2.findHomography(party_matched_points, ballot_matched_points, cv2.RANSAC,
                                                                2.0)

                        h, w = template_ori.shape[0:2]
                        sh, sw = img.shape[0:2]

                        pts = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)
                        dst = cv2.perspectiveTransform(pts.reshape(1, -1, 2), homography).reshape(-1, 2)

                        img_0 = cv2.polylines(img_0, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                        # ------------------</Find homography>---------------------------

                        # im3 = cv2.drawMatches(img, kp1, template, kp2, good_matches, None, flags=2)

                        # ------------------</BRISK descriptors>---------------------------

                        template_ori = Image.fromarray(template_ori)
                        # template_edges = Image.fromarray(template_edges)

                        template_ori = ImageTk.PhotoImage(template_ori)
                        # template_edges = ImageTk.PhotoImage(template_edges)

                        cent_x = ((dst[1][0] + dst[0][0]) / 2 + (dst[2][0] + dst[3][0]) / 2) / 2
                        cent_y = ((dst[1][1] + dst[0][1]) / 2 + (dst[2][1] + dst[3][1]) / 2) / 2

                        dst_centroids.append([template_ori, cent_x, cent_y,str(file)])

                        sum_x += (dst[1][0] + dst[2][0]) / 2
                        # print(str(cent_x) + ',' + str(cent_y))

        dst_centroids = sorted(dst_centroids,key=lambda x:x[2])
        avg_x = sum_x / (count - 1)
        # print(avg_x)
        mc = []
        rect = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cx > avg_x:
                    # print(str((dst[1][1] + dst[2][1]) / 2) + ',' + str(cy))
                    mc.append(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    rect.append([x, y, w, h])
                    cv2.rectangle(img_0, (x, y), (x + w, y + h), (0, 0, 255), 3)

        count = 0
        t_im = None
        t_vote = None
        voted = False
        i = 0
        for x, y, w, h in rect:
            cx = x + w / 2
            cy = y + h / 2

            rectangle = img[y:y + h, x:x + w]
            rect_gray = cv2.cvtColor(rectangle, cv2.COLOR_BGR2GRAY)
            rect_dilate = cv2.fastNlMeansDenoising(rect_gray, None, 18, 7, 21)
            rect_dilate = cv2.erode(rect_dilate.copy(), np.ones((5, 5), np.uint8), iterations=1)
            rect_canny = cv2.Canny(rect_dilate, 50, 100, 5)

            im, cnts, hier = cv2.findContours(rect_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im = rectangle.copy()

            voted = False
            for cnt in cnts:
                hull = cv2.convexHull(cnt, returnPoints=False)
                defects = cv2.convexityDefects(cnt, hull)

                if defects is not None and len(defects) > 5:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        cv2.line(im, start, end, [0, 255, 0], 2)
                        cv2.circle(im, far, 5, [0, 0, 255], -1)
                    voted = True

            if voted:
                count += 1
                # For Tkinter
                t_vote = Image.fromarray(rectangle)
                t_vote = ImageTk.PhotoImage(t_vote)
            # print(voted)
            # print(count)

            # cv2.drawContours(im,cnts,-1,(0,0,255),3)

            # cv2.imshow('box'+str(i),im)
            i += 1

        if count == 1:
            print('Voted:' + str(count))
            min_cent = dst_centroids[0]
            t_im = dst_centroids[0][0]
            minimum_dist = np.sqrt(np.power((dst_centroids[0][1] - cx), 2) + np.power((dst_centroids[0][2] - cy), 2))
            i=0
            for temp_image, x, y,file in dst_centroids:

                d = np.sqrt(np.power((x - cx), 2) + np.power((y - cy), 2))
                if d < minimum_dist:
                    minimum_dist = d
                    t_im = temp_image
                    min_cent = dst_centroids[i]
                i+=1

            print(min_cent)
            arr[min_cent[3]][1]+=1

            mutex.acquire()
            for a in pref_votes:
                if str(a) in arr[min_cent[3]][0]:
                    arr[min_cent[3]][0][str(a)] += 1
                else:
                    arr[min_cent[3]][0][str(a)] = 1
            # self.prg_bar.step(increment)

            mutex.release()

            print(arr)
            if self.panelA is None or self.panelB is None:

                panelA = Label(self.topframe, image=t_im)
                panelA.image = t_im
                panelA.pack(padx=10, pady=10)
                panelA.grid(row=0, column=0, padx=10, pady=10)

                panelB = Label(self.topframe, image=t_vote)
                panelB.image = t_vote
                panelB.pack(padx=10, pady=10)
                panelB.grid(row=0, column=1, padx=10, pady=10)


            else:
                self.panelA.configure(image=t_im)
                self.panelB.configure(image=t_vote)
                self.panelA.image = t_im
                self.panelB.image = t_vote


        else:
            print('invalid vote')


        # self.prg_bar.step(increment)
        # cv2.imshow('Win', im3)
        # print(minimum)
        # im3 = cv2.resize(img_0, (0, 0), fx=0.25, fy=0.25)
        # cv2.imshow('Win', im3)