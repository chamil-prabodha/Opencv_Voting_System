import cv2
import numpy as np

class ImageTransform(object):

    def rotateImage(self,img_main,contour):
        # Rotate Image-------------------------------------------------------------------------
        approximation = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        h1 = np.sqrt(np.power((approximation[0][0][0] - approximation[3][0][0]), 2) + np.power(
            (approximation[0][0][1] - approximation[3][0][1]), 2))
        h2 = np.sqrt(np.power((approximation[1][0][0] - approximation[2][0][0]), 2) + np.power(
            (approximation[1][0][1] - approximation[2][0][1]), 2))

        h = (h1 + h2) / 2

        w1 = np.sqrt(np.power((approximation[0][0][0] - approximation[1][0][0]), 2) + np.power(
            (approximation[0][0][1] - approximation[1][0][1]), 2))
        w2 = np.sqrt(np.power((approximation[2][0][0] - approximation[3][0][0]), 2) + np.power(
            (approximation[2][0][1] - approximation[3][0][1]), 2))

        w = (w1 + w2) / 2

        arr = approximation

        arr = [[arr[1][0][0], arr[1][0][1]], [arr[0][0][0], arr[0][0][1]], [arr[3][0][0], arr[3][0][1]],
               [arr[2][0][0], arr[2][0][1]]]

        pts1 = np.float32(arr[:3])
        pts2 = np.float32([[0, 0], [w, 0], [w, h]])

        M = cv2.getAffineTransform(pts1, pts2)

        dest = cv2.warpAffine(img_main, M, (int(w), int(h)))
        return dest
        # /Rotate image--------------------------------------------------------------

    def get_verticle_line_angles(self,edges):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)
        good_lines = []
        for line in lines:
            for rho, theta in line:
                if np.abs(theta) > 0 and np.abs(theta) < 0.05:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    good_lines.append(theta)
                    # print(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 10000 * (-b))
                    y1 = int(y0 + 10000 * (a))
                    x2 = int(x0 - 10000 * (-b))
                    y2 = int(y0 - 10000 * (a))

        return good_lines

    def sort_corners(self,approximation):
        minx = approximation[0][0][0]
        miny = approximation[0][0][1]
        maxx = approximation[0][0][0]
        maxy = approximation[0][0][1]
        for a in approximation:
            if minx > a[0][0]:
                minx = a[0][0]
            if maxx < a[0][0]:
                maxx = a[0][0]
            if miny > a[0][1]:
                miny = a[0][1]
            if maxy < a[0][1]:
                maxy = a[0][1]
        return [[[minx, miny]], [[maxx, miny]], [[maxx, maxy]], [[minx, maxy]]]
