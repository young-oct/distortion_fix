# -*- coding: utf-8 -*-
# @Time    : 2022-07-14 11:34
# @Author  : young wang
# @FileName: grid_method.py
# @Software: PyCharm

import glob
import numpy as np
import cv2 as cv
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

def find_dots(image):
    target = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    detected_circles = cv.HoughCircles(target,
                                        cv.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=1, maxRadius=40)
    cir_loc = []
    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, = pt[0], pt[1]
            cir_loc.append((a,b))

    return cir_loc

if __name__ == '__main__':
    images = glob.glob('../cal map/*.png')

    # Read image.
    undis = cv.imread(images[0])
    dis = cv.imread(images[-1])

    un_loc = find_dots(undis)
    dis_loc = find_dots(dis)

    fig, ax = plt.subplots(1,3, figsize = (16,9))
    ax[0].imshow(dis)
    ax[0].set_title('distorted dot map', size=20)
    # ax[0].set_axis_off()

    ax[1].imshow(undis)
    ax[1].set_title('undistorted dot map', size=20)
    # ax[1].set_axis_off()

    ax[2].imshow(undis)
    ax[2].set_title('distortion difference', size=20)
    for dot in dis_loc:
        circ = Circle(dot, 5, edgecolor='r', fill=True, linewidth=1, facecolor='r')
        ax[2].add_patch(circ)

    plt.tight_layout()
    plt.show()
