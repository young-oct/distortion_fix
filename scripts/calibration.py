# -*- coding: utf-8 -*-
# @Time    : 2022-07-11 11:14
# @Author  : young wang
# @FileName: calibration.py
# @Software: PyCharm

import glob
import numpy as np
import cv2 as cv
from matplotlib.patches import Circle
from os.path import join
import time
from skimage import exposure

import matplotlib.pyplot as plt

from tools.auxiliary import folder_creator, arrTolist, listtoarr, load_from_oct_file

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

if __name__ == '__main__':

    # data = glob.glob('../data/distorted/*.jpeg')
    # data = glob.glob('../data/distorted/*.oct')
    # data = glob.glob('../data/distorted/*.png')
    # data = glob.glob('../data/2022.07.12_1.25mm(paper)/*.oct')
    # image = glob.glob('../data/2022.07.12_1.25mm(paper)/*.png')
    data = glob.glob('../data/2022.07.12_1mm(3dprint)/trial 2/*.oct')
    image = glob.glob('../data/2022.07.12_1mm(3dprint)/trial 2/*.png')

    # Define the dimensions of checkerboard
    height, width = 4, 4
    checked_board = (height, width)

    gray = cv.imread(image[-1])
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, checked_board, None)
    print(ret)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    # # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv.drawChessboardCorners(gray, checked_board, corners2, ret)

        corners2 = np.squeeze(corners2)
        for corner in corners2:
            coord = (int(corner[0]), int(corner[1]))
            circ = Circle(coord, 10, edgecolor='r', fill=False, linewidth=1)
            ax.add_patch(circ)

    plt.title('screenshot demo', size=20)
    plt.axis('off')
    plt.imshow(gray, 'gray')
    plt.tight_layout()
    plt.show()

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv.filter2D(src=gray, ddepth=-1, kernel=kernel)

    # load .oct files deconvolved and original files
    data_decon = load_from_oct_file(data[0])
    data_ori = load_from_oct_file(data[-1])

    # constructed maximum intensity projections of each oct volume
    #
    ori_mip = np.amax(data_ori, axis=2)
    dec_mip = np.amax(data_decon, axis=2)

    p_factor = 0.65
    vmin, vmax = int(255 * p_factor), 255

    # vmin = 63
    # vmax = 133
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    ax[0, 0].imshow(ori_mip, 'gray', vmin=vmin, vmax=vmax)
    ax[0, 0].set_axis_off()
    ax[0, 0].set_title('mip of the original volume', size=20)

    # plot histogram to find the cut-off for thresholding
    ax[1, 0].hist(np.ravel(ori_mip), density=True)

    ax[0, 1].imshow(dec_mip, 'gray', vmin=vmin, vmax=vmax)
    ax[0, 1].set_title('mip of the deconvolved volume', size=20)
    ax[0, 1].set_axis_off()

    # plot histogram to find the cut-off for thresholding
    ax[1, 1].hist(np.ravel(dec_mip), density=True)

    plt.show()

    # thresholding images to create binary maksk
    ret, msk = cv.threshold(ori_mip,200,vmax,cv.THRESH_BINARY)
    plt.imshow(msk,'gray')
    plt.show()

    # Extract chess-board
    krn = cv.getStructuringElement(cv.MORPH_RECT, (200, 200))
    dlt = cv.dilate(msk, krn, iterations=100)
    res = 255 - cv.bitwise_and(dlt, msk)
    plt.imshow(res,'gray')
    plt.show()

    # Displaying chess-board features
    res = np.uint8(res)
    ret, corners = cv.findChessboardCorners(res, (3,3),
                                             flags=cv.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv.CALIB_CB_FAST_CHECK +
                                                   cv.CALIB_CB_NORMALIZE_IMAGE)

    print(ret)

