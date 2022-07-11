# -*- coding: utf-8 -*-
# @Time    : 2022-07-11 11:14
# @Author  : young wang
# @FileName: calibration.py
# @Software: PyCharm

import glob
import numpy as np
import cv2 as cv
from os.path import join
import time
from skimage import exposure

import matplotlib.pyplot as plt

from tools.auxiliary import folder_creator, arrTolist, listtoarr, load_from_oct_file

if __name__ == '__main__':

    data = glob.glob('../data/distorted/*.jpeg')
    # data = glob.glob('../data/validation/*.oct')

    data_mpi = []

    for i in range(2):
        image = load_from_oct_file(data[i])
        # create the mip image from original image
        img_mpi = np.amax(image, axis=2).astype('uint8')
        data_mpi.append(img_mpi)

    print('done')

    # Define the dimensions of checkerboard
    height,width = 4,4
    checked_board = (height, width)
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:height, 0:width].T.reshape(-1, 2)

    square_size = 0.2 # physical square size is 0.2mm
    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # pics = []
    low = 215
    # for img in data_mpi:
    tmp = np.where(data_mpi[0] <= low, 0,255).astype('uint8')
        # pics.append(tmp)
    plt.imshow(tmp,'gray', vmin = low, vmax = 255)
    plt.show()

    ret, corners = cv.findChessboardCorners(tmp, (3,3), None)
