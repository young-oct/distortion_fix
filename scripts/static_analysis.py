# -*- coding: utf-8 -*-
# @Time    : 2022-07-14 08:48
# @Author  : young wang
# @FileName: static_analysis.py
# @Software: PyCharm

import glob
import numpy as np
import cv2 as cv
from matplotlib.patches import Circle
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from tools.auxiliary import folder_creator, arrTolist, listtoarr, load_from_oct_file
from tools.dicom_converter import oct_to_dicom
from os.path import join


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def max_slice(volume):
    # take a volume and find the index of the maximum intensity
    # slice
    assert volume.ndim == 3

    slice = np.sum(volume, axis=0)
    line = np.sum(slice, axis=0)

    return np.argmax(line)


def mip_stack(volume, index, thickness):
    assert volume.ndim == 3

    low_b, high_b = int(index - thickness), int(index + thickness)

    if low_b >= 0 or high_b <= volume.shape[-1]:
        return np.amax(volume[:, :, low_b::high_b], axis=2)


def binary_mask(slice, vmin, vmax):
    ret, msk = cv.threshold(slice, vmin, vmax, cv.THRESH_BINARY)
    krn = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    return cv.bitwise_and(cv.dilate(msk, krn, iterations=100), msk)


if __name__ == '__main__':
    data = glob.glob('../sample/*.png')

    height, width = 9,6
    checked_board = (height, width)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #
    # # Initialize enpty list to accumulate coordinates
    objpoints = []  # 3d world coordinates
    imgpoints = []  # 2d image coordinates
    #
    objp = np.zeros((1,
                     checked_board[0] * checked_board[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checked_board[0], 0:checked_board[1]].T.reshape(-1, 2)

    validation = []
    for i in range(len(data)):

        image = cv.imread(data[i])
        mask = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #
        ret, corners = cv.findChessboardCorners(mask, checked_board)
        print(ret)
        #
        if ret:
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))

            # If found, add object points, image points (after refining them)
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(mask, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv.drawChessboardCorners(mask, checked_board, corners2, ret)
        #
            corners2 = np.squeeze(corners2)
            for corner in corners2:
                coord = (int(corner[0]), int(corner[1]))
                circ = Circle(coord, 5, edgecolor='r', fill=True, linewidth=1, facecolor='r')
                ax.add_patch(circ)
                ax.set_title('trial '+ str(i), size = 20)
                ax.set_axis_off()
                ax.imshow(mask,'gray')
        else:
            validation.append(data[i])
        plt.tight_layout()
        plt.show()

    #
    h, w = mask.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, mask.shape[::-1], None, None)
    #
    # undistort
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    val_image = cv.cvtColor(cv.imread(validation[0]), cv.COLOR_BGR2GRAY)
    dst = cv.remap(val_image, mapx, mapy, cv.INTER_LINEAR)

    fig,ax = plt.subplots(1,2, figsize = (16,9))
    ax[0].set_title('distorted validation image', size=25)
    ax[0].set_axis_off()
    ax[0].imshow(val_image, 'gray')

    ax[1].set_title('undistorted validation image', size=25)
    ax[1].set_axis_off()
    ax[1].imshow(dst, 'gray')

    plt.tight_layout()
    plt.show()
