# -*- coding: utf-8 -*-
# @Time    : 2022-07-11 11:14
# @Author  : young wang
# @FileName: calibration.py
# @Software: PyCharm

import glob
import numpy as np
import cv2 as cv
from matplotlib.patches import Circle
from scipy import ndimage, misc

import matplotlib.pyplot as plt

from tools.auxiliary import folder_creator, arrTolist, listtoarr, load_from_oct_file


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
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv.drawChessboardCorners(gray, checked_board, corners2, ret)

        corners2 = np.squeeze(corners2)
        for corner in corners2:
            coord = (int(corner[0]), int(corner[1]))
            circ = Circle(coord, 5, edgecolor='r', fill=False, linewidth=1)
            ax.add_patch(circ)

    plt.title('screenshot demo', size=20)
    plt.axis('off')
    plt.imshow(gray, 'gray')
    plt.tight_layout()
    plt.show()

    # data_decon = load_from_oct_file(data[0])
    data_ori = load_from_oct_file(data[-1])

    index = max_slice(data_ori)

    # constructed maximum intensity projections from a stack with
    # certain thickness
    pad = 10
    mip_slice = mip_stack(data_ori, index, pad)

    med_slice = ndimage.median_filter(mip_slice, size=3)

    guassian_slice = ndimage.gaussian_filter(med_slice, sigma=0.2)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp_slice = cv.filter2D(src=guassian_slice, ddepth=-1, kernel=kernel)

    p_factor = 0.65
    vmin, vmax = int(255 * p_factor), 255

    fig, ax = plt.subplots(3, 4, figsize=(16, 9))
    ax[0,0].imshow(mip_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[0,0].set_axis_off()
    ax[0,0].set_title('mip of the original volume', size=20)

    # plot histogram to find the cut-off for thresholding
    ax[1, 0].hist(np.ravel(mip_slice), density=True)

    ax[2,0].imshow(binary_mask(mip_slice,vmin, vmax), 'gray', vmin=vmin, vmax=vmax)
    ax[2,0].set_axis_off()
    ax[2,0].set_title('binary image', size=20)

    ax[0, 1].imshow(med_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[0, 1].set_title('median filter', size=20)
    ax[0, 1].set_axis_off()

    # plot histogram to find the cut-off for thresholding
    ax[1, 1].hist(np.ravel(med_slice), density=True)

    ax[2,1].imshow(binary_mask(med_slice,vmin, vmax), 'gray', vmin=vmin, vmax=vmax)
    ax[2,1].set_axis_off()
    ax[2,1].set_title('binary image', size=20)

    ax[0, 2].imshow(guassian_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[0, 2].set_title('guassian filter', size=20)
    ax[0, 2].set_axis_off()

    # plot histogram to find the cut-off for thresholding
    ax[1, 2].hist(np.ravel(guassian_slice), density=True)

    ax[2,2].imshow(binary_mask(guassian_slice,vmin, vmax ), 'gray', vmin=vmin, vmax=vmax)
    ax[2,2].set_axis_off()
    ax[2,2].set_title('binary image', size=20)

    ax[0, 3].imshow(sharp_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[0, 3].set_title('sharp filter', size=20)
    ax[0, 3].set_axis_off()

    # plot histogram to find the cut-off for thresholding
    ax[1, 3].hist(np.ravel(sharp_slice), density=True)

    ax[2,3].imshow(binary_mask(sharp_slice,vmin, vmax), 'gray', vmin=vmin, vmax=vmax)
    ax[2,3].set_axis_off()
    ax[2,3].set_title('binary image', size=20)

    plt.tight_layout()
    plt.show()

    mask = binary_mask(mip_slice,vmin, vmax )
    mask = ndimage.median_filter(mask, size=2)
    mask = cv.filter2D(src=ndimage.gaussian_filter(mask, sigma=4), ddepth=-1, kernel=kernel)

    res = np.uint8(mask)
    ret, corners = cv.findChessboardCorners(res, checked_board,
                                            flags=cv.CALIB_CB_ADAPTIVE_THRESH +
                                                  cv.CALIB_CB_FAST_CHECK +
                                                  cv.CALIB_CB_NORMALIZE_IMAGE)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv.drawChessboardCorners(gray, checked_board, corners2, ret)

        corners2 = np.squeeze(corners2)
        for corner in corners2:
            coord = (int(corner[0]), int(corner[1]))
            circ = Circle(coord, 5, edgecolor='r', fill=True, linewidth=1, facecolor='r')
            ax.add_patch(circ)

    plt.title('OCT', size=20)
    plt.axis('off')
    plt.imshow(res, 'gray')
    plt.tight_layout()
    plt.show()
    print(ret)
