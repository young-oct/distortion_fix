# -*- coding: utf-8 -*-
# @Time    : 2022-07-14 12:11
# @Author  : young wang
# @FileName: depth verfication.py
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

def pre_precessing(slice,vmin, vmax):
    mask = binary_mask(slice,vmin, vmax )
    mask = ndimage.median_filter(mask, size=3)
    mask = ndimage.gaussian_filter(mask, sigma=0.2)

    return mask
if __name__ == '__main__':

    data = glob.glob('../data/2022.07.12_1mm(3dprint)/trial 2/*.oct')

    # data_decon = load_from_oct_file(data[0])
    data_ori = load_from_oct_file(data[-1])

    index = max_slice(data_ori)
    p_factor = 0.55
    vmin, vmax = int(255 * p_factor), 255

    top, middle , bottom = index, int(index + 5), int(index + 5)
    top_slice = pre_precessing(data_ori[:,:,top], vmin, vmax)
    mid_slice = pre_precessing(data_ori[:,:,middle], vmin, vmax)
    bot_slice = pre_precessing(data_ori[:,:,bottom], vmin, vmax)

    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    ax[0].imshow(top_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[0].set_axis_off()
    ax[0].set_title('top', size=20)

    ax[1].imshow(mid_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[1].set_axis_off()
    ax[1].set_title('middle', size=20)

    ax[2].imshow(bot_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[2].set_axis_off()
    ax[2].set_title('bottom', size=20)

    plt.tight_layout()
    plt.show()

    # mask = binary_mask(mip_slice,vmin, vmax )
    # mask = ndimage.median_filter(mask, size=3)
    # mask = ndimage.gaussian_filter(mask, sigma=0.2)
