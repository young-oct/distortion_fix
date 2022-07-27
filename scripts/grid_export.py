# -*- coding: utf-8 -*-
# @Time    : 2022-07-18 21:11
# @Author  : young wang
# @FileName: grid_export.py
# @Software: PyCharm

import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.auxiliary import load_from_oct_file
from tools.preprocessing import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure
from skimage import restoration
import cv2 as cv
from skimage.feature import corner_harris, corner_subpix, corner_peaks

from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk, black_tophat, square, skeletonize)


from natsort import natsorted


if __name__ == '__main__':

    data_sets = natsorted(glob.glob('../data/1mW/1mm grid/*.oct'))

    data = []
    for i in range(len(data_sets)):
        data.append(load_from_oct_file(data_sets[i]))

    p_factor = 0.65
    vmin, vmax = int(p_factor * 255), 255
    # pad axially on the accessed stack to avoid artifacts

    fig, ax = plt.subplots(3, 3, figsize=(16, 9))

    check_board = []
    for i in range(len(data)):
        pad = int(5 + i)

        volume = data[i]
        # access the frame index of where axial location of the checkerboard
        f_idx = max_slice(volume)
        # stack = volume[:, :, 0:int(f_idx + pad)]
        stack = volume[:, :, 0:int(f_idx)]
        index = int(330-f_idx)

        top_slice = np.amax(stack, axis=2)

        r_no = i // 3  # get row number of the plot from reminder of division
        c_no = i % 3  # row number of the plot
        ax[r_no, c_no].imshow(top_slice, 'gray', vmin=vmin, vmax=vmax)
        ax[r_no, c_no].set_axis_off()
        ax[r_no, c_no].set_title('slice axial index: %d' % index)
        check_board.append((index,top_slice))

    fig.suptitle('original grayscale images', fontsize=16)
    plt.tight_layout()
    plt.show()

    slx = []
    fig, ax = plt.subplots(3, 3, figsize=(16, 9))
    for i in range(len(check_board)):
        index = check_board[i][0]
        slice = check_board[i][-1]

        slice = exposure.equalize_adapthist(slice, clip_limit=0.1)
        slice = gaussian_filter(slice,sigma=1)
        slice = median_filter(slice,size=5)

        background = restoration.rolling_ball(slice, radius=50)
        gray = slice - background

        top = 3
        bot = 100 - top

        p2, p98 = np.percentile(gray, (top, bot))
        gray = exposure.rescale_intensity(gray, in_range=(p2, p98))
        slx.append((index,gray))

        r_no = i // 3  # get row number of the plot from reminder of division
        c_no = i % 3  # row number of the plot
        ax[r_no, c_no].imshow(gray, 'gray')
        ax[r_no, c_no].set_axis_off()
        ax[r_no, c_no].set_title('slice axial index: %d' % index)
    plt.tight_layout()
    plt.show()

    cor_list = []
    fig, ax = plt.subplots(3, 3, figsize=(16, 9))
    for i in range(len(slx)):

        # you can play around with those paramaters
        # k =[0,0.2] 0 gives your the sharp corners, 0.2 gives you blunt ones
        # sigma is for gaussian filters
        # min_distance is for the size of square
        coords = corner_peaks(corner_harris(slx[i][-1], k=0.15, sigma=5.5), min_distance=15, threshold_rel=0.015)

        r_no = i // 3  # get row number of the plot from reminder of division
        c_no = i % 3  # row number of the plot

        ax[r_no, c_no].imshow(slx[i][-1], 'gray')
        ax[r_no, c_no].set_axis_off()
        ax[r_no, c_no].set_title('slice axial index: %d' % slx[i][0])

        # remove some out of bounds points
        for j in range(coords.shape[0]):
            x, y = coords[j, 1], coords[j, 0]
            r = np.sqrt((x-256)**2 + (y-256) **2)
            if r <= 230:

                ax[r_no, c_no].plot(x, y, color='cyan', marker='o',
                    linestyle='None', markersize=3)

        #save checkboard coordinates list in to x,y,z
        # cor_list.append((coords[:, 1],coords[:, 0],slx[i][0]))
            else:
                pass
    plt.tight_layout()
    plt.show()


