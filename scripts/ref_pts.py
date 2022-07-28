# -*- coding: utf-8 -*-
# @Time    : 2022-07-18 21:11
# @Author  : young wang
# @FileName: ref_pts.py
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
import discorpy.losa.loadersaver as io
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post
from skimage.morphology import disk, dilation, square, erosion, binary_erosion, binary_dilation, \
    binary_closing, binary_opening, closing


def locate_points(image, radius=15, ratio=0.5, sen=0.05, nosie_le=1.5):
    binary_map = prep.binarization(image)

    img = lprep.convert_chessboard_to_linepattern(binary_map)

    # Calculate slope and distance between lines
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(img, radius=radius, sensitive=sen)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(img, radius=radius, sensitive=sen)

    # Extract reference-points
    list_points_hor_lines = lprep.get_cross_points_hor_lines(img, slope_ver, dist_ver,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr="bright", radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)

    list_points_ver_lines = lprep.get_cross_points_ver_lines(img, slope_hor, dist_hor,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr="bright", radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)
    return list_points_hor_lines, list_points_ver_lines


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
        index = int(330 - f_idx)

        top_slice = np.amax(stack, axis=2)

        r_no = i // 3  # get row number of the plot from reminder of division
        c_no = i % 3  # row number of the plot
        ax[r_no, c_no].imshow(top_slice, 'gray', vmin=vmin, vmax=vmax)
        ax[r_no, c_no].set_axis_off()
        ax[r_no, c_no].set_title('slice axial index: %d' % index)
        check_board.append((index, top_slice))

    fig.suptitle('original grayscale images', fontsize=16)
    plt.tight_layout()
    plt.show()

    slx = []
    fig, ax = plt.subplots(3, 3, figsize=(16, 9))
    for i in range(len(check_board)):
        index = check_board[i][0]
        slice = check_board[i][-1]

        slice = exposure.equalize_adapthist(slice, clip_limit=0.1)
        slice = gaussian_filter(slice, sigma=1)
        slice = median_filter(slice, size=5)

        background = restoration.rolling_ball(slice, radius=50)
        gray = slice - background

        top = 3
        bot = 100 - top

        p2, p98 = np.percentile(gray, (top, bot))
        gray = exposure.rescale_intensity(gray, in_range=(p2, p98))
        slx.append((index, gray))

        r_no = i // 3  # get row number of the plot from reminder of division
        c_no = i % 3  # row number of the plot
        ax[r_no, c_no].imshow(gray, 'gray')
        ax[r_no, c_no].set_axis_off()
        ax[r_no, c_no].set_title('slice axial index: %d' % index)
    plt.tight_layout()
    plt.show()

    # method one: corner detection
    map_one = []
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
            r = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
            if r <= 220:

                ax[r_no, c_no].plot(x, y, color='cyan', marker='o',
                                    linestyle='None', markersize=3)
                # save checkboard coordinates list in to x,y,z
                map_one.append((coords[j, 1], coords[j, 0], slx[i][0]))
            else:
                pass
    fig.suptitle('method I: harris corner detection')
    plt.tight_layout()
    plt.show()

    # method one: external libaraiy

    map_two = []
    fig, ax = plt.subplots(3, 3, figsize=(16, 9))
    for i in range(len(slx)):

        _, vline = locate_points(slx[0][-1], radius=25, ratio=0.5, sen=0.05, nosie_le=2.5)
        #
        r_no = i // 3  # get row number of the plot from reminder of division
        c_no = i % 3  # row number of the plot
        #
        ax[r_no, c_no].imshow(slx[i][-1], 'gray')
        ax[r_no, c_no].set_axis_off()
        ax[r_no, c_no].set_title('slice axial index: %d' % slx[i][0])
        #
        #     # # remove some out of bounds points
        for j in range(vline.shape[0]):
            x, y = vline[j, 1], vline[j, 0]
            r = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
            if r <= 220:
                ax[r_no, c_no].plot(x, y, color='red', marker='o',
                                    linestyle='None', markersize=3)
            #             # save checkboard coordinates list in to x,y,z
                map_two.append((vline[j, 1], vline[j, 0], slx[i][0]))
            else:
                pass
    fig.suptitle('method II: Discorpy method')
    plt.tight_layout()
    plt.show()
    #
