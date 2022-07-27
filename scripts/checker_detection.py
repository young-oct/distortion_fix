# -*- coding: utf-8 -*-
# @Time    : 2022-07-27 11:29
# @Author  : young wang
# @FileName: checker_detection.py
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

    # img = check_board[0][1]
    # binary_map = prep.binarization(img, thres=200)
    # plt.imshow(binary_map)
    # plt.show()

    #
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

    img = check_board[0][-1]
    from skimage.morphology import disk, dilation,square,erosion,binary_erosion,binary_dilation,\
        binary_closing,binary_opening
    # img = dilation(img, disk(3))
    # img = median_filter(img, size=10)

    binary_map = prep.binarization(img, thres=vmin)
    img = closing(binary_map,disk(3))
    slice = gaussian_filter(slice, sigma=2)
    slice = median_filter(slice, size=5)
    # img = gaussian_filter(img, sigma=0.5)

    #
    # plt.imshow(img)
    # plt.show()

    mat1 = lprep.convert_chessboard_to_linepattern(img)
    plt.imshow(mat1)
    plt.show()

    # Calculate slope and distance between lines
    radius,ratio = 15, 0.5
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(mat1, radius=radius, sensitive=0.25)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(mat1, radius=radius, sensitive=0.25)
    print("Horizontal slope: ", slope_hor, " Distance: ", dist_hor)
    print("Vertical slope: ", slope_ver, " Distance: ", dist_ver)

    # list_points_hor_lines = lprep.get_cross_points_hor_lines(mat1, slope_ver, dist_ver,
    #                                                          ratio=0.3, norm=True, offset=0,
    #                                                          bgr="bright", radius=15,
    #                                                          sensitive=1.0, denoise=True,
    #                                                          subpixel=True)

    list_points_hor_lines = lprep.get_cross_points_hor_lines(mat1, slope_ver, dist_ver,
                                                             ratio=ratio, norm=True, offset=100,
                                                             bgr="bright", radius=radius,
                                                             sensitive=1.5, denoise=True,
                                                             subpixel=True)

    list_points_ver_lines = lprep.get_cross_points_ver_lines(mat1, slope_hor, dist_hor,
                                                             ratio=ratio, norm=True, offset=100,
                                                             bgr="bright", radius=radius,
                                                             sensitive=1.5, denoise=True,
                                                             subpixel=True)

    # Group points into lines
    list_hor_lines = prep.group_dots_hor_lines(list_points_hor_lines, slope_hor, dist_hor,
                                               ratio=0.5, num_dot_miss=3, accepted_ratio=0.8)
    list_ver_lines = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver,
                                               ratio=0.5, num_dot_miss=3, accepted_ratio=0.8)

    # Remove residual dots
    list_hor_lines = prep.remove_residual_dots_hor(list_hor_lines, slope_hor, 10)
    list_ver_lines = prep.remove_residual_dots_ver(list_ver_lines, slope_ver, 10)

    height = 512
    fig, ax = plt.subplots(1,2 ,figsize = (16,9))

    for point_ver in list_points_ver_lines:
        ax[0].plot(point_ver[1], point_ver[0], '.', color='r',
                 markersize=10)
    for point_hor in list_points_hor_lines:
        ax[0].plot(point_hor[1], point_hor[0], '.', color='k',
                 markersize=10)

    ax[0].imshow(mat1)

    ax[0].set_xlim([0, 512])
    ax[0].set_ylim([0, 512])


    for point_ver in list_points_ver_lines:
        ax[1].plot(point_ver[1], point_ver[0], '.', color='r',
                 markersize=10)
    for point_hor in list_points_hor_lines:
        ax[1].plot(point_hor[1], point_hor[0], '.', color='k',
                   markersize=10)

    ax[1].imshow(img)

    ax[1].set_xlim([0, 512])
    ax[1].set_ylim([0, 512])

    plt.show()

    print('done')
