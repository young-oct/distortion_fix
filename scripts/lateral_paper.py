# -*- coding: utf-8 -*-
# @Time    : 2022-08-04 23:13
# @Author  : young wang
# @FileName: lateral_paper.py
# @Software: PyCharm


import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure
from skimage import restoration
import cv2 as cv
from copy import deepcopy

from skimage.feature import corner_harris, corner_subpix, corner_peaks
import matplotlib
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

from skimage import feature
from tools.pos_proc import convert


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 15,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )



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

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    pad = 5

    volume = data[0]
    # access the frame index of where axial location of the checkerboard
    f_idx = max_slice(volume)
    # stack = volume[:, :, 0:int(f_idx + pad)]
    stack = volume[:, :, 0:int(f_idx)]
    index = int(330 - f_idx)

    top_slice = np.amax(stack, axis=2)


    ax.imshow(top_slice, 'gray', vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    ax.set_title('slice axial index: %d' % index)

    fig.suptitle('original grayscale images', fontsize=16)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    img = exposure.equalize_adapthist(top_slice, clip_limit=0.1)
    from skimage import feature
    from skimage import filters

    img = opening(img, square(9))
    img = median_filter(img, 7)

    edges1 = feature.canny(img, sigma=3)

    ax.imshow(edges1, 'gray')
    ax.set_axis_off()
    ax.set_title('slice axial index: %d' % index)
    plt.tight_layout()
    plt.show()

    img = edges1.astype(np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    radius = 15
    ratio = 0.25
    sen = 0.015
    nosie_le = 1.75

    # binary_map = prep.binarization(img)
    #
    img = lprep.convert_chessboard_to_linepattern(img)

    plt.imshow(img, 'gray')
    plt.show()
    #
    # # Calculate slope and distance between lines
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(img, radius=radius, sensitive=sen)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(img, radius=radius, sensitive=sen)
    #
    # # Extract reference-points
    list_points_hor_lines = lprep.get_cross_points_hor_lines(img, slope_ver, dist_ver,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr="bright", radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)
    #
    list_points_ver_lines = lprep.get_cross_points_ver_lines(img, slope_hor, dist_hor,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr="bright", radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)
    #
    # ax.imshow(img, 'gray')
    # ax.set_axis_off()
    # # ax.set_title('slice axial index: %d' % slx[i][0])
    # #
    # for pts in list_points_hor_lines:
    #     ax.plot(pts[1], pts[0], '--o', markersize=5, color='red')
    # for pts in list_points_ver_lines:
    #     ax.plot(pts[1], pts[0], '--o', markersize=5, color='blue')
    #
    # fig.suptitle('method II: Discorpy method')
    # plt.tight_layout()
    # plt.show()
    #
    # #Group points into lines
    # # Group points into lines
    list_hor_lines = prep.group_dots_hor_lines(list_points_hor_lines, slope_hor, dist_hor,
                                               ratio=0.2, num_dot_miss=2, accepted_ratio=0.65)
    list_ver_lines = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver,
                                               ratio=0.2, num_dot_miss=2, accepted_ratio=0.65)
    #
    list_hor_lines = prep.remove_residual_dots_hor(list_hor_lines, slope_hor, 2)
    list_ver_lines = prep.remove_residual_dots_ver(list_ver_lines, slope_ver, 2)
    print(len(list_hor_lines))
    print(len(list_ver_lines))

    for line in list_hor_lines:
        plt.plot(line[:, 1], line[:, 0], '--o', markersize=5)
    for line in list_ver_lines:
        plt.plot(line[:, 1], line[:, 0], '--o', markersize=5)
    plt.imshow(img, 'gray')
    plt.show()
    #
    num_coef = 5
    (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines, list_ver_lines)
    # # # # Calculate radial distortion coefficients
    list_fact = proc.calc_coef_backward(list_hor_lines, list_ver_lines, xcenter,
                                        ycenter, num_coef)

    list_uhor_lines = post.unwarp_line_backward(list_hor_lines, xcenter, ycenter, list_fact)
    list_uver_lines = post.unwarp_line_backward(list_ver_lines, xcenter, ycenter, list_fact)

    for line in list_uhor_lines:
        plt.plot(line[:, 1], line[:, 0], '--o', markersize=5)
    for line in list_uver_lines:
        plt.plot(line[:, 1], line[:, 0], '--o', markersize=5)
    plt.imshow(img, 'gray')
    plt.show()
