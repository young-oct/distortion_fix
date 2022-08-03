# -*- coding: utf-8 -*-
# @Time    : 2022-08-01 09:11
# @Author  : young wang
# @FileName: lateral_fix_01.py
# @Software: PyCharm

import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking
from tools.pos_proc import export_map

import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk, black_tophat, square, skeletonize)

from natsort import natsorted
from skimage.morphology import disk, dilation, square, erosion, binary_erosion, binary_dilation, \
    binary_closing, binary_opening, closing
import discorpy.prep.preprocessing as prep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/2022.08.01/original/*.oct'))

    data = []
    for i in range(len(data_sets)):
        data.append(load_from_oct_file(data_sets[i]))

    p_factor = 0.5

    # for i in range(len(data)):
    # 0: dot, 1: square, 2: circle
    volume = data[-1]

    # access the frame index of where axial location of the checkerboard
    index = surface_index(volume)[-1][-1]
    pad = 5
    stack = volume[:, :, int(index - pad):int(index)]
    top_slice = np.amax(stack, axis=2)

    # de-speckling for better feature extraction
    top_slice = despecking(top_slice, sigma=1, size=3)
    vmin, vmax = int(p_factor * 255), 255

    top_slice = np.where(top_slice <= vmin, vmin, top_slice)
    # create binary image of the top surface
    bi_img = prep.normalization_fft(top_slice, sigma=5)
    bi_img = prep.binarization(bi_img)
    # bi_img = despecking(bi_img, sigma=5, size=3)

    # bi_img = closing(bi_img, square(5))
    # Calculate the median dot size and distance between them.

    (dot_size, dot_dist) = prep.calc_size_distance(bi_img)
    # Remove non-dot objects
    s_img = prep.select_dots_based_size(bi_img, dot_size, ratio=0.75)
    s_img = prep.select_dots_based_ratio(s_img, ratio=0.75)

    img_list = [top_slice, bi_img, s_img]
    #
    # # Calculate the slopes of horizontal lines and vertical lines.
    #
    hor_slope = prep.calc_hor_slope(s_img)
    ver_slope = prep.calc_ver_slope(s_img)
    # #
    # #Group points into lines
    list_hor_lines0 = prep.group_dots_hor_lines(s_img, hor_slope, dot_dist, accepted_ratio=0.3, num_dot_miss=5)
    list_ver_lines0 = prep.group_dots_ver_lines(s_img, ver_slope, dot_dist, accepted_ratio=0.3, num_dot_miss=5)
    #
    # Optional: remove horizontal outliners
    list_hor_lines0 = prep.remove_residual_dots_hor(list_hor_lines0, hor_slope)
    # Optional: remove vertical outliners
    list_ver_lines0 = prep.remove_residual_dots_ver(list_ver_lines0, ver_slope)

    title_lst = ['original image', 'binary image', 'segmented image']

    img_list = [top_slice, bi_img, s_img]
    vmin, vmax = int(p_factor * 255), 255
    fig, axs = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_lst)):
        ax.imshow(image, 'gray', vmin=np.min(image), vmax=np.max(image))
        if n == 2:
            for (hline, vline) in zip(list_hor_lines0, list_ver_lines0):
                ax.plot(hline[:, 1], hline[:, 0], '--o', markersize=1)
                ax.plot(vline[:, 1], vline[:, 0], '--o', markersize=1)

        ax.set_title(title)
        ax.set_axis_off()
    plt.show()
    #
    (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines0, list_ver_lines0)
    (xcenter, ycenter) = proc.find_cod_fine(list_hor_lines0,list_ver_lines0,xcenter,
                                            ycenter,dot_dist)
    # #
    # # Calculate coefficients of the correction model
    coe_num = 3
    list_fact = proc.calc_coef_backward(list_hor_lines0, list_ver_lines0,
                                        xcenter, ycenter, coe_num)
    #
    list_uhor_lines = post.unwarp_line_backward(list_hor_lines0, xcenter, ycenter,
                                                list_fact)
    #
    list_uver_lines = post.unwarp_line_backward(list_ver_lines0, xcenter, ycenter,
                                                list_fact)
    #
    cs_img = post.unwarp_image_backward(s_img, xcenter, ycenter, list_fact)

    d_img = cs_img - s_img
    img_list1 = [s_img, cs_img, d_img]
    title_lst1 = ['original image', 'corrected image', 'difference image']

    fig, axs = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list1, title_lst1)):

        ax.imshow(image, 'gray', vmin=np.min(image), vmax=np.max(image))
        ax.set_title(title)
        ax.set_axis_off()

    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list1, title_lst1)):

        if n == 0:
            for (hline, vline) in zip(list_hor_lines0, list_ver_lines0):
                ax.plot(hline[:, 1], hline[:, 0], '--o', markersize=1)
                ax.plot(vline[:, 1], vline[:, 0], '--o', markersize=1)
                # ax.imshow(s_img, 'gray')
        else:
            for (hline, vline) in zip(list_uhor_lines, list_uver_lines):
                ax.plot(hline[:, 1], hline[:, 0], '--o', markersize=1)
                ax.plot(vline[:, 1], vline[:, 0], '--o', markersize=1)
                # ax.imshow(cs_img, 'gray')

        ax.set_title(title)
        # ax.set_axis_off()
        ax.set_xlim(0,512)
        ax.set_ylim(0,512)

    plt.show()
