# -*- coding: utf-8 -*-
# @Time    : 2022-08-04 18:01
# @Author  : young wang
# @FileName: lateral_fix_update.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import matplotlib
from tools.pos_proc import convert
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post


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


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 15,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )
    data_sets = natsorted(glob.glob('../data/2022.08.01/validation/*.oct'))
    volume = load_from_oct_file(data_sets[0])
    # for i in range(len(data_sets)):
    #     data.append(load_from_oct_file(data_sets[i]))

    p_factor = 0.3

    # for i in range(len(data)):
    # 0: dot, 1: square, 2: circle
    # volume = data[-1]

    # access the frame index of where axial location of the checkerboard
    index = surface_index(volume)[-1][-1]
    pad = 25
    stack = volume[:, :, int(index - pad):int(index)]
    top_slice = np.amax(stack, axis=2)

    # # de-speckling for better feature extraction
    # bi_img = opening(top_slice, square(3))
    # bi_img = dilation(top_slice,disk(3))

    bi_img = median_filter(top_slice, size= 11)

    # top_slice = despecking(top_slice, sigma=1, size=3)
    vmin, vmax = int(p_factor * 255), 255

    bi_img = np.where(bi_img <= vmin, vmin, bi_img)
    # create binary image of the top surface
    bi_img = prep.normalization_fft(bi_img, sigma=5)
    bi_img = prep.binarization(bi_img)
    # bi_img = median_filter(bi_img, size= 3)

    bi_img = dilation(bi_img,disk(4))

    # bi_img = dilation(bi_img,disk(3))
    # bi_img = binary_dilation(bi_img,square(9, dtype=bool))
    # top_slice = despecking(top_slice, sigma=0.1, size=3)
    dis_img = lprep.convert_chessboard_to_linepattern(bi_img)
    img = deepcopy(top_slice)
    # bi_img = cv.filter2D(src=bi_img, ddepth=-1, kernel=kernel)
    img_list = [top_slice,bi_img, dis_img]

    title_lst = ['original image','binary image','line image']

    # img_list = [top_slice,bi_img]
    vmin, vmax = int(p_factor * 255), 255
    fig, axs = plt.subplots(1, len(title_lst), figsize=(16, 9),constrained_layout = True)

    for n, (ax, image,title) in enumerate(zip(axs.flat, img_list,title_lst)):
        ax.imshow(image, 'gray', vmin=np.min(image), vmax=np.max(image))

        ax.set_title(title)
        ax.set_axis_off()
    plt.show()
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(dis_img, radius=20, sensitive=0.2)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(dis_img, radius=20, sensitive=0.2)
    #
    # #
    list_points_hor_lines = lprep.get_cross_points_hor_lines(dis_img, slope_ver, dist_ver,
                                                             ratio=0.05, norm=True, offset=0,
                                                             bgr="bright", radius=20,
                                                             sensitive=2, denoise=True,
                                                             subpixel=True)

    list_points_ver_lines = lprep.get_cross_points_ver_lines(dis_img, slope_hor, dist_hor,
                                                             ratio=0.05, norm=True, offset=0,
                                                             bgr="bright", radius=20,
                                                             sensitive=2, denoise=True,
                                                             subpixel=True)
    #
    img_list = [img, dis_img]
    #
    fig, axs = plt.subplots(1, len(img_list), figsize=(16, 9), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_lst)):
        ax.imshow(image, 'gray')
        ax.set_title(title)
        ax.set_axis_off()
        if n == 1:
            for pts in list_points_hor_lines:
                ax.plot(pts[1], pts[0], '--o', markersize=5, color = 'red')
            for pts in list_points_ver_lines:
                ax.plot(pts[1], pts[0], '--o', markersize=5, color='blue')

        else:
            pass
    plt.show()
    #
    # #Group points into lines
    # Group points into lines
    list_hor_lines = prep.group_dots_hor_lines(list_points_hor_lines, slope_hor, dist_hor,
                                               ratio=0.2, num_dot_miss=5, accepted_ratio=0.4)
    list_ver_lines = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver,
                                               ratio=0.2, num_dot_miss=5, accepted_ratio=0.4)

    #
    #
    list_hor_lines = prep.remove_residual_dots_hor(list_hor_lines, slope_hor, 2)
    list_ver_lines = prep.remove_residual_dots_ver(list_ver_lines, slope_ver, 2)

    fig, axs = plt.subplots(1, len(img_list), figsize=(16, 9), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_lst)):
        ax.imshow(image, 'gray')

        if n == 1:

            for line in list_hor_lines:
                ax.plot(line[:, 1], line[:,0], '--o', markersize=5)
            for line in list_ver_lines:
                ax.plot(line[:, 1], line[:,0], '--o', markersize=5)

        else:
            pass

        ax.set_title(title)
        ax.set_axis_off()

    # plt.show()
    num_coef  = 3
    (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines, list_ver_lines)


    # # #
    #
    print(len(list_hor_lines))
    print(len(list_ver_lines))
    #
    # # # # # Calculate radial distortion coefficients
    list_fact = proc.calc_coef_backward(list_hor_lines, list_ver_lines, xcenter,
                                        ycenter, num_coef)

    # # #
    # list_fact = [11e-01,  2.86548772e-04, 4.32253192e-12]

    list_uhor_lines = post.unwarp_line_backward(list_hor_lines, xcenter, ycenter, list_fact)
    list_uver_lines = post.unwarp_line_backward(list_ver_lines, xcenter, ycenter, list_fact)
    # # #
    cor_img = post.unwarp_image_backward(img, xcenter, ycenter, list_fact)
    # # #
    # #
    img_list2 = [img, bi_img, cor_img]
    title_lst2 = ['ground truth', 'applied known distortion image', 'corrected image']
    fig, axs = plt.subplots(1, len(title_lst2), figsize=(16, 9), constrained_layout=True)
    #
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list2, title_lst2)):
        ax.imshow(image, 'gray')

        if n == 1:

            for line in list_hor_lines:
                ax.plot(line[:, 1], line[:,0], '--o', markersize=5)
            for line in list_ver_lines:
                ax.plot(line[:, 1], line[:,0], '--o', markersize=5)

        elif n== 2:
            for line in list_uhor_lines:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=5)
            for line in list_uver_lines:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=5)

        else:
            pass
        ax.set_title(title)
        ax.set_axis_off()
    #
    plt.show()
    #
    # plt.imshow(cor_img-img)
    # plt.show()