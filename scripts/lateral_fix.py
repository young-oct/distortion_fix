# -*- coding: utf-8 -*-
# @Time    : 2022-07-29 15:30
# @Author  : young wang
# @FileName: lateral_fix.py
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


def locate_center(image, cut=False, radius=100, sigma=4):
    edges = feature.canny(image, sigma=sigma)
    edges = edges.astype(np.float32)

    M = cv.moments(edges)
    # calculate x,y coordinate of center
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    if cut:
        for k in range(image.shape[0]):
            for j in range(image.shape[1]):
                x = k - cX
                y = j - cY
                r = np.sqrt(x ** 2 + y ** 2)
                if r > radius:
                    image[k, j] = 0
                else:
                    pass
        return image, (cX, cY)
    else:
        return image, (cX, cY)


if __name__ == '__main__':


    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/2022.07.31/enhanced/*.oct'))

    data = []
    for i in range(len(data_sets)):
        data.append(load_from_oct_file(data_sets[i]))

    p_factor = 0.4

    volume = data[0]
    # access the frame index of where axial location of the checkerboard
    index = surface_index(volume)[-1][-1]
    stack = volume[:, :, 0:int(index)]

    top_slice = np.amax(stack, axis=2)

    # de-speckling for better feature extraction
    top_slice = despecking(top_slice, sigma=1, size=1)
    vmin, vmax = int(p_factor * 255), 255

    top_slice = np.where(top_slice <= vmin,vmin, top_slice)
    # create binary image of the top surface
    bi_img = prep.binarization(top_slice)

    title_lst = ['original image','binary image']

    img_list = [top_slice,bi_img]
    vmin, vmax = int(p_factor * 255), 255
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))

    for n, (ax, image,title) in enumerate(zip(axs.flat, img_list,title_lst)):
        ax.imshow(image, 'gray', vmin=np.min(image), vmax=np.max(image))
        ax.set_title(title)
        ax.set_axis_off()
    plt.show()


# # for ax, img in enumerate(fig.axes, img_list):
    #     ax.imshow(image, 'gray', vmin=vmin, vmax=vmax)
    #     ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    # # fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout = True)
    # ax[0].imshow(top_slice, 'gray', vmin=vmin, vmax=vmax)
    # ax[0].set_axis_off()
    # # ax.set_title('slice axial index: %d' % index)
    # # check_board.append((index, top_slice))
    # fig.suptitle('original grayscale images', fontsize=16)
    # plt.tight_layout()
    # plt.show()
    #
    # mat1 = prep.binarization(top_slice)
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    #
    # ax.imshow(mat1, 'gray')
    # ax.set_axis_off()
    # # ax.set_title('slice axial index: %d' % index)
    # # check_board.append((index, top_slice))
    # fig.suptitle('original grayscale images', fontsize=16)
    # plt.tight_layout()
    # plt.show()


    #
    # background = restoration.rolling_ball(top_slice, radius=10)
    # gray = top_slice - background
    # plt.imshow(gray, 'gray')
    # plt.show()
    # # slx = []
    # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    # for i in range(len(check_board)):
    #     index = check_board[i][0]
    #     slice = check_board[i][-1]
    #
    #     slice = exposure.equalize_adapthist(slice, clip_limit=0.1)
    #     slice = gaussian_filter(slice, sigma=1)
    #     slice = median_filter(slice, size=5)
    #
    #     background = restoration.rolling_ball(slice, radius=50)
    #     gray = slice - background
    #
    #     top = 3
    #     bot = 100 - top
    #
    #     p2, p98 = np.percentile(gray, (top, bot))
    #     gray = exposure.rescale_intensity(gray, in_range=(p2, p98))
    #     slx.append((index, gray))
    #
    #     # r_no = i // 3  # get row number of the plot from reminder of division
    #     # c_no = i % 3  # row number of the plot
    #     ax.imshow(gray, 'gray')
    #     ax.set_axis_off()
    #     ax.set_title('slice axial index: %d' % index)
    # plt.tight_layout()
    # plt.show()
    #
    # # CHECKERBOARD = (4,4)
    # # top_slice = despecking(top_slice,10,5)
    #
    # # top_slice = top_slice.astype('uint8')
    # #
    # # ret, corners = cv.findChessboardCorners(top_slice, CHECKERBOARD,
    # #                                         cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK +
    # #                                         cv.CALIB_CB_NORMALIZE_IMAGE)
    # # method one: corner detection
    # # map_one = []
    # # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    # # for i in range(len(slx)):
    # #
    # #     # you can play around with those paramaters
    # #     # k =[0,0.2] 0 gives your the sharp corners, 0.2 gives you blunt ones
    # #     # sigma is for gaussian filters
    # #     # min_distance is for the size of square
    # #     coords = corner_peaks(corner_harris(slx[i][-1], k=0.15, sigma=5.5), min_distance=15, threshold_rel=0.015)
    # #
    # #     r_no = i // 3  # get row number of the plot from reminder of division
    # #     c_no = i % 3  # row number of the plot
    # #
    # #     ax.imshow(slx[i][-1], 'gray')
    # #     ax.set_axis_off()
    # #     ax.set_title('slice axial index: %d' % slx[i][0])
    # #
    # #     # remove some out of bounds points
    # #     for j in range(coords.shape[0]):
    # #         x, y = coords[j, 1], coords[j, 0]
    # #         r = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
    # #         if r <= 150:
    # #
    # #             ax.plot(x, y, color='cyan', marker='o',
    # #                                 linestyle='None', markersize=10)
    # #             # save checkboard coordinates list in to x,y,z
    # #             map_one.append((coords[j, 1], coords[j, 0], slx[i][0]))
    # #         else:
    # #             pass
    # # fig.suptitle('method I: harris corner detection')
    # # plt.tight_layout()
    # # plt.show()
    # from copy import deepcopy
    # a = deepcopy(slx[0][-1])
    #
    # map_two = []
    # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    # for i in range(len(slx)):
    #
    #     _, vline = locate_points(a, radius=25, ratio=0.5, sen=0.1, nosie_le=5)
    #     #
    #     # r_no = i // 3  # get row number of the plot from reminder of division
    #     # c_no = i % 3  # row number of the plot
    #     #
    #     ax.imshow(slx[i][-1], 'gray')
    #     ax.set_axis_off()
    #     ax.set_title('slice axial index: %d' % slx[i][0])
    #     #
    #     #     # # remove some out of bounds points
    #     for j in range(vline.shape[0]):
    #         x, y = vline[j, 1], vline[j, 0]
    #         r = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
    #         if r <= 150:
    #             ax.plot(x, y, color='red', marker='o',
    #                                 linestyle='None', markersize=10)
    #         #             # save checkboard coordinates list in to x,y,z
    #             map_two.append((vline[j, 1], vline[j, 0], slx[i][0]))
    #         else:
    #             pass
    # fig.suptitle('method II: Discorpy method')
    # plt.tight_layout()
    # plt.show()
    #
    #
    # radius = 15
    # ratio = 0.1
    # sen = 0.05
    # nosie_le = 1.5
    #
    # binary_map = prep.binarization(a)
    # img = lprep.convert_chessboard_to_linepattern(binary_map)
    #
    # # Calculate slope and distance between lines
    # slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(img, radius=radius, sensitive=sen)
    # slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(img, radius=radius, sensitive=sen)
    #
    # # Extract reference-points
    # list_points_hor_lines = lprep.get_cross_points_hor_lines(img, slope_ver, dist_ver,
    #                                                          ratio=ratio, norm=True, offset=0,
    #                                                          bgr="bright", radius=radius,
    #                                                          sensitive=nosie_le, denoise=True,
    #                                                          subpixel=True)
    #
    # list_points_ver_lines = lprep.get_cross_points_ver_lines(img, slope_hor, dist_hor,
    #                                                          ratio=ratio, norm=True, offset=0,
    #                                                          bgr="bright", radius=radius,
    #                                                          sensitive=nosie_le, denoise=True,
    #                                                          subpixel=True)
    #
    #
    # pts = []
    # fig,ax = plt.subplots(1, 1, figsize = (16,9))
    # for j in range(list_points_ver_lines.shape[0]):
    #     x, y = list_points_ver_lines[j, 1], list_points_ver_lines[j, 0]
    #     r = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
    #     if r <= 150:
    #         ax.plot(x, y, color='red', marker='o',
    #                 linestyle='None', markersize=3)
    #         pts.append((x,y))
    #         #             # save checkboard coordinates list in to x,y,z
    #     else:
    #         pass
    #
    # temp = np.asarray(pts)
    # temp_list = [temp]
    # list_ver_lines = prep.group_dots_ver_lines(temp, slope_ver, dist_ver,
    #                                            ratio=0.05, num_dot_miss=20,
    #                                            accepted_ratio=0.9)
    # # list_ver_lines = prep.remove_residual_dots_ver(temp_list, slope_ver, 0.5)
    #
    # for line in list_ver_lines:
    #     plt.plot(line[:, 1],  line[:, 0], '-o', markersize=10)
    #
    # ax.imshow(a)
    # plt.show()
