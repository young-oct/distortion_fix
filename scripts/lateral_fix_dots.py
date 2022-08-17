# -*- coding: utf-8 -*-
# @Time    : 2022-08-01 09:11
# @Author  : young wang
# @FileName: lateral_fix_dots.py
# @Software: PyCharm

import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking
from tools.pos_proc import export_map
from tools.pre_proc import folder_creator,load_from_oct_file
from tools.pos_proc import oct_to_dicom,imag2uint
from os.path import join
from scipy.ndimage import gaussian_filter, median_filter


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
    bi_img = opening(top_slice, square(5))
    bi_img = median_filter(bi_img, size= 3)

    # top_slice = despecking(top_slice, sigma=1, size=3)
    vmin, vmax = int(p_factor * 255), 255

    bi_img = np.where(bi_img <= vmin, vmin, bi_img)
    # create binary image of the top surface
    bi_img = prep.normalization_fft(bi_img, sigma=5)
    bi_img = prep.binarization(bi_img)
    # bi_img = despecking(bi_img, sigma=5, size=3)

    # bi_img = closing(bi_img, square(5))
    # Calculate the median dot size and distance between them.

    (dot_size, dot_dist) = prep.calc_size_distance(bi_img)
    # Remove non-dot objects
    s_img = prep.select_dots_based_size(bi_img, dot_size, ratio=0.6)
    s_img = prep.select_dots_based_ratio(s_img, ratio=0.6)

    img_list = [top_slice, bi_img, s_img]
    #
    # # Calculate the slopes of horizontal lines and vertical lines.
    #
    hor_slope = prep.calc_hor_slope(s_img)
    ver_slope = prep.calc_ver_slope(s_img)
    # # #
    # #Group points into lines
    list_hor_lines0 = prep.group_dots_hor_lines(s_img, hor_slope, dot_dist, accepted_ratio=0.2, num_dot_miss=5)
    list_ver_lines0 = prep.group_dots_ver_lines(s_img, ver_slope, dot_dist, accepted_ratio=0.2, num_dot_miss=5)
    #
    # Optional: remove horizontal outliners
    list_hor_lines0 = prep.remove_residual_dots_hor(list_hor_lines0, hor_slope)
    # Optional: remove vertical outliners
    list_ver_lines0 = prep.remove_residual_dots_ver(list_ver_lines0, ver_slope)

    title_lst = ['original image', 'binary image', 'segmented image']

    hor_coef_corr, ver_coef_corr = proc._generate_non_perspective_parabola_coef(
        list_hor_lines0, list_ver_lines0)[0:2]

    # # Regenerate grid points with the correction of perspective effect.
    # list_hor_lines1, list_ver_lines1 = proc.regenerate_grid_points_parabola(
    #     list_hor_lines0, list_ver_lines0, perspective=True)

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
    # #

    (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines0, list_ver_lines0)

    (xcenter, ycenter) = proc.find_cod_fine(list_hor_lines0,list_ver_lines0,xcenter,
                                            ycenter,dot_dist)
    # # #
    # # # Calculate coefficients of the correction model
    coe_num = 5
    list_fact = proc.calc_coef_backward(list_hor_lines0, list_ver_lines0,
                                        xcenter, ycenter, coe_num)

    # Regenerate the lines without perspective correction for later use.
    list_hor_lines2, list_ver_lines2 = proc.regenerate_grid_points_parabola(
        list_hor_lines0, list_ver_lines0, perspective=False)

    # Unwarp lines using the backward model:
    list_uhor_lines = post.unwarp_line_backward(list_hor_lines2, xcenter, ycenter, list_fact)
    list_uver_lines = post.unwarp_line_backward(list_ver_lines2, xcenter, ycenter, list_fact)
    #
    c_img = post.unwarp_image_backward(s_img, xcenter, ycenter, list_fact)

    # list_uhor_lines = post.unwarp_line_forward(list_hor_lines0, xcenter, ycenter,
    #                                             list_fact)
    # #
    # list_uver_lines = post.unwarp_line_forward(list_ver_lines0, xcenter, ycenter,
    #                                             list_fact)
    # #
    # cs_img = post.unwarp_image_forward(s_img, xcenter, ycenter, list_fact)
    #
    d_img = c_img - s_img
    img_list1 = [s_img, c_img, d_img]
    title_lst1 = ['original image', 'corrected image', 'difference image']
    #
    fig, axs = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list1, title_lst1)):

        ax.imshow(image, 'gray', vmin=np.min(image), vmax=np.max(image))
        ax.set_title(title)
        ax.set_axis_off()

    plt.show()
    #
    fig, axs = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    for n, (ax, title) in enumerate(zip(axs.flat,title_lst1)):

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

    # Generate source points and target points to calculate coefficients of a perspective model
    source_points, target_points = proc.generate_source_target_perspective_points(list_uhor_lines, list_uver_lines,
                                                                                  equal_dist=True, scale="mean",
                                                                                  optimizing=False)

    # Calculate perspective coefficients:
    pers_coef = proc.calc_perspective_coefficients(source_points, target_points, mapping="backward")
    f_img = post.correct_perspective_image(c_img, pers_coef)
    #
    # plt.imshow(f_img)
    # for (hline, vline) in zip(list_hor_lines2, list_ver_lines2):
    #     plt.plot(hline[:, 1], hline[:, 0], '--o', markersize=1)
    #     plt.plot(vline[:, 1], vline[:, 0], '--o', markersize=1)
    #
    # plt.show()

    #
    # apply correction
    val_path = '../data/2022.08.01/validation/3.oct'

    val_data = load_from_oct_file(val_path)
    plt.imshow(val_data[:, 256, :])
    plt.show()

    corr_data = np.zeros(val_data.shape)

    for i in range(corr_data.shape[-1]):

        corr_data[:,:,i ]  = post.unwarp_image_backward(val_data[:,:,i], xcenter, ycenter, list_fact)
        # corr_data[:,:,i ] = post.correct_perspective_image(temp, pers_coef)
    # #
    # export to dicom

    #create dicom stacks for comparison
    dicom_path = join('../', 'data','validation dicom')
    resolutionx, resolutiony, resolutionz = 0.034, 0.034, 0.034

    folder_creator(dicom_path)

    c_data = imag2uint(corr_data)

    file_path = 'corrected'
    f_path = join(dicom_path,file_path)
    folder_creator(f_path)

    patient_info = {'PatientName': 'validation',
                    'PatientBirthDate': '19600507',
                    'PatientSex': 'M',
                    'PatientAge': '64Y',
                    'PatientID': '202107070001',
                    'SeriesDescription': file_path,
                    'StudyDescription': 'OCT 3D'}

    oct_to_dicom(c_data, resolutionx=resolutionx,
                 resolutiony=resolutiony,resolutionz = resolutionz,
                 dicom_folder=f_path,
                 **patient_info)
    #
    print('Done creating dicom stacks for distorted validation dataset')

    # # list_hor_lines3, list_ver_lines3 = proc.generate_undistorted_perspective_lines(list_hor_lines0,
    #                                                                                list_ver_lines0,
    #                                                                                equal_dist=True, scale='mean',
    #                                                                                optimizing=True)
    #
    # fig, axs = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    # for n, (ax, title) in enumerate(zip(axs.flat,title_lst1)):
    #
    #     if n == 0:
    #         for (hline, vline) in zip(list_hor_lines1, list_ver_lines1):
    #             ax.plot(hline[:, 1], hline[:, 0], '--o', markersize=1)
    #             ax.plot(vline[:, 1], vline[:, 0], '--o', markersize=1)
    #             # ax.imshow(s_img, 'gray')
    #     else:
    #         for (hline, vline) in zip(list_hor_lines3, list_ver_lines3):
    #             ax.plot(hline[:, 1], hline[:, 0], '--o', markersize=1)
    #             ax.plot(vline[:, 1], vline[:, 0], '--o', markersize=1)
    #             # ax.imshow(cs_img, 'gray')
    #
    #     ax.set_title(title)
    #     # ax.set_axis_off()
    #     ax.set_xlim(0,512)
    #     ax.set_ylim(0,512)
    #
    # plt.show()
    #
    #
    a = post.unwarp_image_backward(top_slice,xcenter, ycenter, list_fact)
    fig,ax = plt.subplots(1,2, figsize = (16,9))
    ax[0].imshow(top_slice,'gray', vmin=np.min(top_slice), vmax=np.max(top_slice) )
    ax[1].imshow(a, 'gray', vmin=np.min(a), vmax=np.max(a))
    plt.show()