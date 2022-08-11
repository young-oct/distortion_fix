# -*- coding: utf-8 -*-
# @Time    : 2022-08-11 15:31
# @Author  : young wang
# @FileName: lateral_fix_MEEI.py
# @Software: PyCharm


import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure
import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk, black_tophat, square, skeletonize)
from natsort import natsorted
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post
from skimage.morphology import square
from skimage import feature
from scipy.ndimage import map_coordinates
from tools.pos_proc import export_map,export_list

def map_index(img, xcenter, ycenter, radial_list, perspective_list):

    c1, c2, c3, c4, c5, c6, c7, c8 = perspective_list

    (height, width) = img.shape
    xu_list = np.arange(width) - xcenter
    yu_list = np.arange(height) - ycenter
    xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
    ru_mat = np.sqrt(xu_mat ** 2 + yu_mat ** 2)

    # apply radial model
    fact_mat = np.sum(np.asarray(
        [factor * ru_mat ** i for i, factor in enumerate(radial_list)]), axis=0)

    # shift the distortion center
    xd_mat = xcenter + fact_mat * xu_mat
    yd_mat = ycenter + fact_mat * yu_mat

    # apply  perspective model
    mat_tmp = (c7 * xd_mat + c8 * yd_mat + 1.0)
    xd_mat = (c1 * xd_mat + c2 * yd_mat + c3) / mat_tmp
    yd_mat = (c4 * xd_mat + c5 * yd_mat + c6) / mat_tmp
    xd_mat = np.float32(np.clip(xd_mat, 0, width - 1))
    yd_mat = np.float32(np.clip(yd_mat, 0, height - 1))

    indices = np.vstack((np.ndarray.flatten(yd_mat), np.ndarray.flatten(xd_mat)))

    xd_mat = xd_mat.flatten()/img.shape[0]
    yd_mat = yd_mat.flatten()/img.shape[1]

    idx_map = np.empty((xd_mat.size + yd_mat.size), dtype=xd_mat.dtype)
    idx_map[0::2] = xd_mat
    idx_map[1::2] = yd_mat

    # map img to new indices
    c_img = map_coordinates(img, indices).reshape(img.shape)
    # index normalize to [0,1] for GPU texture

    return c_img, idx_map

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 15,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/MEEI/checkerboard/*.oct'))

    # data = []
    # for i in range(len(data_sets)):
    #     data.append(load_from_oct_file(data_sets[i]))
    volume = load_from_oct_file(data_sets[0])

    p_factor = 0.8
    vmin, vmax = int(p_factor * 255), 255
    # pad axially on the accessed stack to avoid artifacts

    pad = 30
    # volume = data[0]
    # access the frame index of where axial location of the checkerboard
    f_idx = max_slice(volume)
    stack = volume[:, :, 0:int(f_idx + pad)]
    index = int(330 - f_idx)

    img_list = []
    title_list = []
    ori_img = np.amax(stack, axis=2)

    img_list.append(ori_img)

    # fig, ax = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
    # ax.imshow(ori_img, 'gray', vmin=np.mean(ori_img)*0.9, vmax=np.max(ori_img))
    # ax.set_axis_off()
    # plt.savefig('/Users/youngwang/Desktop/distortion_fix/validation/target.png',bbox_inches='tight',pad_inches = 0)
    # plt.show()
    title_list.append('original en-face image')

    temp_img = median_filter(ori_img, size=3)
    temp_img = gaussian_filter(ori_img, sigma=0.3)
    # temp_img = median_filter(temp_img, size=5)


    temp_img = exposure.equalize_adapthist(temp_img, clip_limit=0.1)
    # temp_img = opening(temp_img, disk(4))
    # img = median_filter(temp_img, 4)
    # temp_img = median_filter(temp_img, 5)


    edge_img = feature.canny(temp_img, sigma=3.1)
    edge_img = gaussian_filter(edge_img, sigma=0.1)

    img_list.append(edge_img)
    title_list.append('edge map & feature lines')

    edge_img = edge_img.astype(np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
    ax.imshow(edge_img, 'gray', vmin=np.mean(edge_img)*0.9, vmax=np.max(edge_img))
    ax.set_axis_off()
    plt.show()

    radius = 20
    ratio = 0.3
    sen = 0.35
    nosie_le = 1.25

    img_list.append(ori_img)
    title_list.append('original en-face image')

    # # Calculate slope and distance between lines
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(edge_img, radius=radius, sensitive=sen)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(edge_img, radius=radius, sensitive=sen)

    print(slope_hor,dist_hor)
    print(slope_ver,dist_ver)

    # # Extract reference-points
    list_points_hor_lines = lprep.get_cross_points_hor_lines(edge_img, slope_ver, dist_ver,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr="dark", radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)
    #
    list_points_ver_lines = lprep.get_cross_points_ver_lines(edge_img, slope_hor, dist_hor,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr="dark", radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)

    for pt in list_points_ver_lines:
        ax.plot(pt[1], pt[0], 'o', markersize=3, color='red')
    for pt in list_points_hor_lines:
        ax.plot(pt[1], pt[0], 'o', markersize=3, color='lawngreen')

    ax.imshow(edge_img, 'gray', vmin=np.mean(edge_img)*0.9, vmax=np.max(edge_img))
    ax.set_axis_off()
    plt.show()

    print(list_points_ver_lines.shape)
    #
    # Group dots into lines.
    r1 ,ndm,acr= 0.3, 5, 0.5
    list_hor_lines0 = prep.group_dots_hor_lines(list_points_hor_lines, slope_hor, dist_hor,
                                                ratio=r1, num_dot_miss=ndm, accepted_ratio=acr)
    list_ver_lines0 = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver,
                                                ratio=r1, num_dot_miss=ndm, accepted_ratio=acr)

    list_hor_lines0 = prep.remove_residual_dots_hor(list_hor_lines0, slope_hor, 5.0)
    list_ver_lines0 = prep.remove_residual_dots_ver(list_ver_lines0, slope_ver, 5.0)
    #
    print(len(list_hor_lines0))
    print(len(list_ver_lines0))

    # Regenerate grid points with the correction of perspective effect.
    list_hor_lines1, list_ver_lines1 = proc.regenerate_grid_points_parabola(
        list_hor_lines0, list_ver_lines0, perspective=True)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)

    for line in list_hor_lines0:
        ax.plot(line[:, 1], line[:, 0], linestyle= 'solid', markersize=4)
    for line in list_ver_lines0:
        ax.plot(line[:, 1], line[:, 0], linestyle='solid', markersize=4)
    ax.imshow(edge_img, 'gray', vmin=np.mean(edge_img) * 0.9, vmax=np.max(edge_img))
    ax.set_axis_off()
    plt.show()
    #
    # num_coef = 5
    # (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines1, list_ver_lines1)
    # # # # # Calculate radial distortion coefficients
    # list_fact = proc.calc_coef_backward(list_hor_lines1, list_ver_lines1, xcenter,
    #                                     ycenter, num_coef)
    #
    # rc_timg = post.unwarp_image_backward(ori_img, xcenter, ycenter, list_fact)
    # img_list.append(rc_timg)
    # title_list.append('only correct radial distortion &\n'
    #                   'feature points(red)')
    #
    # # Regenerate the lines without perspective correction for later use.
    # list_hor_lines2, list_ver_lines2 = proc.regenerate_grid_points_parabola(
    #     list_hor_lines0, list_ver_lines0, perspective=False)
    #
    # list_uhor_lines = post.unwarp_line_backward(list_hor_lines2, xcenter, ycenter, list_fact)
    # list_uver_lines = post.unwarp_line_backward(list_ver_lines2, xcenter, ycenter, list_fact)
    #
    # # Generate source points and target points to calculate coefficients of a perspective model
    # source_points, target_points = proc.generate_source_target_perspective_points(list_uhor_lines, list_uver_lines,
    #                                                                               equal_dist=True, scale="mean",
    #                                                                               optimizing=False)
    # pers_coef = proc.calc_perspective_coefficients(source_points, target_points, mapping="backward")
    # f_img = post.correct_perspective_image(rc_timg, pers_coef)
    #
    # img_list.append(f_img)
    # title_list.append('corrct tangential & radial distortion &\n'
    #                   'ground truth points(green)')
    #
    # fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    # for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_list)):
    #     ax.imshow(image, 'gray', vmin=np.mean(image)*0.9, vmax=np.max(image))
    #     ax.set_title(title)
    #     ax.set_axis_off()
    #
    #     if n == 1:
    #         for line in list_hor_lines2:
    #             ax.plot(line[:, 1], line[:, 0], '--o', markersize=4)
    #         for line in list_ver_lines2:
    #             ax.plot(line[:, 1], line[:, 0], '--o', markersize=4)
    #     elif n == 2:
    #         for line in list_hor_lines1:
    #             ax.plot(line[:, 1], line[:, 0], linestyle= 'solid', markersize=4)
    #         for line in list_ver_lines1:
    #             ax.plot(line[:, 1], line[:, 0], linestyle='solid', markersize=4)
    #     elif n == 4:
    #         for pt in source_points:
    #             ax.plot(pt[1], pt[0], 'o', markersize=3, color = 'red')
    #         for pt in target_points:
    #             ax.plot(pt[1], pt[0], 'o', markersize=3, color = 'lawngreen')
    #
    #     elif n == 5:
    #         for pt in target_points:
    #             ax.plot(pt[1], pt[0], 'o', markersize=3, color = 'lawngreen')
    #
    #     else:
    #         pass
    # plt.show()
    # #
    # # # check for the final wrapper function
    # # c_img, idx_map = map_index(ori_img, xcenter, ycenter, list_fact, pers_coef)
    # # img_list1 = [ori_img,c_img]
    # # title_list1 = ['original en-face image', 'final correction']
    # #
    # # fig, axs = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    # # for n, (ax, image, title) in enumerate(zip(axs.flat, img_list1, title_list1)):
    # #     ax.imshow(image, 'gray', vmin=np.mean(image)*0.9, vmax=np.max(image))
    # #     ax.set_title(title)
    # #     ax.set_axis_off()
    # # plt.show()
    # #
    # # bin_file = '../data/correction map/radial_correction.bin'
    # # export_list(idx_map, bin_file)
