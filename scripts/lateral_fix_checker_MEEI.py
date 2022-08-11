# -*- coding: utf-8 -*-
# @Time    : 2022-08-11 17:47
# @Author  : young wang
# @FileName: lateral_fix_checker_MEEI.py
# @Software: PyCharm

import glob
import numpy as np
import copy
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import map_index,max_slice
import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk, black_tophat, square, skeletonize)
from natsort import natsorted
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post

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
    volume = load_from_oct_file(data_sets[0])

    p_factor = 0.5
    vmin, vmax = int(p_factor * 255), 255
    pad = 50
    # access the frame index of where axial location of the checkerboard
    f_idx = max_slice(volume)
    stack = volume[:, :, 0:int(f_idx + pad)]
    index = int(330 - f_idx)

    img_list = []
    title_list = []
    ori_oct = np.amax(stack, axis=2)

    img_list.append(ori_oct)
    title_list.append('original en-face image')

    data = plt.imread('../validation/checkerboard-lines-only-white-on-black.png')
    ori_img = copy.deepcopy(data[:,:,0])

    temp_img = ori_img
    img_list.append(ori_img)
    title_list.append('segmented image')
    #
    radius = 20
    ratio = 0.3
    sen = 0.35
    nosie_le = 0.5

    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(ori_img, radius=radius, sensitive=sen)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(ori_img, radius=radius, sensitive=sen)

    print(slope_hor,dist_hor)
    print(slope_ver,dist_ver)

    # # Extract reference-points
    list_points_hor_lines = lprep.get_cross_points_hor_lines(ori_img, slope_ver, dist_ver,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr='dark', radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)
    list_points_ver_lines = lprep.get_cross_points_ver_lines(ori_img, slope_hor, dist_hor,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr='dark', radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)

    # backgroud = np.zeros(ori_img.shape)
    img_list.append(ori_oct)
    title_list.append('extracted features')

    r1 ,ndm,acr= 0.3, 5, 0.5
    list_hor_lines0 = prep.group_dots_hor_lines(list_points_hor_lines, slope_hor, dist_hor,
                                                ratio=r1, num_dot_miss=ndm, accepted_ratio=acr)
    list_ver_lines0 = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver,
                                                ratio=r1, num_dot_miss=ndm, accepted_ratio=acr)

    list_hor_lines0 = prep.remove_residual_dots_hor(list_hor_lines0, slope_hor, 2.0)
    list_ver_lines0 = prep.remove_residual_dots_ver(list_ver_lines0, slope_ver, 2.0)
    #
    print(len(list_hor_lines0))
    print(len(list_ver_lines0))

    # Regenerate grid points with the correction of perspective effect.
    list_hor_lines1, list_ver_lines1 = proc.regenerate_grid_points_parabola(
        list_hor_lines0, list_ver_lines0, perspective=True)

    num_coef = 5
    (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines1, list_ver_lines1)
    # # # # Calculate radial distortion coefficients
    list_fact = proc.calc_coef_backward(list_hor_lines1, list_ver_lines1, xcenter,
                                        ycenter, num_coef)

    rc_timg = post.unwarp_image_backward(temp_img, xcenter, ycenter, list_fact)
    img_list.append(rc_timg)
    title_list.append('only correct radial distortion &\n'
                      'feature points(red)')

    # Regenerate the lines without perspective correction for later use.
    list_hor_lines2, list_ver_lines2 = proc.regenerate_grid_points_parabola(
        list_hor_lines0, list_ver_lines0, perspective=False)

    list_uhor_lines = post.unwarp_line_backward(list_hor_lines2, xcenter, ycenter, list_fact)
    list_uver_lines = post.unwarp_line_backward(list_ver_lines2, xcenter, ycenter, list_fact)

    # Generate source points and target points to calculate coefficients of a perspective model
    source_points, target_points = proc.generate_source_target_perspective_points(list_uhor_lines, list_uver_lines,
                                                                                  equal_dist=True, scale="mean",
                                                                                  optimizing=False)
    pers_coef = proc.calc_perspective_coefficients(source_points, target_points, mapping="backward")
    f_img = post.correct_perspective_image(rc_timg, pers_coef)

    img_list.append(f_img)
    title_list.append('corrct tangential & radial distortion &\n'
                      'ground truth points(green)')

    c_img, idx_map = map_index(ori_oct, xcenter, ycenter, list_fact, pers_coef)

    img_list.append(c_img)
    title_list.append('corrected OCT image')
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_list)):
        ax.imshow(image, 'gray', vmin=np.mean(image)*0.9, vmax=np.max(image))
        ax.set_title(title)
        ax.set_axis_off()

        if n == 2:
            for line in list_hor_lines2:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=4)
            for line in list_ver_lines2:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=4)
        elif n == 2:
            for line in list_hor_lines1:
                ax.plot(line[:, 1], line[:, 0], linestyle= 'solid', markersize=4)
            for line in list_ver_lines1:
                ax.plot(line[:, 1], line[:, 0], linestyle='solid', markersize=4)
        elif n == 3:
            for pt in source_points:
                ax.plot(pt[1], pt[0], 'o', markersize=3, color = 'red')

        elif n == 5:
            for pt in target_points:
                ax.plot(pt[1], pt[0], 'o', markersize=3, color = 'lawngreen')

        else:
            pass
    plt.show()

