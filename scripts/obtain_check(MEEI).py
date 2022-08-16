# -*- coding: utf-8 -*-
# @Time    : 2022-08-16 16:51
# @Author  : young wang
# @FileName: obtain_check(MEEI).py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
from natsort import natsorted

from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.losa.loadersaver as io

import discorpy.post.postprocessing as post
from skimage.morphology import square, \
    closing, dilation, erosion, disk, diamond, opening
from tools.proc import median_filter


if __name__ == '__main__':
    # termination criteria

    mat0 = io.load_image('../validation/Artboard 1.png')  # Load image
    mat0 = closing(mat0, disk(3))
    # mat0 = median_filter(mat0, 5)

    mat1 = lprep.convert_chessboard_to_linepattern(mat0)
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(mat1, radius=15, sensitive=0.5)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(mat1, radius=15, sensitive=0.5)
    print("Horizontal slope: ", slope_hor, " Distance: ", dist_hor)
    print("Vertical slope: ", slope_ver, " Distance: ", dist_ver)
    # plt.imshow(mat1, 'gray')
    # plt.show()

    # Extract reference-points
    list_points_hor_lines = lprep.get_cross_points_hor_lines(mat1, slope_ver, dist_ver,
                                                             ratio=0.2, norm=True, offset=0,
                                                             bgr="bright", radius=15,
                                                             sensitive=1.0, denoise=True,
                                                             subpixel=True)
    list_points_ver_lines = lprep.get_cross_points_ver_lines(mat1, slope_hor, dist_hor,
                                                             ratio=0.2, norm=True, offset=0,
                                                             bgr="bright", radius=15,
                                                             sensitive=1.0, denoise=True,
                                                             subpixel=True)
    # Group points into lines
    list_hor_lines = prep.group_dots_hor_lines(list_points_hor_lines, slope_hor, dist_hor,
                                               ratio=0.1, num_dot_miss=6, accepted_ratio=0.5)
    list_ver_lines = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver,
                                               ratio=0.1, num_dot_miss=6, accepted_ratio=0.5)

    # # # Remove residual dots
    list_hor_lines0 = prep.remove_residual_dots_hor(list_hor_lines, slope_hor, 2.0)
    list_ver_lines0 = prep.remove_residual_dots_ver(list_ver_lines, slope_ver, 2.0)

    plt.imshow(mat1, 'gray')
    for line in list_hor_lines:
        plt.plot(line[:, 1], line[:, 0], '--o', markersize=4)
    for line in list_ver_lines:
        plt.plot(line[:, 1], line[:, 0], '--o', markersize=4)
    plt.show()

    # Regenerate grid points after correcting the perspective effect.
    list_hor_lines1, list_ver_lines1 = proc.regenerate_grid_points_parabola(
        list_hor_lines0, list_ver_lines0, perspective=True)
    num_coef = 4

    # Calculate parameters of the radial correction model
    (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines1, list_ver_lines1)
    list_fact = proc.calc_coef_backward(list_hor_lines1, list_ver_lines1,
                                        xcenter, ycenter, num_coef)

    list_hor_lines2, list_ver_lines2 = proc.regenerate_grid_points_parabola(
        list_hor_lines0, list_ver_lines0, perspective=False)

    list_uhor_lines = post.unwarp_line_backward(list_hor_lines2, xcenter, ycenter,
                                                list_fact)
    list_uver_lines = post.unwarp_line_backward(list_ver_lines2, xcenter, ycenter,
                                                list_fact)

    plt.imshow(mat1, 'gray')
    for line in list_uhor_lines:
        plt.plot(line[:, 1], line[:, 0], '--o', markersize=4)
    for line in list_uver_lines:
        plt.plot(line[:, 1], line[:, 0], '--o', markersize=4)
    plt.show()

    mat_rad_corr = post.unwarp_image_backward(mat0, xcenter, ycenter, list_fact)

    # Generate source points and target points to calculate coefficients of a perspective model
    source_points = [[46, 126],
                     [100.5, 128.5],
                     [39.5, 161.5],
                     [98.5, 165.5]]
    # Generate undistorted points. Note that the output coordinate is in the yx-order.
    s_points, t_points = proc.generate_4_source_target_perspective_points(
        source_points, input_order="xy", scale="mean", equal_dist=True)

    # source_points, target_points = proc.generate_source_target_perspective_points(list_uhor_lines, list_uver_lines,

    pers_coef = proc.calc_perspective_coefficients(s_points, t_points, mapping="backward")
    #
    image_pers_corr = post.correct_perspective_image(mat_rad_corr, pers_coef)
    plt.imshow(image_pers_corr, 'gray')
    plt.show()
    #
    # plt.axis("off")
    # fig = plt.imshow(mat_rad_corr, 'gray',
    #                  interpolation='nearest')
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    #
    # plt.savefig('/Users/youngwang/Desktop/Screen Shot 2022-08-16 at 17.38.33.png', transparent=True,
    #             bbox_inches='tight', pad_inches=0)