# -*- coding: utf-8 -*-
# @Time    : 2022-08-04 18:44
# @Author  : young wang
# @FileName: radial_model.py
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


def apply_radial_model(img, coefficient):
    k0, k1, k2 = coefficient[0], coefficient[1], coefficient[2]

    height, width = img.shape
    # construct meshgrid for interpolation mapping
    xx, yy = np.meshgrid(np.float32(np.arange(height)),
                         np.float32(np.arange(width)))

    # set the distortion center to be the mid point
    x_c, y_c = width / 2, height / 2
    xx -= x_c
    yy -= y_c

    # # scale the grid for radius calculation
    xx /= x_c
    yy /= y_c

    radius = np.sqrt(xx ** 2 + yy ** 2)  # distance from the center of image
    # for i, factor in enumerate(coefficient):
    # f = np.sum(np.asarray(
    #         [factor * radius ** i for i, factor in enumerate(coefficient)]), axis=0)

    f = k0 + k1 * radius + k2 * radius ** 2

    # 1 + k1 * radius + k2 * radius ** 2 + k3 * radius ** 3  # radial distortion model
    # apply the model
    xx = xx * f
    yy = yy * f
    # # reset all the shifting
    xx = xx * x_c + x_c
    yy = yy * y_c + y_c

    img_dist_idx = map_coordinates(img, [xx.ravel(), yy.ravel()])

    img_dis = np.reshape(img_dist_idx, img.shape)

    return img_dis


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 15,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    img = plt.imread('../validation/checkerboard.png')
    if img.ndim == 3:
        img = img[:, :, 0]
    else:
        pass

    dis_coes = [1, 0.25, 0.025]

    title_lst = ['ground truth', 'applied known distortion image']
    dis_img = apply_radial_model(img, dis_coes)
    from copy import deepcopy

    orginal = deepcopy(img)
    dis_image = deepcopy(dis_img)

    img_list = [img, dis_img]

    fig, axs = plt.subplots(1, len(title_lst), figsize=(16, 9), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_lst)):
        ax.imshow(image, 'gray')
        ax.set_title(title)
        ax.set_axis_off()
    plt.show()

    dis_img = convert(dis_img, 0, 255, np.float64)
    #
    num_coef = 5  # Number of polynomial coefficients
    (height, width) = dis_img.shape

    dis_img = lprep.convert_chessboard_to_linepattern(dis_img)

    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(dis_img, radius=10, sensitive=0.5)

    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(dis_img, radius=10, sensitive=0.5)

    # #
    list_points_hor_lines = lprep.get_cross_points_hor_lines(dis_img, slope_ver, dist_ver,
                                                             ratio=0.1, norm=True, offset=0,
                                                             bgr="bright", radius=10,
                                                             sensitive=1.0, denoise=True,
                                                             subpixel=True)

    list_points_ver_lines = lprep.get_cross_points_ver_lines(dis_img, slope_hor, dist_hor,
                                                             ratio=0.1, norm=True, offset=0,
                                                             bgr="bright", radius=10,
                                                             sensitive=1.0, denoise=True,
                                                             subpixel=True)

    img_list = [img, dis_img]

    fig, axs = plt.subplots(1, len(title_lst), figsize=(16, 9), constrained_layout=True)
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

    #Group points into lines
    # Group points into lines
    list_hor_lines = prep.group_dots_hor_lines(list_points_hor_lines, slope_hor, dist_hor,
                                               ratio=0.1, num_dot_miss=2, accepted_ratio=0.8)
    list_ver_lines = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver,
                                               ratio=0.1, num_dot_miss=2, accepted_ratio=0.8)



    list_hor_lines = prep.remove_residual_dots_hor(list_hor_lines, slope_hor, 2)
    list_ver_lines = prep.remove_residual_dots_ver(list_ver_lines, slope_ver, 2)

    fig, axs = plt.subplots(1, len(title_lst), figsize=(16, 9), constrained_layout=True)
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

    plt.show()
    # #
    (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines, list_ver_lines)
    # #
    # # # Calculate radial distortion coefficients
    list_fact = proc.calc_coef_backward(list_hor_lines, list_ver_lines, xcenter,
                                        ycenter, num_coef)
    # #
    list_uhor_lines = post.unwarp_line_backward(list_hor_lines, xcenter, ycenter, list_fact)
    list_uver_lines = post.unwarp_line_backward(list_ver_lines, xcenter, ycenter, list_fact)
    # #
    cor_img = post.unwarp_image_backward(dis_image, xcenter, ycenter, list_fact)
    # #
    #
    img_list2 = [orginal, dis_image, cor_img]
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
