# -*- coding: utf-8 -*-
# @Time    : 2022-08-16 16:51
# @Author  : young wang
# @FileName: obtain_check(MEEI).py
# @Software: PyCharm

import numpy as np
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

    mat0 = io.load_image('../validation/Artboard 1.png')  # Load image
    mat0 = closing(mat0, disk(3))

    mat1 = lprep.convert_chessboard_to_linepattern(mat0)
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(mat1, radius=15, sensitive=0.5)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(mat1, radius=15, sensitive=0.5)
    print("Horizontal slope: ", slope_hor, " Distance: ", dist_hor)
    print("Vertical slope: ", slope_ver, " Distance: ", dist_ver)

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
                                               ratio=0.1, num_dot_miss=6, accepted_ratio=0.4)
    list_ver_lines = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver,
                                               ratio=0.1, num_dot_miss=6, accepted_ratio=0.4)

    # # # Remove residual dots
    list_hor_lines0 = prep.remove_residual_dots_hor(list_hor_lines, slope_hor, 2.0)
    list_ver_lines0 = prep.remove_residual_dots_ver(list_ver_lines, slope_ver, 2.0)
    # Regenerate grid points after correcting the perspective effect.
    list_hor_lines1, list_ver_lines1 = proc.regenerate_grid_points_parabola(
        list_hor_lines0, list_ver_lines0, perspective=True)
    num_coef = 3

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

    mat_rad_corr = post.unwarp_image_backward(mat0, xcenter, ycenter, list_fact)

    # Generate undistorted points. Note that the output coordinate is in the yx-order.
    source_points, target_points = proc.generate_source_target_perspective_points(list_uhor_lines,
                                                                                  list_uver_lines,
                                                                                  equal_dist=True,
                                                                                  scale="median",
                                                                                  optimizing=True)

    pers_coef = proc.calc_perspective_coefficients(source_points, target_points, mapping="backward")

    image_pers_corr = post.correct_perspective_image(mat_rad_corr, pers_coef)

    img_list = [mat0,mat1,mat1,
                mat1,mat_rad_corr,image_pers_corr]
    title_list = ['original', 'line pattern image', 'line feature',
                  'perspective lines','radially corrected', 'radial+perspective']
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_list)):
        ax.imshow(image, 'gray', vmin=np.mean(image)*0.9, vmax=np.max(image))
        ax.set_title(title)
        ax.set_axis_off()

        if n == 2:
            for line in list_hor_lines0:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=4)
            for line in list_ver_lines0:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=4)
        elif n == 3:
            for line in list_hor_lines1:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=4)
            for line in list_ver_lines1:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=4)
        elif n == 4:
            for line in list_uhor_lines:
                ax.plot(line[:, 1], line[:, 0], linestyle= 'solid', markersize=4)
            for line in list_uver_lines:
                ax.plot(line[:, 1], line[:, 0], linestyle='solid', markersize=4)
        elif n == 5:
            for pt in target_points:
                ax.plot(pt[1], pt[0], 'o', markersize=3, color = 'lawngreen')
        else:
            pass
    plt.show()
