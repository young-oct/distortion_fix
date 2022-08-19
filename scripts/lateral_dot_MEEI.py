# -*- coding: utf-8 -*-
# @Time    : 2022-08-18 09:25
# @Author  : young wang
# @FileName: lateral_dot_MEEI.py
# @Software: PyCharm

from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
import cv2 as cv
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking, mip_stack
import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file, pre_volume, \
    clean_small_object, obtain_inner_edge
import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,
                                white_tophat, disk, black_tophat, square, skeletonize)
from natsort import natsorted
import discorpy.prep.preprocessing as prep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post


def get_feature_map(img, low_ratio=0.3, high_ratio=3):
    h, w = img.shape
    fig_dpi = 100
    fig, ax = plt.subplots(1, 1,
                           figsize=(h / fig_dpi, w / fig_dpi), dpi=fig_dpi)

    label_image = label(img)
    ax.imshow(np.zeros_like(img), 'gray')

    for region in regionprops(label_image):
        up_sze = high_ratio * dot_size
        lw_sze = low_ratio * dot_size

        if lw_sze < region.area < up_sze:
            y, x = region.centroid
            circle = mpatches.Circle((x, y),
                                     radius=3,
                                     fill=True,
                                     edgecolor='white',
                                     facecolor='white',
                                     linewidth=1)
            ax.add_patch(circle)
    ax.set_axis_off()
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    return fig2array(fig)

def fig2array(fig):
    """adapted from: https://stackoverflow.com/questions/21939658/"""
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    return img[:, :, 0]


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    dset_lst = ['../data/MEEI/dot/*.oct']
    dataset = []

    for i in range(len(dset_lst)):
        data_sets = natsorted(glob.glob(dset_lst[i]))
        data = load_from_oct_file(data_sets[0])
        dataset.append(data)

    data = load_from_oct_file(data_sets[0])
    idx = surface_index(data)[-1][-1]
    p_factor = 0.3
    vmin, vmax = p_factor * 255, 255

    img_list, tit_list = [], []

    img_ori = mip_stack(data, index=idx, thickness=idx)
    img_list.append(img_ori)
    tit_list.append('original image')

    img_norm = prep.normalization_fft(img_ori, sigma=0.5)
    img_list.append(img_norm)
    tit_list.append('denoised image')

    threshold = prep.calculate_threshold(img_norm, bgr="dark", snr=0.05)
    img_bin = prep.binarization(img_norm, ratio=0.05, thres=threshold)
    img_list.append(img_bin)
    tit_list.append('binary image')

    img_cls = opening(img_bin, disk(3))

    # mask to break down the connected pixels
    mask = erosion(img_cls, np.ones((21, 1)))
    mask = dilation(mask, np.ones((1, 3)))

    img_cls = cv.bitwise_xor(img_cls, mask)

    img_list.append(img_cls)
    tit_list.append('segmentation image')

    (dot_size, dot_dist) = prep.calc_size_distance(img_cls)

    # create label map of the initial image
    label_image = label(img_cls)

    pt_lst = []
    low_ratio,high_ratio  = 0.5, 3
    for region in regionprops(label_image):
        # take regions with large enough areas
        up_sze = high_ratio * dot_size
        lw_sze = low_ratio * dot_size

        if lw_sze < region.area < up_sze:

            y, x = region.centroid
            pt_lst.append((x, y))
        else:
            pass

    img_fea = get_feature_map(img_cls,low_ratio=low_ratio,high_ratio=high_ratio)

    img_list.append(img_fea)
    tit_list.append('dot feature')

    proc_img_list = []
    pro_title_list = []

    (dot_size, dot_dist) = prep.calc_size_distance(img_fea)

    # Calculate the slopes of horizontal lines and vertical lines.
    hor_slope = prep.calc_hor_slope(img_fea)
    ver_slope = prep.calc_ver_slope(img_fea)

    # Group points into lines
    list_hor_lines = prep.group_dots_hor_lines(img_fea, hor_slope, dot_dist, ratio=0.5,
                                               num_dot_miss=15, accepted_ratio=0.65)
    list_ver_lines = prep.group_dots_ver_lines(img_fea, ver_slope, dot_dist, ratio=0.5,
                                               num_dot_miss=15, accepted_ratio=0.65)
    # Remove outliners
    list_hor_lines = prep.remove_residual_dots_hor(list_hor_lines, hor_slope,
                                                   residual=2.0)
    list_ver_lines = prep.remove_residual_dots_ver(list_ver_lines, ver_slope,
                                                   residual=2.0)

    img_list.append(img_fea)
    tit_list.append('grouped line')

    # Optional: for checking perspective distortion
    (xcen_tmp, ycen_tmp) = proc.find_cod_bailey(list_hor_lines, list_ver_lines)
    list_hor_coef = proc._para_fit_hor(list_hor_lines, xcen_tmp, ycen_tmp)[0]
    list_ver_coef = proc._para_fit_ver(list_ver_lines, xcen_tmp, ycen_tmp)[0]

    pers_title = ['horizontal', 'vertical']
    fig, ax = plt.subplots(1, 2, figsize=(16, 9),
                           constrained_layout=True)
    ax[0].plot(list_hor_coef[:, 2], list_hor_coef[:, 0], "-o")
    ax[0].plot(list_ver_coef[:, 2], list_ver_coef[:, 0], "-o")

    ax[0].set_xlabel('c-coefficient', fontweight='bold')
    ax[0].set_ylabel('a-coefficient', fontweight='bold')

    ax[1].plot(list_hor_coef[:, 2], -list_hor_coef[:, 1], "-o")
    ax[1].plot(list_ver_coef[:, 2], list_ver_coef[:, 1], "-o")

    ax[1].set_xlabel('c-coefficient', fontweight='bold')
    ax[1].set_ylabel('b-coefficient', fontweight='bold')
    fig.suptitle('perspective distortion')
    plt.show()
    #
    num_coef = 5
    list_hor_lines1, list_ver_lines1 = proc.regenerate_grid_points_parabola(
        list_hor_lines, list_ver_lines, perspective=True)
    (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines1, list_ver_lines1)
    list_fact = proc.calc_coef_backward(list_hor_lines1, list_ver_lines1,
                                        xcenter, ycenter, num_coef)

    # Regenerate the lines without perspective correction for later use.
    list_hor_lines2, list_ver_lines2 = proc.regenerate_grid_points_parabola(
        list_hor_lines, list_ver_lines, perspective=False)

    # Unwarp lines using the backward model:
    list_uhor_lines = post.unwarp_line_backward(list_hor_lines2, xcenter, ycenter, list_fact)
    list_uver_lines = post.unwarp_line_backward(list_ver_lines2, xcenter, ycenter, list_fact)

    # Unwarp the image
    img_rad = post.unwarp_image_backward(img_fea, xcenter, ycenter, list_fact)

    img_list.append(img_rad)
    tit_list.append('radial')

    # Generate source points and target points to calculate coefficients of a perspective model
    source_points, target_points = proc.generate_source_target_perspective_points(list_uhor_lines, list_uver_lines,
                                                                                  equal_dist=True, scale="max",
                                                                                  optimizing=True)
    # Calculate perspective coefficients:
    pers_coef = proc.calc_perspective_coefficients(source_points,
                                                   target_points,
                                                   mapping="backward")
    img_pers = post.correct_perspective_image(img_rad, pers_coef, order=3)

    img_list.append(img_pers)
    tit_list.append('radial + perspective')

    c_num = np.ceil(len(img_list) / 2)
    fig, axs = plt.subplots(2, int(c_num), figsize=(16, 9),
                            constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, tit_list)):
        ax.imshow(image, 'gray', vmin=np.min(image), vmax=np.max(image))
        ax.set_title(title)
        ax.set_axis_off()
        if n == len(tit_list) - 5:
            for pts in pt_lst:
                x, y = pts
                circle = mpatches.Circle((x, y),
                                         radius=3,
                                         fill=True,
                                         edgecolor='red',
                                         linewidth=1)
                ax.add_patch(circle)

        elif n == len(tit_list) - 3:
            for line in list_hor_lines:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=1)
            for line in list_ver_lines:
                ax.plot(line[:, 1], line[:, 0], '--o', markersize=1)
        #

        else:
            pass
    plt.show()
