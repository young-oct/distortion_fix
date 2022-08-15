# -*- coding: utf-8 -*-
# @Time    : 2022-08-14 15:25
# @Author  : young wang
# @FileName: FOV_square_geo.py
# @Software: PyCharm
import time
from tools.proc import despecking
from skimage import exposure
from tools.pos_proc import convert
import glob
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import circle_cut, wall_index, median_filter
import matplotlib
from tools.plot import line_fit_plot
from tools.proc import line_fit
from skimage.morphology import closing,disk,dilation,square
from natsort import natsorted
from scipy import ndimage
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)

def pre_volume(volume,low = 2, inner_radius=50, edge_radius = 240):
    high = 100 - low
    new_volume = np.zeros_like(volume)
    p_factor = np.mean(data)/np.max(data)
    vmin, vmax = int(p_factor * 255), 255

    for i in range(volume.shape[-1]):
        temp_slice = volume[:, :, i]
        temp_slice = circle_cut(temp_slice,
                                inner_radius = inner_radius,
                                edge_radius= edge_radius)

        temp = despecking(temp_slice, sigma=2, size=5)
        temp_slice = np.where(temp <= vmin, vmin, temp)

        low_p, high_p = np.percentile(temp_slice, (low, high))
        temp = exposure.rescale_intensity(temp_slice,
                                          in_range=(low_p, high_p))
        temp = closing(temp, diamond(20))
        new_volume[:, :, i] = temp

    new_volume = np.where(new_volume < np.mean(new_volume), 0, 255)

    return convert(new_volume, 0, 255, np.float64)


def clean_small_object(volume):
    new_volume = np.zeros_like(volume)
    for i in range(volume.shape[-1]):
        c_slice = volume[:,:,i]
        label_im, nb_labels = ndimage.label(c_slice)
        sizes = ndimage.sum(c_slice, label_im, range(nb_labels + 1))

        mask_size = sizes < np.max(sizes) * 0.5
        remove_pixel = mask_size[label_im]

        label_im[remove_pixel] = 0
        new_volume[:, :, i] = label_im
    return new_volume

def obtain_inner_edge(volume):

    iedge_volume = np.zeros_like(volume)
    for i in range(volume.shape[-1]):
        c_slice = volume[:,:,i]
        contours = measure.find_contours(c_slice)

        #1 is the inner edge, 0 is the outer edge
        edge_arr = np.zeros_like(c_slice)

        try:
            for j in range(len(contours[1]) - 1):
                x, y = contours[1][j]
                edge_arr[int(x), int(y)] = 255
        except:
            pass

        iedge_volume[:,:,i] = edge_arr
    return iedge_volume

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/MEEI/FOV/square/original/*.oct'))
    data = load_from_oct_file(data_sets[-1])
    start = time.perf_counter()

    volume = pre_volume(data, low = 1,inner_radius=50,
                        edge_radius = 255)

    re_volume = clean_small_object(volume)
    edge_vol = obtain_inner_edge(re_volume)

    end = time.perf_counter()
    print("Elapsed (after compilation) = %.2fs" % (end - start))

    plt.imshow(re_volume[:,:,-20])
    plt.show()

    # test = re_volume[:,:,-1]
    # contours = measure.find_contours(test)
    #
    # # 1 is the inner edge, 0 is the outer edge
    # edge_arr = np.zeros_like(test)
    #
    # try:
    #     for j in range(len(contours[1]) - 1):
    #         x, y = contours[1][j]
    #         edge_arr[int(x), int(y)] = 255
    # except:
    #     pass


    index_list = [0, 165, 229]
    title_list = ['top', 'middle', 'bottom']

    fig, axs = plt.subplots(1, len(index_list), figsize=(16, 9))
    for n, (ax, idx, title) in enumerate(zip(axs.flat, index_list, title_list)):
        temp = edge_vol[:, :, idx]
        ax.imshow(temp, 'gray')
        v_loc, h_loc = wall_index(temp, distance = 100, height = 0.6)
        vpts_list = ['pt1', 'pt2']
        hpts_list = ['pt3', 'pt4']

        for n, (vpts, vpts_t) in enumerate(zip(v_loc, vpts_list)):
            ax.plot(vpts[0], vpts[1], 'o', ms=10, c='blue')
            ax.text(vpts[0] * 1.1, vpts[1] * 0.9, vpts_t,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transData, size=25, color='red')
        for n, (hpts, vpts_t) in enumerate(zip(h_loc, hpts_list)):
            ax.plot(hpts[0], hpts[1], 'o', ms=10, c='red')
            ax.text(hpts[0] * 0.9, hpts[1] * 0.9, vpts_t,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transData, size=25, color='red')
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    depth_profile = []
    for i in range(volume.shape[-1]):
        temp = edge_vol[:, :, i]
        v_loc, h_loc = wall_index(temp, distance = 100, height = 0.6)
        depth_profile.append((i, v_loc, h_loc))

    pts1, pts2, pts3, pts4 = [], [], [], []
    for i in range(len(depth_profile)):
        idx, vpt, hpt = depth_profile[i]
        try:
            pts1.append((idx, vpt[0][-1]))
            pts2.append((idx, vpt[-1][-1]))

            pts3.append((idx, hpt[0][0]))
            pts4.append((idx, hpt[-1][0]))
        except:
            pass

    title_list = ['point 1', 'point 2', 'point 3', 'point 4']
    y_list =['y index', 'y index','x index','x index']
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    piont_list = [pts1, pts2, pts3, pts4]
    for n, (ax, pts, ylt, title) in enumerate(zip(axs.flat, piont_list, y_list,title_list)):
        line_fit_plot(pts, ylt, ax, order=1)
        ax.set_title(title)
    fig.suptitle('absolute distance map')
    plt.show()

    pt1_rel,pt2_rel,pt3_rel,pt4_rel = [], [],[],[]

    pt1_ref = pts1[0][-1]
    pt2_ref = pts2[0][-1]

    pt3_ref = pts3[0][-1]
    pt4_ref = pts4[0][-1]

    for i in range(len(pts1)):
        pt1_rel.append((i, np.abs(pts1[i][-1] - pt1_ref)))
        pt2_rel.append((i, np.abs(pts2[i][-1] - pt2_ref)))
        pt3_rel.append((i, np.abs(pts3[i][-1] - pt3_ref)))
        pt4_rel.append((i, np.abs(pts4[i][-1] - pt4_ref)))

    fig, axs = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    piont_list_rel = [pt1_rel, pt2_rel, pt3_rel, pt4_rel]
    for n, (ax, pts, ylt, title) in enumerate(zip(axs.flat, piont_list_rel, y_list,title_list)):
        line_fit_plot(pts, ylt, ax, order=1)
        ax.set_title(title)
    fig.suptitle('relative distance map')
    plt.show()

    print(len(pts1) / volume.shape[-1])
