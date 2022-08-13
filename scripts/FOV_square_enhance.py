# -*- coding: utf-8 -*-
# @Time    : 2022-08-13 15:03
# @Author  : young wang
# @FileName: FOV_square_enhance.py
# @Software: PyCharm

from skimage import filters
from tools.proc import despecking
from skimage import exposure
from tools.pos_proc import convert
import cv2 as cv
import glob
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import circle_cut, wall_index,median_filter
import matplotlib
from tools.plot import line_fit_plot
from tools.proc import line_fit
from skimage.morphology import closing,disk
from natsort import natsorted

def pre_volume(volume, p_factor = 0.55, low = 2, edge_radius = 240):
    high = 100 - low
    new_volume = np.zeros(volume.shape)
    vmin, vmax = int(p_factor * 255), 255

    for i in range(volume.shape[-1]):
        temp_slice = volume[:, :, i]
        temp_slice = circle_cut(temp_slice, inner_radius=40, edge_radius= edge_radius)

        temp = despecking(temp_slice, sigma=3, size=5)
        temp_slice = np.where(temp < vmin, vmin, temp)

        temp_slice = median_filter(temp_slice, size=5)
        temp_slice = closing(temp_slice, disk(5))

        low_p, high_p = np.percentile(temp_slice, (low, high))
        temp = exposure.rescale_intensity(temp_slice,
                                          in_range=(low_p, high_p))

        new_volume[:, :, i] = median_filter(temp, size=5)

    return convert(new_volume, 0, 255, np.float64)


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/MEEI/FOV/circle/original/*.oct'))
    data = load_from_oct_file(data_sets[-1])
    volume = pre_volume(data, p_factor=0.55,
                        low = 2,
                        edge_radius = 235)

    index_list = [0, 165, 229]
    title_list = ['top', 'middle', 'bottom']

    fig, axs = plt.subplots(1, len(index_list), figsize=(16, 9))
    for n, (ax, idx, title) in enumerate(zip(axs.flat, index_list, title_list)):
        temp = volume[:, :, idx]
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
        temp = volume[:, :, i]
        v_loc, h_loc = wall_index(temp, distance = 100, height = 0.6)
        depth_profile.append((i, v_loc, h_loc))

    pts1, pts2, pts3, pts4 = [], [], [], []
    for i in range(len(depth_profile)):
        idx, vpt, hpt = depth_profile[i]
        try:
            pts1.append((idx, vpt[0][-1]))
            pts2.append((idx, vpt[-1][-1]))
            #
            pts3.append((idx, hpt[0][0]))
            pts4.append((idx, hpt[-1][0]))
        except:
            pass
    print(len(pts1) / volume.shape[-1])

    title_list = ['point 1', 'point 2', 'point 3', 'point 4']
    y_list =['y index', 'y index','x index','x index']
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    piont_list = [pts1, pts2, pts3, pts4]
    for n, (ax, pts, ylt, title) in enumerate(zip(axs.flat, piont_list, y_list,title_list)):
        line_fit_plot(pts, ylt, ax, order=1)
        ax.set_title(title)
    fig.suptitle('absolute distance map')
    plt.show()

