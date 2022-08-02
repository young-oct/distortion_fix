# -*- coding: utf-8 -*-
# @Time    : 2022-08-02 11:27
# @Author  : young wang
# @FileName: axial_circle_fit.py
# @Software: PyCharm

import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import surface_index, frame_index, \
    filter_mask, circle_fit, slice_index
from tools.pos_proc import heatmap, export_map
import pyransac3d as pyrsc
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib


def angle_est(x, y, origin, radius, ax):
    xmin_idx, xmax_idx = np.argmin(x), np.argmax(x)
    ymin, ymax = y[xmin_idx], y[xmax_idx]
    xc, yc = origin[0], origin[1]

    ax.plot(xmin_idx, ymin, color='green', label='x1', marker='8', ms=10)
    ax.plot(xmax_idx, ymax, color='green', label='x2', marker='8', ms=10)

    angle_1 = np.degrees(np.arcsin(abs(xc - xmin_idx) / radius))
    angle_2 = np.degrees(np.arcsin(abs(xc - xmax_idx) / radius))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    textstr = '\n'.join((
        r'$\theta_2=%.2f$' % (angle_1,),
        r'$\theta_2=%.2f$' % (angle_2,)))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax.annotate('x1', xy=(xmin_idx, ymin), xycoords='data',
                xytext=(xmin_idx - radius / 4, ymin + radius / 4), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )

    ax.annotate('x2', xy=(xmax_idx, ymax), xycoords='data',
                xytext=(xmax_idx + radius / 4, ymax + radius / 4), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )
    return ax


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/1mW/flat surface/*.oct'))
    folder_path = '../data/correction map'

    p_factor = np.linspace(0.75, 0.8, len(data_sets))
    shift = 0

    for j in range(1):
        # for j in range(len(data_sets)):
        data = load_from_oct_file(data_sets[j], clean=False)
        vmin, vmax = int(p_factor[j] * 255), 255

        xz_mask = np.zeros_like(data)

        # perform points extraction in the xz direction
        for i in range(data.shape[0]):
            xz_mask[i, :, :] = filter_mask(data[i, :, :], vmin=vmin, vmax=vmax)

        xz_pts = surface_index(xz_mask, shift)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    idx = 256
    ax = fig.add_subplot(221)

    # slc = xz_mask[idx, :, :].T
    slc = xz_mask[idx, :, :].T

    ax.imshow(slc, cmap='gray', vmin=vmin, vmax=vmax)

    xz_slc = slice_index(slc, shift)
    x, y = zip(*xz_slc)
    ax.plot(x, y, linewidth=5, alpha=0.8, color='r')
    ax.set_title('slice %d from the xz direction' % idx)

    ax = fig.add_subplot(222)

    # estimating circle of this slice
    est_cir = circle_fit(xz_slc)
    radius, origin = est_cir.radius, est_cir.origin
    ax_lim = max(abs(origin[0]), abs(origin[1]))

    est_cir.plot(ax)

    ax.plot(x, y, linewidth=5, alpha=0.8, color='black', label='actual points')
    angle_est(x, y, origin, radius, ax)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    scale = 1
    ax.set_ylim(int((ax_lim + radius) * scale), -int((ax_lim + radius) * scale))
    ax.set_xlim(-int((ax_lim + radius) * scale), int((ax_lim + radius) * scale))

    ax = fig.add_subplot(223)
    slc = xz_mask[:, idx, :].T

    ax.imshow(slc, cmap='gray', vmin=vmin, vmax=vmax)

    xz_slc = slice_index(slc, shift)
    x, y = zip(*xz_slc)
    ax.plot(x, y, linewidth=5, alpha=0.8, color='r')
    ax.set_title('slice %d from the yz direction' % idx)

    ax = fig.add_subplot(224)
    # estimating circle of this slice
    est_cir = circle_fit(xz_slc)
    radius, origin = est_cir.radius, est_cir.origin
    ax_lim = max(abs(origin[0]), abs(origin[1]))

    est_cir.plot(ax)

    ax.plot(x, y, linewidth=5, alpha=0.8, color='black', label='actual points')
    angle_est(x, y, origin, radius, ax)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    ax.set_ylim(int((ax_lim + radius) * scale), -int((ax_lim + radius) * scale))
    ax.set_xlim(-int((ax_lim + radius) * scale), int((ax_lim + radius) * scale))

    # plt.tight_layout()
    plt.show()
