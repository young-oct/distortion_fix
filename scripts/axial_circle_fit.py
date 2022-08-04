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
    filter_mask, circle_fit, slice_index,index_mid
import matplotlib
from tools.plot import angle_est, heatmap,linear_fit_plot



def origin_distance(x, y, origin):

    xc, yc = origin[0], origin[1]

    x_mid = index_mid(x)
    y_mid = index_mid(y)

    dis = np.sqrt((yc - y_mid) ** 2 + (xc - x_mid) ** 2)

    return dis


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

    xz_center_dis = []
    yz_center_dis = []
    # for i in range(2):

    for i in range(len(data_sets)):
        # for j in range(len(data_sets)):
        data = load_from_oct_file(data_sets[i], clean=False)
        vmin, vmax = int(p_factor[i] * 255), 255

        xz_mask = np.zeros_like(data)

        # perform points extraction in the xz direction
        for j in range(data.shape[0]):
            xz_mask[j, :, :] = filter_mask(data[i, :, :], vmin=vmin, vmax=vmax)
        # perform points extraction in the yz direction
        for k in range(data.shape[1]):
            xz_mask[:, k, :] = filter_mask(data[:, k, :], vmin=vmin, vmax=vmax)

        xz_pts = surface_index(xz_mask, shift)

        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        idx = 256
        ax = fig.add_subplot(221)

        slc = xz_mask[idx, :, :].T

        ax.imshow(slc, cmap='gray', vmin=vmin, vmax=vmax)

        xz_slc = slice_index(slc, shift)
        x, y = zip(*xz_slc)
        ax.plot(x, y, linewidth=2, alpha=0.8, color='r')
        mid_pt_xz = index_mid(y)
        ax.axhline(y=mid_pt_xz, color='yellow', linestyle='--', linewidth=1)

        ax.set_title('slice %d from the xz direction' % idx)

        ax = fig.add_subplot(222)

        # estimating circle of this slice
        est_cir = circle_fit(xz_slc)
        radius, origin = est_cir.radius, est_cir.origin
        ax_lim = max(abs(origin[0]), abs(origin[1]))

        # estimating the distance from origin to the middle
        # point on the curve in the xz direction
        xz_dis = origin_distance(x, y, origin)
        xz_center_dis.append((mid_pt_xz, xz_dis))

        est_cir.plot(ax)

        ax.plot(x, y, linewidth=5, alpha=0.8, color='black', label='actual points')
        angle_est(x, y, origin, radius, ax)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=3, labelspacing=0.05, fontsize = 10)

        scale = 1
        ax.set_ylim(int((ax_lim + radius) * scale), -int((ax_lim + radius) * scale))
        ax.set_xlim(-int((ax_lim + radius) * scale), int((ax_lim + radius) * scale))

        ax = fig.add_subplot(223)
        slc = xz_mask[:, idx, :].T

        ax.imshow(slc, cmap='gray', vmin=vmin, vmax=vmax)

        xz_slc = slice_index(slc, shift)
        x, y = zip(*xz_slc)
        ax.plot(x, y, linewidth=2, alpha=0.8, color='r')
        mid_pt_yz = index_mid(y)

        ax.axhline(y=mid_pt_yz, color='yellow', linestyle='--', linewidth=1)
        ax.set_title('slice %d from the yz direction' % idx)

        ax = fig.add_subplot(224)
        # estimating circle of this slice
        est_cir = circle_fit(xz_slc)
        radius, origin = est_cir.radius, est_cir.origin
        ax_lim = max(abs(origin[0]), abs(origin[1]))
        # kw = dict(size=75, unit="points", text=r"$60°$")

        # am6 = plot_angle(ax, (2.0, 0), 60, textposition="inside", **kw)
        est_cir.plot(ax)

        # estimating the distance from origin to the middle
        # point on the curve in the xz direction
        yz_dis = origin_distance(x, y, origin)
        yz_center_dis.append((mid_pt_yz, yz_dis))

        ax.plot(x, y, linewidth=5, alpha=0.8, color='black', label='actual points')
        angle_est(x, y, origin, radius, ax)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=3, labelspacing=0.05, fontsize = 10)

        ax.set_ylim(int((ax_lim + radius) * scale), -int((ax_lim + radius) * scale))
        ax.set_xlim(-int((ax_lim + radius) * scale), int((ax_lim + radius) * scale))

        z_idx = np.mean((mid_pt_xz, mid_pt_yz))

        # alternative approach with fitting a plane
        # construct ideal plane

        # method II
        # ideal_plane = pyrsc.Plane()
        # pts = np.asarray(xz_pts)
        # best_eq, best_inliers = ideal_plane.fit(pts, 0.01)
        #
        # z_idx = - best_eq[3]

        fig.suptitle('index at %d plane' % z_idx)

        plt.show()
        print('done with %d out of %d' % (int(i + 1), len(data_sets)))


    fig,ax = plt.subplots(1, 2,figsize=(16, 9), constrained_layout=True)
    linear_fit_plot(xz_center_dis,ax[0] ,'xz plane')
    linear_fit_plot(yz_center_dis,ax[1] ,'yz plane')

    fig.suptitle('2D extracted distance(origin to curve mid-point) versus z-depth')

    plt.show()


