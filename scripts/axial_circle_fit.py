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
from tools.proc import surface_index, frame_index, plane_fit,\
    filter_mask,circle_fit
from tools.pos_proc import heatmap,export_map
import pyransac3d as pyrsc
import os
import cv2 as cv
from scipy.ndimage import median_filter,gaussian_filter
import matplotlib
from skimage import feature

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

    dis_map = []
    raw_dis_map = []
    res_error = []
    for j in range(1):
    # for j in range(len(data_sets)):
        data = load_from_oct_file(data_sets[j], clean=False)
        vmin, vmax = int(p_factor[j] * 255), 255

        xz_mask = np.zeros_like(data)

        # perform points extraction in the xz direction
        for i in range(data.shape[0]):
            xz_mask[i, :, :] = filter_mask(data[i, :, :], vmin=vmin, vmax=vmax)

        xz_pts = surface_index(xz_mask, shift)

        fig = plt.figure(figsize=(16, 9))
        idx = 256
        ax = fig.add_subplot(131)
        ax.imshow(xz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)

        xz_slc = frame_index(xz_mask, 'x', idx, shift)
        x, y = zip(*xz_slc)
        ax.plot(y, x, linewidth=5, alpha=0.8, color='r')
        ax.set_title('slice %d from the xz direction' % idx)

        ax = fig.add_subplot(132)
        # ax.imshow(xz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)

        xz_slc = frame_index(xz_mask, 'x', idx, shift)

        # estimating circle of this slice
        est_cir = circle_fit(xz_slc)
        radius, origin = est_cir.radius, est_cir.origin
        est_cir.plot(ax)

        # x, y = zip(*xz_slc)
        # ax.plot(y, x, linewidth=5, alpha=0.8, color='r')
        # ax.set_title('slice %d from the xz direction' % idx)

        ax = fig.add_subplot(133, projection='3d')
        xp, yp, zp = zip(*xz_pts)
        ax.scatter(xp, yp, zp, s=0.1, alpha=0.1, c='r')
        ax.set_title('raw points cloud')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0, data.shape[0]])
        ax.set_ylim([0, data.shape[1]])
        ax.set_zlim([0, data.shape[2]])

        # construct ideal plane
        ideal_plane = pyrsc.Plane()
        pts = np.asarray(xz_pts)

        best_eq, best_inliers = ideal_plane.fit(pts, 0.01)

        a, b, c, d = best_eq[0], best_eq[1], - best_eq[2], best_eq[3]

        xx, yy = np.meshgrid(np.arange(0, data.shape[1], 1), np.arange(0, data.shape[1], 1))
        z_ideal = (d - a * xx - b * yy) / c
        z_mean = np.mean(z_ideal)

        # obtained the raw point difference map
        raw_map = np.zeros((512, 512))
        for i in range(len(xz_pts)):
            lb, hb = z_mean * 0.5, z_mean * 1.5
            if lb <= xz_pts[i][2] <= hb:
                raw_map[xz_pts[i][0], xz_pts[i][1]] = int(xz_pts[i][2] - z_mean)
            else:
                pass

        raw_map = gaussian_filter(raw_map, sigma=4)

        fig.suptitle('index at %d plane' % z_mean)

        plt.tight_layout()
        plt.show()
