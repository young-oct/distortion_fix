# -*- coding: utf-8 -*-
# @Time    : 2022-07-22 13:44
# @Author  : young wang
# @FileName: axial_discrepancy.py
# @Software: PyCharm

import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from tools.auxiliary import load_from_oct_file
from tools.preprocessing import filter_mask, surface_index, frame_index, plane_fit
import pyransac3d as pyrsc


def heatmap(data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


if __name__ == '__main__':

    data_sets = natsorted(glob.glob('../data/1mW/flat surface(correctd)/*.oct'))

    dis_map = []
    for j in range(len(data_sets)):
        data = load_from_oct_file(data_sets[j], clean=False)
        p_factor = 0.75
        vmin, vmax = int(p_factor * 255), 255

        xz_mask = np.zeros_like(data)

        # perform points extraction in the xz direction
        for i in range(data.shape[0]):
            xz_mask[i, :, :] = filter_mask(data[i, :, :], vmin=vmin, vmax=vmax)

        xz_pts = surface_index(xz_mask)

        fig = plt.figure(figsize=(16, 9))
        idx = 256
        ax = fig.add_subplot(121)
        ax.imshow(xz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)

        xz_slc = frame_index(xz_mask, 'x', idx)
        x, y = zip(*xz_slc)
        ax.plot(y, x, linewidth=5, alpha=0.8, color='r')
        ax.set_title('slice %d from the xz direction' % idx, size=15)

        ax = fig.add_subplot(122, projection='3d')
        xp, yp, zp = zip(*xz_pts)
        ax.scatter(xp, yp, zp, s=0.1, alpha=0.1, c='r')
        ax.set_title('raw points cloud', size=15)
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

        a, b, c, d = best_eq[0], best_eq[1], -best_eq[2], best_eq[3]

        xx, yy = np.meshgrid(np.arange(0, data.shape[1], 1), np.arange(0, data.shape[1], 1))
        z_ideal = (d - a * xx - b * yy) / c
        z_mean = np.mean(z_ideal)

        fig.suptitle('index at %d plane' % z_mean, fontsize=15)

        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(16, 9))

        ax = fig.add_subplot(3, 4, 1, projection='3d')
        xp,yp,zp = zip(*xz_pts)
        ax.scatter(xp, yp, zp,s = 0.5, alpha = 0.5, c = 'r')

        idx_x = np.setdiff1d(np.arange(0,data.shape[1]),pts[:,0])
        idx_y = np.setdiff1d(np.arange(0,data.shape[2]),pts[:,1])

        surf = ax.plot_wireframe(xx, yy, z_ideal,alpha = 0.2)

        ax.set_title('raw points cloud \n'
                     '& ideal plane',size = 15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0, data.shape[0]])
        ax.set_ylim([0, data.shape[1]])
        z_low, z_high = int(z_mean - 30), int(z_mean + 30)
        ax.set_zlim([z_low, z_high])

        ax = fig.add_subplot(3, 4, 2, projection='3d')
        ax.set_title('raw points cloud \n'
                     '& linearly fitted plane',size = 15)

        l_plane = plane_fit(xz_pts,order=1).zc
        ax.scatter(xp, yp, zp,s = 0.5, alpha = 0.5, c = 'r')

        plane_fit(xz_pts,order=1).plot(ax,z_low,z_high)

        ax = fig.add_subplot(3, 4, 6, projection='3d')
        dl_map = l_plane - z_ideal
        surf = ax.plot_wireframe(xx, yy, dl_map,alpha = 0.2)

        ax = fig.add_subplot(3, 4, 10)
        im, cbar = heatmap(dl_map, ax=ax,
                           cmap="hot", cbarlabel='depth variation')

        ax = fig.add_subplot(3, 4, 3, projection='3d')
        ax.set_title('raw points cloud \n'
                     '& quadratically fitted plane',size = 15)

        q_plane = plane_fit(xz_pts,order=2).zc
        ax.scatter(xp, yp, zp,s = 0.5, alpha = 0.5, c = 'r')
        plane_fit(xz_pts,order=2).plot(ax,z_low,z_high)

        ax = fig.add_subplot(3, 4, 7, projection='3d')
        dq_map = q_plane - z_ideal
        surf = ax.plot_wireframe(xx, yy, dq_map,alpha = 0.2)

        ax = fig.add_subplot(3, 4, 11)
        im, cbar = heatmap(dq_map, ax=ax,
                           cmap="hot", cbarlabel='depth variation')

        ax = fig.add_subplot(3, 4, 4, projection='3d')
        ax.set_title('raw points cloud \n'
                     '& cubically fitted plane',size = 15)

        c_plane = plane_fit(xz_pts,order=3).zc
        ax.scatter(xp, yp, zp,s = 0.5, alpha = 0.5, c = 'r')
        plane_fit(xz_pts,order=3).plot(ax,z_low,z_high)

        ax = fig.add_subplot(3, 4, 8, projection='3d')
        dc_map = c_plane - z_ideal
        surf = ax.plot_wireframe(xx, yy, dc_map,alpha = 0.2)

        ax = fig.add_subplot(3, 4, 12)
        im, cbar = heatmap(dc_map, ax=ax,
                           cmap="hot", cbarlabel='depth variation')

        fig.suptitle('index at %d plane' % z_mean, fontsize=15)
        plt.tight_layout()
        plt.show()

        # you can export between linear, quadratic, cubic interpretation map
        dis_map.append((z_mean,dc_map))
        print('index at %d plane with linear plane has std %.2f' % (z_mean, np.std(dl_map)))
        print('index at %d plane with quadratic plane has std %.2f' % (z_mean, np.std(dq_map)))
        print('index at %d plane with cubic plane has std %.2f' % (z_mean,np.std(dc_map)))
        print('done with %d out of %d' % (int(j+1), len(data_sets)))