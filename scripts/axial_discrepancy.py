# -*- coding: utf-8 -*-
# @Time    : 2022-07-22 13:44
# @Author  : young wang
# @FileName: axial_discrepancy.py
# @Software: PyCharm

import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import surface_index, frame_index, plane_fit,filter_mask
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

    data_sets = natsorted(glob.glob('../data/1mW/flat surface fixed/*.oct'))
    folder_path = '../data/correction map'

    p_factor = np.linspace(0.75, 0.8, len(data_sets))
    shift = 0

    dis_map = []
    raw_dis_map = []
    res_error = []
    # for j in range(2):
    for j in range(len(data_sets)):
        data = load_from_oct_file(data_sets[j], clean=False)
        vmin, vmax = int(p_factor[j] * 255), 255

        xz_mask = np.zeros_like(data)

        # perform points extraction in the xz direction
        for i in range(data.shape[0]):
            xz_mask[i, :, :] = filter_mask(data[i, :, :], vmin=vmin, vmax=vmax)

        xz_pts = surface_index(xz_mask, shift)

        fig = plt.figure(figsize=(16, 9))
        idx = 256
        ax = fig.add_subplot(121)
        ax.imshow(xz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)

        xz_slc = frame_index(xz_mask, 'x', idx, shift)
        x, y = zip(*xz_slc)
        ax.plot(y, x, linewidth=5, alpha=0.8, color='r')
        ax.set_title('slice %d from the xz direction' % idx)

        ax = fig.add_subplot(122, projection='3d')
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

        fig = plt.figure(figsize=(16, 9))

        ax = fig.add_subplot(3, 4, 1, projection='3d')
        xp, yp, zp = zip(*xz_pts)
        ax.scatter(xp, yp, zp, s=0.5, alpha=0.5, c='r')

        surf = ax.plot_wireframe(xx, yy, z_ideal, alpha=0.2)

        ax.set_title('raw points cloud \n'
                     '& ideal plane')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0, data.shape[0]])
        ax.set_ylim([0, data.shape[1]])
        z_low, z_high = int(z_mean - 30), int(z_mean + 30)
        ax.set_zlim([z_low, z_high])

        ax = fig.add_subplot(3, 4, 5, projection='3d')
        surf = ax.plot_wireframe(xx, yy, raw_map, alpha=0.2)

        ax = fig.add_subplot(3, 4, 9)
        im, cbar = heatmap(raw_map, ax=ax,
                           cmap="hot", cbarlabel='depth variation')

        ax = fig.add_subplot(3, 4, 2, projection='3d')
        ax.set_title('raw points cloud \n'
                     '& linearly fitted plane')

        l_plane = plane_fit(xz_pts, order=1).zc
        ax.scatter(xp, yp, zp, s=0.5, alpha=0.5, c='r')

        plane_fit(xz_pts, order=1).plot(ax, z_low, z_high)

        ax = fig.add_subplot(3, 4, 6, projection='3d')
        dl_map = l_plane - z_ideal
        surf = ax.plot_wireframe(xx, yy, dl_map, alpha=0.2)

        ax = fig.add_subplot(3, 4, 10)
        im, cbar = heatmap(dl_map.T, ax=ax,
                           cmap="hot", cbarlabel='depth variation')

        ax = fig.add_subplot(3, 4, 3, projection='3d')
        ax.set_title('raw points cloud \n'
                     '& quadratically fitted plane')

        q_plane = plane_fit(xz_pts, order=2).zc
        ax.scatter(xp, yp, zp, s=0.5, alpha=0.5, c='r')
        plane_fit(xz_pts, order=2).plot(ax, z_low, z_high)

        ax = fig.add_subplot(3, 4, 7, projection='3d')
        dq_map = q_plane - z_ideal
        surf = ax.plot_wireframe(xx, yy, dq_map, alpha=0.2)

        ax = fig.add_subplot(3, 4, 11)
        im, cbar = heatmap(dq_map.T, ax=ax,
                           cmap="hot", cbarlabel='depth variation')

        ax = fig.add_subplot(3, 4, 4, projection='3d')
        ax.set_title('raw points cloud \n'
                     '& cubically fitted plane')

        c_plane = plane_fit(xz_pts, order=3).zc
        ax.scatter(xp, yp, zp, s=0.5, alpha=0.5, c='r')
        plane_fit(xz_pts, order=3).plot(ax, z_low, z_high)

        ax = fig.add_subplot(3, 4, 8, projection='3d')
        dc_map = c_plane - z_ideal
        surf = ax.plot_wireframe(xx, yy, dc_map, alpha=0.2)

        ax = fig.add_subplot(3, 4, 12)
        im, cbar = heatmap(dc_map.T, ax=ax,
                           cmap="hot", cbarlabel='depth variation')

        fig.suptitle('index at %d plane' % z_mean)
        plt.tight_layout()
        plt.show()


        # you can export between linear, quadratic, cubic interpretation map
        dis_map.append((z_mean, dc_map))
        # export the raw point difference map
        raw_dis_map.append((z_mean, raw_map))
        res_error.append((z_mean,np.std(raw_map)))

        # temp_name = data_sets[j].split('/')[-1]
        # file_name = temp_name.split('.')[0]
        # file_path = (os.path.join(folder_path, '%s.bin' % file_name))

        # export_map(raw_map, file_path)

        print('done with %d out of %d' % (int(j + 1), len(data_sets)))

    # export the orientation map
    orientation = np.ones((512, 512))
    for i in range(orientation.shape[0]):
        for j in range(orientation.shape[1]):
            if 0 <= i <= 256 and 0 <= j <= 256:
                orientation[i, j] = 0
            else:
                pass

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    surf = ax.plot_wireframe(xx, yy, orientation, alpha=0.2)
    ax = fig.add_subplot(1, 2, 2)
    im, cbar = heatmap(orientation, ax=ax,
                       cmap="hot", cbarlabel='depth variation')
    fig.suptitle('orientation map', fontsize=15)
    plt.tight_layout()
    plt.show()
    #
    fig,ax = plt.subplots(2,2, figsize=(16, 9), constrained_layout=True)
    depth, er_map = zip(*raw_dis_map)
    dep,error = zip(*res_error)

    ax[0, 0].imshow(er_map[0])
    ax[0,0].set_axis_off()
    #
    ax[1,0].plot(dep, error, marker = 'o',ms = 10)
    ax[1,0].set_ylabel('axial depth [pixel]', size = 20)
    ax[1,0].set_ylabel('standard deviation', size = 20)
    ax[1,0].set_title('without trimming the edge', size=20)
    ax[1,0].set_ylim(bottom=0, top=1)

    radius = 180

    error_crop = []
    for i in range(len(er_map)):

        img = raw_dis_map[i][-1]
        edges = feature.canny(raw_dis_map[i][-1], sigma=1)
        edges = edges.astype(np.float32)

        M = cv.moments(edges)
        # calculate x,y coordinate of center
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    #
        for k in range(img.shape[0]):
            for j in range(img.shape[1]):
                x = k - cX
                y = j - cY
                r = np.sqrt(x**2 + y **2)
                if r > radius:
                    img[k,j] = 0
                else:
                    pass
        error_crop.append(np.std(img))

        # ax[1,1].plot(depth[i],np.std(img), marker = 'o',ms = 10)

        if i == 0:
            ax[0, 1].imshow(img)
            ax[0,1].set_axis_off()
            ax[0, 1].plot(cY,cX, marker = 'o',ms = 10, c ='r')
        else:
            pass

    ax[1,1].plot(depth,error_crop, marker = 'o',ms = 10)
    ax[1,1].set_ylim(bottom=0, top=1)

    ax[1,1].set_ylabel('axial depth [pixel]', size = 20)
    ax[1,1].set_ylabel('standard deviation', size = 20)
    ax[1,1].set_title('trimming the edge', size=20)

    fig.suptitle('residual error against ideal plane versus depth', size = 25)
    plt.show()

    # # fig, ax = plt.subplots(1, 4, figsize=(16, 9))
        # fig = plt.figure(figsize=(16, 9))
        # ax = fig.add_subplot(2, 4, 1)
        # ax.imshow(raw_map)
        # ax.set_title('raw')
        # ax = fig.add_subplot(2, 4, 5, projection='3d')
        # surf = ax.plot_wireframe(xx, yy, raw_map, alpha=0.2)
        #
        # ax = fig.add_subplot(2, 4, 2)
        # ksize = 7
        # temp_median = median_filter(raw_map, size=ksize)
        # ax.imshow(temp_median)
        # ax.set_title('raw with median size of %d'%ksize)
        # ax = fig.add_subplot(2, 4, 6, projection='3d')
        # surf = ax.plot_wireframe(xx, yy, temp_median, alpha=0.2)
        #
        # ax = fig.add_subplot(2, 4, 3)
        # gsize = 4
        # temp_guass= gaussian_filter(raw_map, sigma=gsize)
        # ax.imshow(temp_guass)
        # ax.set_title('raw with guassian size of %.3f'%gsize)
        # ax = fig.add_subplot(2, 4, 7, projection='3d')
        # surf = ax.plot_wireframe(xx, yy, temp_guass, alpha=0.2)
        #
        #
        # ax = fig.add_subplot(2, 4, 4)
        # temp= gaussian_filter(temp_median, sigma=gsize)
        # ax.imshow(temp)
        # ax.set_title('combo A')
        # ax = fig.add_subplot(2, 4, 8, projection='3d')
        # surf = ax.plot_wireframe(xx, yy, temp, alpha=0.2)
        #
        # plt.tight_layout()
        # plt.show()