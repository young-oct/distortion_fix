# -*- coding: utf-8 -*-
# @Time    : 2022-07-22 13:44
# @Author  : young wang
# @FileName: axial_dist.py
# @Software: PyCharm

import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import surface_index, frame_index, plane_fit,filter_mask
from tools.pos_proc import export_map
import pyransac3d as pyrsc
import os
import cv2 as cv
from scipy.ndimage import median_filter,gaussian_filter
import matplotlib
from skimage import feature
from tools.plot import angle_est,heatmap


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/MEEI/v4-cal-corrected-surfaces/*.oct'))

    p_factor = np.linspace(0.6, 0.65, len(data_sets))
    shift = 0

    dis_map = []
    raw_dis_map = []
    res_error = []

    #for j in range(len(data_sets)):
    for j in range(1):
        data = load_from_oct_file(data_sets[j], clean=False)
        vmin, vmax = int(p_factor[j] * 255), 255

        xz_mask = np.zeros_like(data)

        # perform points extraction in the xz direction
        for i in range(data.shape[0]):
            xz_mask[i, :, :] = filter_mask(data[i, :, :], vmin=vmin, vmax=vmax)

        xz_pts = surface_index(xz_mask, shift)

####################################### Plot Raw Point Cloud #########################################
        fig = plt.figure(figsize=(16, 9))
        idx = 256
        ax = fig.add_subplot(121)
        ax.imshow(xz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)

        xz_slc = frame_index(xz_mask, 'x', idx, shift)
        x, y = zip(*xz_slc)
        ax.plot(y, x, linewidth=5, alpha=0.8, color='r')
        ax.set_title('slice %d from the xz direction' % idx)

        xz_pts_raw = xz_pts
        # np.delete(xz_pts, np.where(xz_pts))
        xz_pts = []
        cutoff = 36  # 269
        outter_cutoff = 220
        x_offset = 0
        y_offset = -11
        for points in range(len(xz_pts_raw)):
            radius = np.sqrt((xz_pts_raw[points][0]-data.shape[0]/2 + x_offset )**2 + (xz_pts_raw[points][1]-data.shape[1]/2 + y_offset)**2)
            if (radius > cutoff):
                xz_pts.append(xz_pts_raw[points])

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

        best_eq, best_inliers = ideal_plane.fit(pts, 0.1)

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

        #raw_map = gaussian_filter(raw_map, sigma=4)

        fig.suptitle('index at %d plane' % z_mean)

        plt.tight_layout()
        plt.show()

#########################################################################################################

        fig = plt.figure(figsize=(16, 9))

        ax = fig.add_subplot(3, 2, 1, projection='3d')
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

        ax = fig.add_subplot(3, 2, 3, projection='3d')
        cut_index = raw_map > 0.0

        surf = ax.plot_wireframe(xx, yy, raw_map, alpha=0.2)

        ax = fig.add_subplot(3, 2, 5)
        im, cbar = heatmap(raw_map, ax=ax,
                           cmap="hot", cbarlabel='depth variation')

        ax = fig.add_subplot(3, 2, 2, projection='3d')
        ax.set_title('raw points cloud \n'
                     '& linearly fitted plane')

        l_plane = plane_fit(xz_pts, order=1).zc
        ax.scatter(xp, yp, zp, s=0.5, alpha=0.5, c='r')

        plane_fit(xz_pts, order=1).plot(ax, z_low, z_high)

        ax = fig.add_subplot(3, 2, 4, projection='3d')
        dl_map = l_plane - raw_map#z_ideal
        surf = ax.plot_wireframe(xx, yy, dl_map, alpha=0.2)

        ax = fig.add_subplot(3, 2, 6)
        im, cbar = heatmap(dl_map.T, ax=ax,
                           cmap="hot", cbarlabel='depth variation')

        # you can export between linear, quadratic, cubic interpretation map
        dis_map.append((z_mean, dl_map))

        # export the raw point difference map
        raw_dis_map.append((z_mean, raw_map))
        res_error.append((z_mean,np.std(raw_map)))

        print('done with %d out of %d' % (int(j + 1), len(data_sets)))

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

    # from mpl_toolkits.mplot3d import Axes3D
    #
    # m = 512  # size of the matrix
    #
    # X1, X2 = np.mgrid[:m, :m]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(3, 1, 1, projection='3d')
    # jet = plt.get_cmap('jet')
    #
    # # plot the initial topological surface
    # #ax.plot_wireframe(xx, yy, raw_map, alpha=0.2)
    # ax.plot_surface(xx, yy, raw_map)
    #
    # # Regression
    # X = np.hstack((np.reshape(yy, (m * m, 1)), np.reshape(xx, (m * m, 1))))
    # X = np.hstack((np.ones((m * m, 1)), X))
    # YY = np.reshape(raw_map, (m * m, 1))
    #
    # theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    #
    # plane = np.reshape(np.dot(X, theta), (m, m));
    #
    # ax = fig.add_subplot(3, 1, 2, projection='3d')
    # ax.plot_surface(yy, xx, plane)
    # ax.plot_surface(yy, xx, raw_map, rstride=1, cstride=1, cmap=jet, linewidth=0)
    #
    # # Subtraction
    # Y_sub = raw_map - plane
    # ax = fig.add_subplot(3, 1, 3, projection='3d')
    # ax.plot_surface(yy, xx, Y_sub, rstride=1, cstride=1, cmap=jet, linewidth=0)
    #
    # plt.show()