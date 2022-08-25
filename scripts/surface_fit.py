# -*- coding: utf-8 -*-
# @Time    : 2022-07-22 13:44
# @Author  : young wang
# @FileName: axial_dist.py
# @Software: PyCharm

import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from tools.proc import plane_fit, filter_mask
from tools.proc_surface_fit import load_from_oct_file_reversed, surface_index_reverse, frame_index_reverse
from tools.pos_proc import export_map
import pyransac3d as pyrsc
import os
import cv2 as cv
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib
from skimage import feature
from tools.plot import angle_est, heatmap

from operator import itemgetter

from itertools import combinations_with_replacement
from scipy.optimize import leastsq
from scipy.optimize import minimize


def ConvertToSpherical(data, shape, xAngle=0.329062, yAngle=0.329062, xOffset=0, yOffset=0):
    # Setup scan conversion parameters
    startRadius = 13.603;
    stopRadius = 23.603;
    startTheta = -xAngle + xOffset;
    stopTheta = xAngle + xOffset;
    startPhi = -yAngle + yOffset;
    stopPhi = yAngle + yOffset;
    startX = stopRadius * np.sin(startTheta);
    startY = stopRadius * np.sin(startPhi);
    startZ = startRadius * np.cos(startTheta);
    stopX = stopRadius * np.sin(stopTheta);
    stopY = stopRadius * np.sin(stopPhi);
    stopZ = stopRadius;
    newXSpacing = (stopX - startX) / (shape[0] - 1);
    newYSpacing = (stopY - startY) / (shape[1] - 1);
    newZSpacing = (stopZ - startZ) / (shape[2] - 1);
    depthSpacing = (stopRadius - startRadius) / (shape[2] - 1);
    thetaSpacing = (stopTheta - startTheta) / (shape[0] - 1);
    phiSpacing = (stopPhi - startPhi) / (shape[1] - 1);

    xp, yp, zp = zip(*data)

    # Calculate new Cartesian coordinates
    newX = (np.asarray(xp) - xOffset) * newXSpacing + startX
    newY = (np.asarray(yp) - yOffset) * newYSpacing + startY
    newZ = np.asarray(zp) * newZSpacing + startZ

    # Cartesian to spherical
    r = np.sqrt(np.square(newX) + np.square(newZ) + np.square(newY))
    th = np.arctan2(newX, newZ)
    phi = np.arctan2(newY, newZ)  # np.arcsin(newY / r)

    # Normalize
    r = ((r - startRadius) / depthSpacing) / (shape[2])
    th = ((th - startTheta) / thetaSpacing) / (shape[0])
    phi = ((phi - startPhi) / phiSpacing) / (shape[1])

    # Normalize angle range
    th = (th - np.min(th)) / np.ptp(th)
    phi = (phi - np.min(phi)) / np.ptp(phi)

    return r, th, phi

def func(x):
    radius_ideal, th_ideal, phi_ideal = ConvertToSpherical(ideal_pts, [data.shape[0], data.shape[1], data.shape[2]],
                                                           xAngle=x[0], yAngle=x[1], xOffset=x[2], yOffset=x[3])
    mean_radius = np.mean(radius)

    # obtained the raw point difference map
    map = np.zeros((512, 512)).astype(np.float32)
    for i in range(len(xz_pts)):
        map[xz_pts[i][0], xz_pts[i][1]] = radius[i] - radius_ideal[i]

    map = gaussian_filter(map, sigma=4)

    return np.sum(np.abs(map))


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/1mW/Flat-Surface/*.oct'))
    folder_path = '../data/correction map'

    p_factor = 0.55
    shift = 0

    data = load_from_oct_file_reversed(data_sets[0], clean=False)
    vmin, vmax = int(p_factor * 255), 255

    xz_mask = np.zeros_like(data)

    # perform points extraction in the xz direction
    for i in range(data.shape[0]):
        xz_mask[i, :, :] = filter_mask(data[i, :, :], vmin=vmin, vmax=vmax)

    xz_pts = surface_index_reverse(xz_mask, shift)
    # np.delete(xz_pts, np.where(xz_pts))

    cutoff = 258
    for points in range(len(xz_pts)):
        if (xz_pts[points][2] < cutoff):
            xz_pts[points] = (xz_pts[points][0], xz_pts[points][1], cutoff)

    fig = plt.figure(figsize=(16, 9))
    idx = 256
    ax = fig.add_subplot(121)
    ax.imshow(xz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)

    xz_slc = frame_index_reverse(xz_mask, 'x', idx, shift)
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

    ####################  Surface Fitting  #######################
    th, phi, radius = zip(*xz_pts)
    th = np.asarray(th) / data.shape[0]
    phi = np.asarray(phi) / data.shape[1]
    radius = np.asarray(radius) / data.shape[2]

    maxZ = min(xz_pts, key=itemgetter(2))[2]

    ideal_pts = []
    for k in range(len(xz_pts)):
        ideal_pts.append((xz_pts[k][0], xz_pts[k][1], maxZ))

    bnds = ((0.1, 0.5), (0.1, 0.5), (-0.5, 0.5), (-0.5, 0.5))
    x = (0.3, 0.3, 0.0, 0.0)
    options = {
                  "ftol": 0.000001,
                  "eps": 0.00001,
                  "maxiter": 1000,
                  "disp": True
                }
    minResults = minimize(func, x, method='SLSQP', bounds=bnds, options=options)

    # plot best fit
    radius_ideal, th_ideal, phi_ideal = ConvertToSpherical(ideal_pts, [data.shape[0], data.shape[1], data.shape[2]],
                                                           xAngle=minResults.x[0], yAngle=minResults.x[1], xOffset=minResults.x[2], yOffset=minResults.x[3])

    # obtained the raw point difference map
    spherical_raw_map = np.zeros((512, 512)).astype(np.float32)
    for i in range(len(xz_pts)):
        spherical_raw_map[xz_pts[i][0], xz_pts[i][1]] = radius[i] - radius_ideal[i]

    spherical_raw_map = gaussian_filter(spherical_raw_map, sigma=4)

    ####################  Plot surface fit  #######################
    fig = plt.figure(figsize=(16, 9))
    mean_radius = np.mean(radius)
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(th, phi, radius, s=0.1, alpha=0.1, c='r')
    # ax.scatter(th_ideal, phi_ideal, radius_ideal, s=0.1, alpha=0.1, c='b')
    ax.set_title('Extracted surface', size=15)
    ax.set_xlabel('th')
    ax.set_ylabel('phi')
    ax.set_zlabel('radius')
    radius_low, radius_high = mean_radius - 0.1, mean_radius + 0.1
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_zlim([radius_low, radius_high])

    ax = fig.add_subplot(223, projection='3d')
    # ax.scatter(th, phi, radius, s=0.1, alpha=0.1, c='r')
    ax.scatter(th_ideal, phi_ideal, radius_ideal, s=0.1, alpha=0.1, c='b')
    ax.set_title('Fitted spherical model surface', size=15)
    ax.set_xlabel('th')
    ax.set_ylabel('phi')
    ax.set_zlabel('radius')
    radius_low, radius_high = mean_radius - 0.1, mean_radius + 0.1
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_zlim([radius_low, radius_high])

    ax = fig.add_subplot(222, projection='3d')
    ax.set_title('Difference map', size=15)
    xx, yy = np.meshgrid(np.arange(0, data.shape[1], 1), np.arange(0, data.shape[1], 1))
    surf = ax.plot_wireframe(xx, yy, spherical_raw_map, alpha=0.2)

    ax = fig.add_subplot(224)
    ax.set_title('Difference heat map', size=15)
    im, cbar = heatmap(spherical_raw_map, ax=ax,
                       cmap="hot", cbarlabel='depth variation')

    ####################  Distortion Map Export  #######################
    # temp_name = data_sets[j].split('/')[-1]
    # file_name = temp_name.split('.')[0]
    # file_path = (os.path.join(folder_path, '%s.bin' % file_name))
    # export_map(spherical_raw_map, file_path)