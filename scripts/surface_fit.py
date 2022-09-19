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


def ConvertToSpherical(data, shape, radius = 13.5, xAngle=0.329062, yAngle=0.329062, xOffset=0, yOffset=0, alpha = 0.0, beta = 0.0, zOffset = 0):
    # Setup scan conversion parameters
    startRadius = radius #13.603;
    stopRadius = radius+10.0 #23.603;
    # startTheta = -xAngle + xOffset;
    # stopTheta = xAngle + xOffset;
    # startPhi = -yAngle + yOffset;
    # stopPhi = yAngle + yOffset;
    startTheta = -xAngle
    stopTheta = xAngle
    startPhi = -yAngle
    stopPhi = yAngle

    startX = stopRadius * np.sin(startTheta);
    startY = stopRadius * np.sin(startPhi);
    startZ = startRadius * np.cos(startTheta);
    stopX = stopRadius * np.sin(stopTheta);
    stopY = stopRadius * np.sin(stopPhi);
    stopZ = stopRadius;

    # print("********************************")
    # print("startX: " + str(startX))
    # print("stopX: " + str(stopX))
    # print("startY: " + str(startY))
    # print("stopY: " + str(stopY))
    # print("startZ: " + str(startZ))
    # print("stopZ: " + str(stopZ))

    if beta > 0.0:
        stopYVec = (
            0.0,
            stopRadius * np.sin(stopPhi),
            stopRadius * np.cos(stopPhi))
        startYVec = (
            0.0,
            stopRadius * np.sin(startPhi) if (beta < stopPhi) else startRadius * np.sin(startPhi),
            stopRadius * np.cos(startPhi) if (beta < stopPhi) else startRadius * np.cos(startPhi))
    else:
        startYVec = (
            0.0,
            stopRadius * np.sin(startPhi),
            stopRadius * np.cos(startPhi))
        stopYVec = (
            0.0,
            startRadius * np.sin(stopPhi) if (beta < startPhi) else stopRadius * np.sin(stopPhi),
            startRadius * np.cos(stopPhi) if (beta < startPhi) else stopRadius * np.cos(stopPhi))

    if alpha > 0.0:
        startXVec = (
            stopRadius * np.sin(startTheta),
            startRadius * np.sin(startPhi) if (beta < startPhi) else startYVec[1],
            stopRadius * np.cos(startTheta))
        stopXVec = (
            stopRadius * np.sin(stopTheta) if (alpha < stopTheta) else startRadius * np.sin(stopTheta),
            (stopRadius * np.sin(stopPhi) if (beta < startPhi) else startRadius * np.sin(stopPhi)) if (beta <= 0.0) else (startRadius * np.sin(stopPhi) if (beta < stopPhi) else stopRadius * np.sin(stopPhi)),
            stopRadius * np.cos(stopTheta) if (alpha < stopTheta) else startRadius * np.cos(stopTheta))

    else:
        startXVec = (
            startRadius * np.sin(startRadius) if (alpha < startTheta) else stopRadius * np.sin(startTheta),
            (stopRadius * np.sin(startPhi) if (beta < startPhi) else startRadius * np.sin(startPhi)) if (beta <= 0.0) else (startRadius * np.sin(startPhi) if (beta < stopPhi) else stopRadius * np.sin(startPhi)),
            startRadius * np.cos(startTheta) if (alpha < startTheta) else stopRadius * np.cos(startTheta))
        stopXVec = (
            stopRadius * np.sin(stopTheta),
            (startRadius * np.sin(stopPhi) if (beta < startPhi) else (stopRadius * np.sin(stopPhi) if (beta < stopTheta) else startRadius * np.sin(stopPhi))),
            stopRadius * np.cos(stopTheta))

    tiltXStart = startXVec[0] * np.cos(alpha) - np.abs(startXVec[1] * np.sin(alpha) * np.sin(beta)) - startXVec[2] * np.sin(alpha) * np.cos(beta)
    tiltXStop = stopXVec[0] * np.cos(alpha) + np.abs(stopXVec[1] * np.sin(alpha) * np.sin(beta)) - stopXVec[2] * np.sin(alpha) * np.cos(beta)
    tiltYStart = startYVec[1] * np.cos(beta) + startYVec[2] * np.sin(beta)
    tiltYStop = stopYVec[1] * np.cos(beta) + stopYVec[2] * np.sin(beta)

    if alpha <= 0.0:
        startZ = startRadius * np.cos(stopTheta) * np.cos(alpha) + startRadius * np.sin(stopTheta) * np.sin(alpha)
        if alpha > startTheta:
            stopZ = stopRadius
        else:
            stopZ = stopRadius * np.cos(startTheta) * np.cos(alpha) + stopRadius * np.sin(startTheta) * np.sin(alpha)


    else:
        startZ = startRadius * np.cos(startTheta) * np.cos(alpha) + startRadius * np.sin(startTheta) * np.sin(alpha)
        if alpha > stopTheta:
            stopZ = (stopRadius * np.cos(stopTheta)) * np.cos(alpha) + (stopRadius * np.sin(stopTheta)) * np.sin(alpha)
        else:
            stopZ = stopRadius

    if (beta <= 0.0):
        startZ = startRadius * np.sin(stopPhi) * np.sin(beta) + startZ * np.cos(beta)
        if (beta < startPhi):
            stopZ = stopZ * np.cos(stopPhi) * np.cos(beta) - stopRadius * np.sin(stopPhi) * np.sin(beta)

    else:
        startZ = startRadius * np.sin(startPhi) * np.sin(beta) + startZ * np.cos(beta);
        if (beta > stopPhi):
            stopZ = stopZ * np.cos(startPhi) * np.cos(beta) - stopRadius * np.sin(startPhi) * np.sin(beta)

    # print("tiltXStart: " + str(tiltXStart))
    # print("tiltXStop: " + str(tiltXStop))
    # print("tiltYStart: " + str(tiltYStart))
    # print("tiltYStop: " + str(tiltYStop))
    # print("tilt startZ: " + str(startZ))
    # print("tilt stopZ: " + str(stopZ))
    # print("********************************")

    newXSpacing = (tiltXStop - tiltXStart) / (shape[0] - 1);
    newYSpacing = (tiltYStop - tiltYStart) / (shape[1] - 1);
    newZSpacing = (stopZ - startZ) / (shape[2] - 1);
    depthSpacing = (stopRadius - startRadius) / (shape[2] - 1);
    thetaSpacing = (stopTheta - startTheta) / (shape[0] - 1);
    phiSpacing = (stopPhi - startPhi) / (shape[1] - 1);

    xp, yp, zp = zip(*data)
    xp = (np.asarray(xp))
    yp = (np.asarray(yp))
    zp = (np.asarray(zp))
    # correct for uncertain ideal plane location
    zp = zp + zOffset

    # Calculate new Cartesian coordinates
    #newX = (np.asarray(xp) - xOffset) * newXSpacing + startX
    newX = (xp) * newXSpacing + tiltXStart
    #newY = (np.asarray(yp) - yOffset) * newYSpacing + startY
    newY = (yp) * newYSpacing + tiltYStart
    newZ = zp * newZSpacing + startZ

    tiltX = newX * np.cos(alpha) + newZ * np.sin(alpha)
    tiltY = newX * np.sin(beta) * np.sin(alpha) + newY * np.cos(beta) + newZ * np.sin(beta) * np.cos(alpha)
    tiltZ = -1.0*newX * np.cos(beta) * np.sin(alpha) + newY * np.sin(beta) + newZ * np.cos(beta) * np.cos(alpha)

    # Cartesian to spherical
    # r = np.sqrt(np.square(newX) + np.square(newZ) + np.square(newY))
    # th = np.arctan2(newX, newZ)
    # phi = np.arctan2(newY, newZ)  # np.arcsin(newY / r)

    r = np.sqrt(np.square(tiltX) + np.square(tiltZ) + np.square(tiltY))
    th = np.arctan2(tiltX, tiltZ)
    phi = np.arctan2(tiltY, tiltZ)  # np.arcsin(newY / r)

    # Normalize
    r = ((r - startRadius) / depthSpacing) / (shape[2])
    th = ((th - startTheta) / thetaSpacing) / (shape[0])
    phi = ((phi - startPhi) / phiSpacing) / (shape[1])

    # Normalize angle range
    th = (th - np.min(th)) / np.ptp(th)
    phi = (phi - np.min(phi)) / np.ptp(phi)

    return r, th, phi

def func(x):
    # radius_ideal, th_ideal, phi_ideal = ConvertToSpherical(ideal_pts, [data.shape[0], data.shape[1], data.shape[2]],
    #                                                        radius=x[0], xAngle=x[1], yAngle=x[2], xOffset=x[3], yOffset=x[4],
    #                                                        alpha=x[5], beta=x[6])
    radius_ideal, th_ideal, phi_ideal = ConvertToSpherical(ideal_pts, [data.shape[0], data.shape[1], data.shape[2]],
                                                           radius=x[0],
                                                           xAngle=x[1], yAngle=x[2],
                                                           alpha=x[3], beta=x[4],
                                                           zOffset=x[5])
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

    data_sets = natsorted(glob.glob('../data/MEEI/MEEI flat surface-002/*.oct'))
    folder_path = '../data/correction map'

    p_factor = 0.6
    shift = 0

    data = load_from_oct_file_reversed(data_sets[0], clean=True)
    vmin, vmax = int(p_factor * 255), 255

    xz_mask = np.zeros_like(data)

    # perform points extraction in the xz direction
    for i in range(data.shape[0]):
        xz_mask[i, :, :] = filter_mask(data[i, :, :], vmin=vmin, vmax=vmax)

    xz_pts_raw = surface_index_reverse(xz_mask, shift)
    # np.delete(xz_pts, np.where(xz_pts))
    xz_pts = []
    cutoff = 296 #259
    for points in range(len(xz_pts_raw)):
        if (xz_pts_raw[points][2] >= cutoff):
            xz_pts.append(xz_pts_raw[points])

    fig = plt.figure(figsize=(16, 9))
    idx = 228
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

    #bnds = ((10.0, 20.0), (0.1, 0.5), (0.1, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
    bnds = ((10.0, 20.0), (0.1, 0.5), (0.1, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-10, 10))
    x = (14.0, 0.2, 0.2, 0.0, 0.0, 0.0)
    options = {
                  #"ftol": 0.000001,
                  "eps": 0.00001,
                  "maxiter": 1000,
                  "disp": True
                }
    minResults = minimize(func, x, method='SLSQP', bounds=bnds, options=options)

    # plot best fit
    radius_ideal, th_ideal, phi_ideal = ConvertToSpherical(ideal_pts, [data.shape[0], data.shape[1], data.shape[2]],
                                                           radius=minResults.x[0],
                                                           xAngle=minResults.x[1], yAngle=minResults.x[2],
                                                           alpha=minResults.x[3], beta=minResults.x[4],
                                                           zOffset=x[5])

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
    temp_name = data_sets[0].split('/')[-1]
    file_name = temp_name.split('.')[0]
    file_path = (os.path.join(folder_path, 'RadialCorrectionMap.bin'))
    export_map(spherical_raw_map, file_path)