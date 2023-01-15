# -*- coding: utf-8 -*-
# @Time    : 2022-07-22 13:44
# @Author  : young wang
# @FileName: axial_dist.py
# @Software: PyCharm

import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from tools.pre_proc import load_from_oct_file
from tools.proc import surface_index, frame_index, plane_fit,filter_mask
from tools.proc_surface_fit import load_from_oct_file_reversed, surface_index_reverse, frame_index_reverse
from tools.pos_proc import export_map
import pyransac3d as pyrsc
import os
import cv2 as cv
from scipy.ndimage import median_filter,gaussian_filter
import matplotlib
from skimage import feature
from tools.plot import angle_est,heatmap

import pandas as pd
import seaborn as sns

def CalculateResidualError(mean_err_list, std_err_list, data_sets, enable_plots=False, cutoff=32, outter_cutoff_percent=0.75, bottom_crop_off=5):
    for j in range(1, len(data_sets)):
        data = load_from_oct_file_reversed(data_sets[j], clean=False)
        vmin, vmax = int(0), 255

        xz_mask = np.zeros_like(data)

        # perform points extraction in the xz direction
        for i in range(data.shape[0]):
            xz_mask[i, :, :] = filter_mask(data[i, :, :], vmin=vmin, vmax=vmax)

        xz_pts = surface_index_reverse(xz_mask, 0)

        idx = 256
        xz_slc = frame_index_reverse(xz_mask, 'x', idx, 0)

        xz_pts_raw = xz_pts
        xp, yp, zp = zip(*xz_pts)

        xz_pts = []
        outter_cutoff = outter_cutoff_percent * min(np.amax(xp), np.amax(yp)) / 2
        x_offset = 0
        y_offset = -11
        for points in range(len(xz_pts_raw)):
            radius = np.sqrt((xz_pts_raw[points][0] - data.shape[0] / 2 + x_offset) ** 2 + (
                        xz_pts_raw[points][1] - data.shape[1] / 2 + y_offset) ** 2)
            if (radius > cutoff and xz_pts_raw[points][2] > bottom_crop_off):
                xz_pts.append(xz_pts_raw[points])

        xp, yp, zp = zip(*xz_pts)

        # construct ideal plane
        ideal_plane = pyrsc.Plane()
        pts = np.asarray(xz_pts)

        best_eq, best_inliers = ideal_plane.fit(pts, 0.1)

        a, b, c, d = best_eq[0], best_eq[1], - best_eq[2], best_eq[3]

        xx, yy = np.meshgrid(np.arange(0, data.shape[1], 1), np.arange(0, data.shape[1], 1))
        z_ideal = (d - a * xx - b * yy) / c
        z_mean = np.mean(z_ideal)

        #########################################################################################################
        # Calculate the linear plane fit
        l_plane = plane_fit(xz_pts, order=1).zc

        # obtained the raw point difference map
        raw_map = np.zeros((512, 512))
        for i in range(len(xz_pts)):
            lb, hb = z_mean * 0.5, z_mean * 1.5
            if lb <= xz_pts[i][2] <= hb:
                raw_map[xz_pts[i][0], xz_pts[i][1]] = int(xz_pts[i][2])
            else:
                pass

        # Calculate the normalized difference map
        dl_map = l_plane - raw_map
        dl_map = np.where(np.abs(l_plane) == np.abs(dl_map), 0.0, dl_map)
        dl_map_mean = dl_map[dl_map > 0.0].mean()
        dl_map[dl_map == 0.0] = dl_map_mean
        dl_map = np.abs((dl_map - dl_map_mean) / data.shape[2])

        # Optional filtering
        # dl_map = median_filter(dl_map, size=10)
        # dl_map = gaussian_filter(dl_map, sigma=1.0)

        # Calculate error measures for the current volume
        dl_map_nan = dl_map.copy()
        dl_map_nan[dl_map_nan == 0] = np.nan
        mean_depth_error_non_zero = np.nanmean(dl_map_nan)
        std_depth_error_non_zero = np.nanstd(dl_map_nan)

        # Append the residual error measures
        mean_err_list.append((z_mean, mean_depth_error_non_zero))
        std_err_list.append((z_mean, std_depth_error_non_zero))

        ####################################### Plot Raw Point Cloud #########################################
        if (enable_plots):
            fig = plt.figure(figsize=(16, 9))
            fig.suptitle('index at %d plane' % z_mean)

            ax = fig.add_subplot(121)
            ax.set_title('slice %d from the xz direction' % idx)

            x, y = zip(*xz_slc)
            ax.plot(y, x, linewidth=5, alpha=0.8, color='r')

            ax.imshow(xz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)

            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(xp, yp, zp, s=0.1, alpha=0.1, c='r')

            ax.set_title('raw points cloud')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim([0, data.shape[0]])
            ax.set_ylim([0, data.shape[1]])
            ax.set_zlim([0, data.shape[2]])
            plt.tight_layout()
            plt.show()

        ############################## Plot Fitted plane and Error map ###############################################
            fig = plt.figure(figsize=(16, 9))

            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.set_title('raw points cloud \n'
                         '& linearly fitted plane')

            z_low, z_high = int(z_mean - 30), int(z_mean + 30)
            plane_fit(xz_pts, order=1).plot(ax, z_low, z_high)

            ax = fig.add_subplot(1, 2, 2)
            im, cbar = heatmap(dl_map.T, ax=ax,
                               cmap="hot", cbarlabel='depth variation')

        print('done with %d out of %d' % (int(j + 1), len(data_sets)))

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    # error list (corrected)
    mean_err_list = []
    std_err_list = []

    data_sets = natsorted(glob.glob('D:/Distoriton correction/2022.1.09(MEEI)/Spiral/MDL-85/flat surface - deconv - corrected/*.oct'))

    CalculateResidualError(mean_err_list, std_err_list, data_sets)

    # Convert to numpy arrays
    mean_err_list = np.asarray(mean_err_list)
    std_err_list = np.asarray(std_err_list)

    # error list (uncorrected)
    mean_err_list_uncorr = []
    std_err_list_uncorr = []

    data_sets_uncorr = natsorted(glob.glob('D:/Distoriton correction/2022.1.09(MEEI)/Spiral/MDL-85/flat surface - deconv/*.oct'))

    CalculateResidualError(mean_err_list_uncorr, std_err_list_uncorr, data_sets_uncorr)

    # Convert to numpy arrays
    mean_err_list_uncorr = np.asarray(mean_err_list_uncorr)
    std_err_list_uncorr = np.asarray(std_err_list_uncorr)

    # Plot residual error results
    Depth_res = 40.0
    Depth_dim = 330
    Depth_dim_phy = 13.2

    depth_index = Depth_dim_phy*(mean_err_list[:,0]/Depth_dim)
    corrected_depth_mean_error = Depth_res*Depth_dim*mean_err_list[:,1]
    corrected_depth_std_error = Depth_res*Depth_dim*std_err_list[:,1]
    uncorrected_depth_mean_error = Depth_res*Depth_dim*mean_err_list_uncorr[:,1]
    uncorrected_depth_std_error = Depth_res*Depth_dim*std_err_list_uncorr[:,1]

    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('Residual Depth Error Comparison')
    # ax.set_xlabel('Depth Index')
    # ax.set_ylabel('Surface Deviation From Flatness (um)')
    #
    # ax.scatter(depth_index, corrected_depth_mean_error, color='b')
    # p1 = ax.errorbar(depth_index, corrected_depth_mean_error, yerr=corrected_depth_std_error, fmt="o", capsize=5)
    #
    # ax.scatter(depth_index, corrected_depth_mean_error, color='r')
    # p2 = ax.errorbar(depth_index, uncorrected_depth_mean_error, yerr=uncorrected_depth_std_error, fmt="o", capsize=5, color='r')
    #
    # ax.legend([p1,p2],['Corrected','Uncorrected'])

    ###################### Bar plot #############################
    # Construct Pandas data  frame
    surface = ["Corrected" for x in range(len(corrected_depth_mean_error))] + ["Uncorrected" for x in range(len(corrected_depth_mean_error))]
    depth_index_bars = np.concatenate((np.flip(depth_index), np.flip(depth_index)))
    mean = np.concatenate((np.flip(corrected_depth_mean_error), np.flip(uncorrected_depth_mean_error)))
    std = np.concatenate((np.flip(corrected_depth_std_error), np.flip(uncorrected_depth_std_error)))

    dataFrame = pd.DataFrame({'surface' : surface,
                         'depth_index' : depth_index_bars,
                         'mean' : mean,
                         'std' : std})

    # Specify desired colours
    sns.set(style="whitegrid")
    colors = ["#14CCEB", "#EB3314"]
    customPalette = sns.set_palette(sns.color_palette(colors))

    # Create bar plot
    fig, ax = plt.subplots(figsize=(16, 9))
    g = sns.barplot(
        ax=ax,data=dataFrame,
        x='depth_index', y='mean', hue='surface',
        palette=customPalette, alpha=.6 )

    # Drawing a horizontal line at point 1.25
    g.axhline(Depth_res, ls='--', c='k',alpha=0.2,label='test')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, Depth_res, "{:.0f}".format(Depth_res),
            color="k", alpha=0.2, transform=trans,ha="right", va="center")

    # Format xlabels
    xlabels = [t.get_text()  for t in ax.get_xticklabels()]
    xlabels = list(map(float, xlabels))
    xlabels = ['{:.2f}'.format(x) for x in xlabels]
    g.set_xticklabels(xlabels)

    # Draw error bars
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.patches]
    y_coords = [p.get_height() for p in g.patches]
    half_len = int(len(x_coords)/2)
    ax.errorbar(x=x_coords[0:half_len], y=y_coords[0:half_len], yerr=dataFrame["std"][0:half_len], fmt="none", c="#14CCEB")
    ax.errorbar(x=x_coords[half_len:], y=y_coords[half_len:], yerr=dataFrame["std"][half_len:], fmt="none", c="#EB3314")

    # Format plot
    ax.legend(loc='upper left')
    ax.set_title('Absolute Mean Depth Error Comparison', fontweight='bold', fontsize=18)
    ax.set_xlabel('Surface Depth [mm]', fontweight='bold', fontsize=15)
    ax.set_ylabel('Absolute Mean Depth Error [um]', fontweight='bold', fontsize=15)
    ax.set_ylim(ymin=0)