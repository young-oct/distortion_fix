
import glob
from natsort import natsorted
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import griddata
import discorpy.prep.preprocessing as prep
from tools.plot import heatmap
from sklearn.metrics import mean_squared_error
import math

# Axial error estimation
from tools.proc import surface_index, frame_index, plane_fit,filter_mask
from tools.proc_surface_fit import load_from_oct_file_reversed, surface_index_reverse, frame_index_reverse
import pyransac3d as pyrsc
from scipy.ndimage import gaussian_filter, median_filter

def ExtractDotGrid(grid):
    # Create binary image of uncorrected grid
    binaryImageGrid = prep.binarization(grid)

    # Calculate the median dot size and distance between them
    (dot_size, dot_dist) = prep.calc_size_distance(binaryImageGrid)

    #Extract each row of dots
    list_hor_lines = prep.group_dots_hor_lines(binaryImageGrid,
                                               prep.calc_hor_slope(binaryImageGrid),
                                               2 * dot_dist,
                                               ratio=0.3,
                                               accepted_ratio=0.2)

    # Convert extracted lines into dot array (x and y are swapped)
    extractedGrid = np.vstack(list_hor_lines)
    extractedGrid[:, [0, 1]] = extractedGrid[:, [1, 0]]

    return extractedGrid

def CalcDifferenceMap(uncorrected_grid, correction_grid):
    extracted_uncorrected_grid = ExtractDotGrid(uncorrected_grid) / uncorrected_grid.shape[0]
    extracted_correction_grid = ExtractDotGrid(correction_grid) / correction_grid.shape[0]

    corrNormalized = (extracted_correction_grid)

    diffNormalized = np.sqrt(
        np.square((extracted_correction_grid[:, 0] - extracted_uncorrected_grid[:, 0])) +
        np.square((extracted_correction_grid[:, 1] - extracted_uncorrected_grid[:, 1])))

    # Interpolate grids to produce calibration maps
    grid_y, grid_x = np.mgrid[0:1:512j, 0:1:512j]
    map = griddata(
        corrNormalized, diffNormalized,
        (grid_y, grid_x), method='cubic', fill_value=0.0).astype('float32')

    return map


if __name__ == '__main__':
    uncorrected_grids = natsorted(glob.glob('../data/later distortion analysis/uncorrected grids/*.png'))
    correction_grids = natsorted(glob.glob('../data/later distortion analysis/correction grids/*.png'))

    error_list_mean = []
    error_list_std = []

    uncorrected_grid = (io.imread(uncorrected_grids[-1], as_gray=True) * 255).astype(np.uint8)
    corrected_grid = (io.imread(uncorrected_grids[0], as_gray=True) * 255).astype(np.uint8)
    ref_grid = (io.imread(correction_grids[-1], as_gray=True) * 255).astype(np.uint8)

    map_uncorrected = CalcDifferenceMap(uncorrected_grid, ref_grid)
    map_corrected = CalcDifferenceMap(corrected_grid, ref_grid)

    # Error calculation
    map_nan_uncorrected = map_uncorrected.copy()
    map_nan_uncorrected[map_nan_uncorrected == 0] = np.nan
    map_nan_corrected = map_corrected.copy()
    map_nan_corrected[map_nan_corrected == 0] = np.nan

    # arrays without nan values
    uncorrected_cropped = map_nan_uncorrected[~np.isnan(map_nan_uncorrected)]
    corrected_cropped = map_nan_corrected[~np.isnan(map_nan_corrected)]

    # Statistics
    error_list_mean.append(np.nanmean(map_nan_uncorrected))
    error_list_std.append(np.nanstd(map_nan_uncorrected))
    error_list_mean.append(np.nanmean(map_nan_corrected))
    error_list_std.append(np.nanstd(map_nan_corrected))

    # RMSE corrected
    MSE_corrected = mean_squared_error(corrected_cropped, np.zeros(len(corrected_cropped)))
    RMSE_corrected = math.sqrt(MSE_corrected)

    # RMSE uncorrected
    MSE_uncorrected = mean_squared_error(uncorrected_cropped, np.zeros(len(uncorrected_cropped)))
    RMSE_uncorrected = math.sqrt(MSE_uncorrected)

    FOV = 30.0
    label_size = 20
    FOV_width = 10000 # um
    axial_length = 13000 # um

    ####################### Plot the heatmap (uncorrected) #####################
    fig, ax = plt.subplots(1, figsize=(16, 10))
    im = ax.imshow(np.abs(map_uncorrected*FOV), cmap="hot")

    # Set colour bar axis height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Absolute angular error (degrees)', rotation=-90, va="bottom", size=20)
    cbar.ax.tick_params(labelsize=label_size)
    cbar.ax.locator_params(nbins=6)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_axis_off()
    fig.tight_layout()

    pic_dis = '../plots/uncorrected_angle_error_unscaled.pdf'
    plt.savefig(pic_dis, dpi=600,
                transparent=True,
                bbox_inches='tight', pad_inches=0)

    ####################### Plot the heatmap (residual error) #####################
    fig, ax = plt.subplots(1, figsize=(16, 10))
    im = ax.imshow(np.abs(map_corrected * FOV), cmap="hot")

    # Set colour bar axis height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Residual absolute angular error (degrees)', rotation=-90, va="bottom", size=20)
    cbar.ax.tick_params(labelsize=label_size)
    cbar.ax.locator_params(nbins=8)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_axis_off()
    fig.tight_layout()

    pic_dis = '../plots/corrected_angle_error_unscaled.pdf'
    plt.savefig(pic_dis, dpi=600,
                transparent=True,
                bbox_inches='tight', pad_inches=0)

    ################################ Axial error ############################################

    # load in surface data
    data_sets_axial = natsorted(
        glob.glob('../data/MEEI/2022.1.09(MEEI)/flat surface - corrected/*.oct'))
    data_axial = load_from_oct_file_reversed(data_sets_axial[0], clean=False)

    ################################# Surface fitting ####################################
    p_factor = 0.65
    vmin, vmax = int(p_factor * 255), 255
    xz_mask = np.zeros_like(data_axial)

    # perform points extraction in the xz direction
    for i in range(data_axial.shape[0]):
        xz_mask[i, :, :] = filter_mask(data_axial[i, :, :], vmin=vmin, vmax=vmax)

    # Extract surface
    xz_pts = surface_index_reverse(xz_mask, 0)
    xz_pts_raw = xz_pts
    xp, yp, zp = zip(*xz_pts)

    # Crop out central artefact (GRIN rod)
    xz_pts = []
    outter_cutoff_percent = 0.75
    outter_cutoff = outter_cutoff_percent * min(np.amax(xp), np.amax(yp)) / 2
    cutoff = 25
    bottom_crop_off = 5
    x_offset = 11
    y_offset = -5
    for points in range(len(xz_pts_raw)):
        radius = np.sqrt((xz_pts_raw[points][0] - data_axial.shape[0] / 2 + x_offset) ** 2 + (
                xz_pts_raw[points][1] - data_axial.shape[1] / 2 + y_offset) ** 2)
        if (radius > cutoff and xz_pts_raw[points][2] > bottom_crop_off):
            xz_pts.append(xz_pts_raw[points])

    # Unzip cropped points
    xp, yp, zp = zip(*xz_pts)

    ################################ Test Plots ##########################################
    # fig = plt.figure(figsize=(16, 9))
    # #fig.suptitle('index at %d plane' % z_mean)
    # ax = fig.add_subplot(121)
    # ax.set_title('slice %d from the xz direction' % idx)
    # idx = 256
    # xz_slc = frame_index_reverse(xz_mask, 'x', idx, 0)
    # x, y = zip(*xz_slc)
    # ax.plot(y, x, linewidth=5, alpha=0.8, color='r')
    # ax.imshow(xz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    #
    # ax = fig.add_subplot(122, projection='3d')
    # ax.scatter(xp, yp, zp, s=0.1, alpha=0.1, c='r')
    # ax.set_title('raw points cloud')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_xlim([0, data_axial.shape[0]])
    # ax.set_ylim([0, data_axial.shape[1]])
    # ax.set_zlim([0, data_axial.shape[2]])
    # plt.tight_layout()
    # plt.show()
    ######################################################################################

    # construct ideal plane
    ideal_plane = pyrsc.Plane()
    pts = np.asarray(xz_pts)

    best_eq, best_inliers = ideal_plane.fit(pts, 0.1)
    a, b, c, d = best_eq[0], best_eq[1], - best_eq[2], best_eq[3]
    xx, yy = np.meshgrid(np.arange(0, data_axial.shape[1], 1), np.arange(0, data_axial.shape[1], 1))
    z_ideal = (d - a * xx - b * yy) / c
    z_mean = np.mean(z_ideal)

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
    dl_map = np.abs((dl_map - dl_map_mean) / data_axial.shape[2])
    #dl_map = (dl_map - dl_map_mean) / data_axial.shape[2]

    fig, ax = plt.subplots(1, figsize=(16, 10))
    im = ax.imshow(dl_map*13000, cmap="hot")

    # Set colour bar axis height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Absolute radial error (um)', rotation=-90, va="bottom", size=20)
    cbar.ax.tick_params(labelsize=label_size)
    cbar.ax.locator_params(nbins=6)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_axis_off()
    fig.tight_layout()

    pic_dis = '../plots/dl_map.pdf'
    plt.savefig(pic_dis, dpi=600,
                transparent=True,
                bbox_inches='tight', pad_inches=0)

    # filter map
    dl_map_filtered = dl_map.copy()
    dl_map_filtered = gaussian_filter(dl_map_filtered, sigma=4)
    dl_map_filtered = median_filter(dl_map_filtered, size=9)

    # crop
    dl_map_filtered[dl_map==0]=0
    dl_map_filtered[(dl_map_filtered*axial_length) > 72.0] = 0.0

    fig, ax = plt.subplots(1, figsize=(16, 10))
    im = ax.imshow(dl_map_filtered*axial_length, cmap="hot")

    # Set colour bar axis height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Absolute radial error (um)', rotation=-90, va="bottom", size=20)
    cbar.ax.tick_params(labelsize=label_size)
    cbar.ax.locator_params(nbins=10)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_axis_off()
    fig.tight_layout()

    pic_dis = '../plots/dl_map_filtered.pdf'
    plt.savefig(pic_dis, dpi=600,
                transparent=True,
                bbox_inches='tight', pad_inches=0)

    ########################## Combined heatmap ####################
    # Crop axial error map to lateral error map
    dl_map_cropped = dl_map.copy()
    dl_map_cropped[map_uncorrected == 0] = 0
    dl_map_cropped[dl_map == 0] = 0

    dl_map_cropped = gaussian_filter(dl_map_cropped, sigma=4)
    dl_map_cropped = median_filter(dl_map_cropped, size=9)
    dl_map_cropped[dl_map==0]=0

    total_error_map = np.sqrt((map_corrected*FOV_width)**2+(dl_map_cropped*axial_length)**2)
    total_error_map[dl_map==0]=0

    fig, ax = plt.subplots(1, figsize=(16, 10))
    im = ax.imshow(total_error_map, cmap="hot")

    # Set colour bar axis height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Create colorbar
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Absolute total error (um)', rotation=-90, va="bottom", size=20)
    cbar.ax.tick_params(labelsize=label_size)
    cbar.ax.locator_params(nbins=8)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_axis_off()
    fig.tight_layout()

    pic_dis = '../plots/total_error_map.pdf'
    plt.savefig(pic_dis, dpi=600,
                transparent=True,
                bbox_inches='tight', pad_inches=0)

    # Calculate RMSE of non-zero elements
    total_non_zero = total_error_map[total_error_map != 0]
    MSE = mean_squared_error(total_non_zero, np.zeros_like(total_non_zero))
    RMSE = math.sqrt(MSE)

    # Find the percentage of pixels less than RMSE
    percentage_RMSE = ((total_non_zero <= 40.0).sum() / len(total_non_zero))*100
