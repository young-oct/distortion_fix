
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

if __name__ == '__main__':
    uncorrected_grids = natsorted(glob.glob('../data/later distortion analysis/uncorrected grids/*.png'))
    correction_grids = natsorted(glob.glob('../data/later distortion analysis/correction grids/*.png'))

    error_list_mean = []
    error_list_std = []

    for g in range(1):
        uncorrected_grid = (io.imread(uncorrected_grids[-1], as_gray=True) * 255).astype(np.uint8)
        correction_grid = (io.imread(correction_grids[-1], as_gray=True) * 255).astype(np.uint8)

        extracted_uncorrected_grid = ExtractDotGrid(uncorrected_grid)
        extracted_correction_grid = ExtractDotGrid(correction_grid)

        # plt.style.use('dark_background')
        # fig = plt.figure(figsize=(16, 9))
        # uncorr_ax = fig.add_subplot(121)
        # uncorr_ax.set_xlim(0, correction_grid.shape[0])
        # uncorr_ax.set_ylim(0, correction_grid.shape[1])
        # uncorr_ax.set_ylim(uncorr_ax.get_ylim()[::-1])
        # uncorr_ax.set_title('Uncorrected Dot Grid')
        # uncorr_ax.set_xlabel('x')
        # uncorr_ax.set_ylabel('y')
        # uncorr_ax.scatter(extracted_uncorrected_grid[:, 0], extracted_uncorrected_grid[:, 1])
        #
        # corr_ax = fig.add_subplot(122)
        # corr_ax.set_xlim(0, correction_grid.shape[0])
        # corr_ax.set_ylim(0, correction_grid.shape[1])
        # corr_ax.set_ylim(corr_ax.get_ylim()[::-1])
        # corr_ax.set_title('Corrected Dot Grid')
        # corr_ax.set_xlabel('x')
        # corr_ax.set_ylabel('y')
        # corr_ax.scatter(extracted_correction_grid[:, 0], extracted_correction_grid[:, 1])

        # Normalize dot grids
        #uncorrNormalized = (extracted_uncorrected_grid - (np.amin(extracted_uncorrected_grid))) / np.ptp(extracted_uncorrected_grid)
        #corrNormalized = (extracted_correction_grid - (np.amin(extracted_correction_grid))) / np.ptp(extracted_correction_grid)

        #uncorrNormalized = (extracted_uncorrected_grid) / correction_grid.shape[0]
        #corrNormalized = (extracted_correction_grid) / correction_grid.shape[0]

        uncorrNormalized = (extracted_uncorrected_grid - np.min(extracted_uncorrected_grid)) / np.ptp(extracted_uncorrected_grid)
        corrNormalized = (extracted_correction_grid - np.min(extracted_correction_grid)) / np.ptp(extracted_correction_grid)

        max_theta = 30.022
        max_phi = 31.21

        diffNormalized = np.sqrt(np.square((corrNormalized[:, 0] - uncorrNormalized[:, 0])*max_theta) + np.square((corrNormalized[:, 1] - uncorrNormalized[:, 1])*max_phi))

        # Interpolate grids to produce calibration maps
        grid_y, grid_x = np.mgrid[0:1:512j, 0:1:512j]
        map = griddata(
            corrNormalized, uncorrNormalized,
            (grid_y, grid_x), method='cubic', fill_value=0.0).astype('float32')

        # # Error calculation
        # map_nan = map.copy()
        # map_nan[map_nan == 0] = np.nan
        #
        # error_list_mean.append(np.nanmean(map_nan))
        # error_list_std.append(np.nanstd(map_nan))
        #
        # fig = plt.figure(figsize=(16, 9))
        # ax = fig.add_subplot()
        # ax.set_title('Difference heat map', size=15)
        # im, cbar = heatmap(map, ax=ax,
        #                    cmap="hot", cbarlabel='angle variation')
        #
        # pic_dis = '../plots/lateral_error_map.pdf'
        # # plt.axis('off')
        #
        # plt.savefig(pic_dis, dpi=600,
        #             transparent=True,
        #             bbox_inches='tight', pad_inches=0)
        ######################## Plot the heatmap #####################
        # fig, ax = plt.subplots(1, figsize=(16, 10))
        # im = ax.imshow(np.abs(map), cmap="hot")
        #
        # # Clip to ROI
        # # corners = [[21, 15], [21, 496], [477, 496], [477, 15]]
        # # path = Path(corners)
        # # patch = PathPatch(path, facecolor='none', edgecolor='none')
        # #
        # # plt.gca().add_patch(patch)
        # # im.set_clip_path(patch)
        #
        # # Set colour bar axis height
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0)
        #
        # # Create colorbar
        # cbar = ax.figure.colorbar(im, cax=cax)
        # cbar.ax.set_ylabel('Absolute Angular Error (degrees)', rotation=-90, va="bottom", size=20)
        # cbar.ax.tick_params(labelsize=15)
        # cbar.ax.locator_params(nbins=6)
        #
        # ax.set_aspect(1)
        # cax.set_aspect(18.8)
        #
        # # Turn spines off and create white grid.
        # ax.spines[:].set_visible(False)
        # ax.set_axis_off()
        # fig.tight_layout()

        x_correction_map = map[:, :, 0].flatten(order='F')
        y_correction_map = map[:, :, 1].flatten(order='F')
        map_size = np.uint32(x_correction_map.size)

        correction_map = np.empty((2 * map_size), dtype=x_correction_map.dtype)
        correction_map[0::2] = x_correction_map
        correction_map[1::2] = y_correction_map
        correction_map_size = np.uint32(correction_map.size / 2)
    #
    # Save correction maps to disk
    output_folder = r'../CorrectionMap.bin'
    # output_filename = join(output_folder, 'middle_ear-processed-volume-TM-only-phase-variance.bin')
    with open(output_folder, 'wb') as f:
        f.write(correction_map_size)
        f.write(correction_map)
    #
    #
    # correction_error = corrNormalized - uncorrNormalized
    #
    # max_th_error = np.amax(np.abs(correction_error[:,0]))
    # max_ph_error = np.amax(np.abs(correction_error[:,1]))
    #
    # mean_th_error = np.mean(np.abs(correction_error[:, 0]))
    # mean_ph_error = np.mean(np.abs(correction_error[:, 1]))
    #
    # std_th_error = np.std(np.abs(correction_error[:, 0]))
    # std_phi_error = np.std(np.abs(correction_error[:, 1]))
    #
    # var_th_error = np.var(np.abs(correction_error[:, 0]))
    # var_phi_error = np.var(np.abs(correction_error[:, 1]))