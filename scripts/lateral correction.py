
import glob
from natsort import natsorted
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import discorpy.prep.preprocessing as prep

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
    grids = natsorted(glob.glob('../data/correction map/lateral correction/grids/MEEI-v4 - angle-deviation/*.png'))

    uncorrectedGrid = (io.imread(grids[-1],as_gray=True)*255).astype(np.uint8)
    correctedGrid = (io.imread(grids[0], as_gray=True) * 255).astype(np.uint8)

    extractedUncorrectedGrid = ExtractDotGrid(uncorrectedGrid)
    extractedCorrectedGrid = ExtractDotGrid(correctedGrid)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9))
    uncorr_ax = fig.add_subplot(121)
    uncorr_ax.set_xlim(0, correctedGrid.shape[0])
    uncorr_ax.set_ylim(0, correctedGrid.shape[1])
    uncorr_ax.set_ylim(uncorr_ax.get_ylim()[::-1])
    uncorr_ax.set_title('Uncorrected Dot Grid')
    uncorr_ax.set_xlabel('x')
    uncorr_ax.set_ylabel('y')
    uncorr_ax.scatter(extractedUncorrectedGrid[:, 0], extractedUncorrectedGrid[:, 1])

    corr_ax = fig.add_subplot(122)
    corr_ax.set_xlim(0, correctedGrid.shape[0])
    corr_ax.set_ylim(0, correctedGrid.shape[1])
    corr_ax.set_ylim(corr_ax.get_ylim()[::-1])
    corr_ax.set_title('Corrected Dot Grid')
    corr_ax.set_xlabel('x')
    corr_ax.set_ylabel('y')
    corr_ax.scatter(extractedCorrectedGrid[:, 0], extractedCorrectedGrid[:, 1])

    # Normalize dot grids
    uncorrNormalized = (extractedUncorrectedGrid - np.amin(extractedUncorrectedGrid)) / np.ptp(extractedUncorrectedGrid)
    corrNormalized = (extractedCorrectedGrid - np.amin(extractedCorrectedGrid)) / np.ptp(extractedCorrectedGrid)

    grid_y, grid_x = np.mgrid[0:1:512j, 0:1:512j]
    map = griddata(corrNormalized, uncorrNormalized, (grid_y, grid_x), method='cubic', fill_value=0.0).astype(
        'float32')
    x_correction_map = map[:, :, 0].flatten(order='F')
    y_correction_map = map[:, :, 1].flatten(order='F')
    map_size = np.uint32(x_correction_map.size)

    correction_map = np.empty((2 * map_size), dtype=x_correction_map.dtype)
    correction_map[0::2] = x_correction_map
    correction_map[1::2] = y_correction_map
    correction_map_size = np.uint32(correction_map.size / 2)

    # Save correction maps to disk
    output_folder = r'../CorrectionMap.bin'
    # output_filename = join(output_folder, 'middle_ear-processed-volume-TM-only-phase-variance.bin')
    with open(output_folder, 'wb') as f:
        f.write(correction_map_size)
        f.write(correction_map)


    correction_error = corrNormalized - uncorrNormalized

    max_th_error = np.amax(np.abs(correction_error[:,0]))
    max_ph_error = np.amax(np.abs(correction_error[:,1]))

    mean_th_error = np.mean(np.abs(correction_error[:, 0]))
    mean_ph_error = np.mean(np.abs(correction_error[:, 1]))

    std_th_error = np.std(np.abs(correction_error[:, 0]))
    std_phi_error = np.std(np.abs(correction_error[:, 1]))

    var_th_error = np.var(np.abs(correction_error[:, 0]))
    var_phi_error = np.var(np.abs(correction_error[:, 1]))