# -*- coding: utf-8 -*-
# @Time    : 2022-08-04 18:01
# @Author  : young wang
# @FileName: lateral_fix_update.py
# @Software: PyCharm



import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure
from skimage import restoration
import cv2 as cv
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk, black_tophat, square, skeletonize)

from natsort import natsorted
import discorpy.losa.loadersaver as io
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post
from skimage.morphology import disk, dilation, square, erosion, binary_erosion, binary_dilation, \
    binary_closing, binary_opening, closing

from skimage import feature
import shutil





if __name__ == '__main__':


    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/2022.07.31/original/*.oct'))

    data = []
    for i in range(len(data_sets)):
        data.append(load_from_oct_file(data_sets[i]))

    p_factor = 0.1

    # for i in range(len(data)):
    volume = data[4]

    # access the frame index of where axial location of the checkerboard
    index = surface_index(volume)[-1][-1]
    pad = 10
    stack = volume[:, :, int(index - pad ):int(index)]

    top_slice = np.amax(stack, axis=2)

    # de-speckling for better feature extraction

    # top_slice = despecking(top_slice, sigma=0.5, size=10)

    bi_img = opening(top_slice, square(13))
    bi_img = median_filter(bi_img, size= 5)
    vmin, vmax = int(p_factor * 255), 255

    bi_img = np.where(bi_img <= vmin,vmin, bi_img)
    # create binary image of the top surface
    bi_img = prep.binarization(bi_img)

    bi_img = binary_dilation(bi_img,square(9, dtype=bool))
    top_slice = despecking(top_slice, sigma=1, size=10)

    # bi_img = -bi_img

    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])

    # bi_img = bi_img.astype('uint8')

    # bi_img = cv.filter2D(src=bi_img, ddepth=-1, kernel=kernel)
    img_list = [top_slice, bi_img]

    title_lst = ['original image','binary image']

    # img_list = [top_slice,bi_img]
    vmin, vmax = int(p_factor * 255), 255
    fig, axs = plt.subplots(1, 2, figsize=(16, 9),constrained_layout = True)

    for n, (ax, image,title) in enumerate(zip(axs.flat, img_list,title_lst)):
        ax.imshow(image, 'gray', vmin=np.min(image), vmax=np.max(image))

        ax.set_title(title)
        ax.set_axis_off()
    plt.show()


