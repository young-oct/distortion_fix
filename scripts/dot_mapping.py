# -*- coding: utf-8 -*-
# @Time    : 2022-09-18 21:54
# @Author  : young wang
# @FileName: dot_mapping.py
# @Software: PyCharm


from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
import matplotlib.patches as mpatches
import cv2 as cv
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking, mip_stack
import glob
from skimage import transform

import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file, pre_volume, \
    clean_small_object, obtain_inner_edge
import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,
                                white_tophat, disk, black_tophat, square, skeletonize)
from natsort import natsorted
import discorpy.prep.preprocessing as prep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post


def get_featuremap(image, ax, thres=0.8):
    if not ax:
        ax = plt.gca()

    label_image, no = label(image, background=0, return_num=True)
    im = ax.imshow(np.zeros(image.shape), 'gray')

    centroid_list = []
    total_area = []
    for region in regionprops(label_image):
        total_area.append(region.area)

    threshold_area = thres * np.max(total_area)
    for region in regionprops(label_image):

        if region.area > threshold_area:
            y, x = region.centroid
            circle = mpatches.Circle((x, y),
                                     radius=1,
                                     fill=False,
                                     edgecolor='red',
                                     facecolor='white',
                                     linewidth=1)
            centroid_list.append((x, y))
            ax.add_patch(circle)
        else:
            pass
    return im, centroid_list


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    #
    dset_lst = ['../data/image/*.png']
    img_set = []


    data_sets = glob.glob(dset_lst[-1])
    for i in range(len(data_sets)):
        temp = cv.imread(data_sets[i], cv.IMREAD_GRAYSCALE)
        img_set.append(temp)

    #
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    cor_list = []
    for i in range(len(img_set)):
        img1, list1 = get_featuremap(img_set[i], ax[i])
        cor_list.append((i,list1))
    plt.tight_layout()
    plt.show()
