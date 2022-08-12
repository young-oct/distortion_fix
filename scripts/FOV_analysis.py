# -*- coding: utf-8 -*-
# @Time    : 2022-08-12 17:28
# @Author  : young wang
# @FileName: FOV_analysis.py
# @Software: PyCharm

import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import surface_index, frame_index, plane_fit,filter_mask
from tools.pos_proc import export_map
import pyransac3d as pyrsc
from skimage import feature
from skimage import filters

import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking,slice_index,binary_mask
from tools.pos_proc import export_map
from tools.pre_proc import folder_creator,load_from_oct_file
from tools.pos_proc import oct_to_dicom,imag2uint
from os.path import join
import os
from tools.pos_proc import convert
import cv2 as cv
from scipy.ndimage import median_filter,gaussian_filter
import matplotlib
from skimage import feature
from tools.plot import angle_est,heatmap
import glob
import numpy as np
import copy
import matplotlib.pyplot as plt
from tools.pos_proc import image_export
from tools.pre_proc import load_from_oct_file
from tools.proc import map_index, max_slice
import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk, black_tophat, square, skeletonize)
from natsort import natsorted


from scipy.signal import find_peaks


def wall_index(img):
    mid_index = int(img.shape[0]/2)
    v_loc, h_loc = [], []
    v_line = img[:, mid_index]
    h_line = img[mid_index,:]

    pk_heights = np.max(img) * 0.4

    vpeaks, _ = find_peaks(v_line, distance=100,height = pk_heights )
    hpeaks, _ = find_peaks(h_line, distance=100,height = pk_heights)

    if len(vpeaks) >=2 and len(hpeaks) >=2:
        v_loc.append((mid_index,vpeaks[0]))
        v_loc.append((mid_index,vpeaks[-1]))

        h_loc.append((hpeaks[0],mid_index))
        h_loc.append((hpeaks[-1],mid_index))

    else:
        pass

    return v_loc,h_loc

def circle_cut(img,cut_ori = (256,256), radius =40):
    x, y = cut_ori

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            criteria = np.sqrt((i-x) ** 2 + (j-y) ** 2) - radius
            if criteria < 0:
                img[i,j] = 0
            else:
                pass
    return img

def pre_volume(volume,p_factor = 0.55):

    new_volume = np.zeros(volume.shape)
    vmin, vmax = int(p_factor * 255), 255

    for i in range(volume.shape[-1]):

        temp_slice = volume[:,:,i]
        temp_slice = circle_cut(temp_slice)
        _,bw = cv.threshold(temp_slice, vmin, vmax, cv.THRESH_BINARY)
        edge_sobel = filters.sobel(bw)
        temp = despecking(edge_sobel, sigma=3, size=10)
        new_volume[:,:,i] = temp
    return convert(new_volume,0,255,np.float64)

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/MEEI/FOV/*.oct'))
    data = load_from_oct_file(data_sets[-1])

    volume = pre_volume(data)

    index_list = [0,256,329]
    title_list = ['top', 'middle','bottom']

    fig, axs = plt.subplots(1, len(index_list), figsize=(16, 9))
    for n, (ax, idx, title) in enumerate(zip(axs.flat, index_list,title_list)):
        temp = volume[:,:,idx]
        ax.imshow(temp, 'gray')
        v_loc,h_loc = wall_index(temp)
        for vpts in v_loc:
            ax.plot(vpts[0], vpts[1], 'o', ms=10, c='blue')
        for hpts in h_loc:
            ax.plot(hpts[0], hpts[1], 'o', ms=10, c='red')
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    depth_profile = []
    for i in range(volume.shape[-1]):
        temp = volume[:, :, i]
        v_loc, h_loc = wall_index(temp)
        depth_profile.append((i,v_loc, h_loc))

    print('done')