# -*- coding: utf-8 -*-
# @Time    : 2022-03-25 8:18 a.m.
# @Author  : young wang
# @FileName: pre_proc.py
# @Software: PyCharm
"""preprocessing module"""

from skimage import measure,exposure
from skimage.morphology import closing,disk,dilation,square
from scipy import ndimage
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)
import os
from scripts.tools.proc import clean_removal,circle_cut,despecking
from scripts.tools.pos_proc import imag2uint,convert
import numpy as np
from scripts.tools.OssiviewBufferReader import OssiviewBufferReader

def folder_creator(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        # directory already exists
        pass

def load_from_oct_file(oct_file, clean = False):
    """
    read .oct file uising OssiviewBufferReader
    export an array in the shape of [512,512,330]
    the aarry contains pixel intensity data(20log) in float16 format
    """
    obr = OssiviewBufferReader(oct_file)
    data_fp16 = np.squeeze(obr.data)

    data = imag2uint(data_fp16)

    if clean:
        data = clean_removal(data)
    else:
        pass
    return data

def arrTolist(volume, Yflag=False):
    '''
    convert volume array into list for parallel processing
    :param volume: complex array
    :return:
    volume_list with 512 elements long, each element is Z x X = 330 x 512
    '''

    volume_list = []

    if not Yflag:
        for i in range(volume.shape[0]):
            volume_list.append(volume[i, :, :])
    else:
        for i in range(volume.shape[1]):
            volume_list.append(volume[:, i, :])

    return volume_list

def listtoarr(volume_list, Yflag=False):
    '''convert the volume list back to array format
    :param volume_list: complex array
    :return:
    volume with 512 elements long, each element is Z x X = 330 x 512
    '''

    if not Yflag:
        volume = np.empty((len(volume_list), 512, 330))
        for i in range(len(volume_list)):
            volume[i, :, :] = volume_list[i]
    else:
        volume = np.empty((512, len(volume_list), 330))
        for i in range(len(volume_list)):
            volume[:, i, :] = volume_list[i]

    return volume


def pre_volume(volume,low = 2, inner_radius=50, edge_radius = 240):
    high = 100 - low
    new_volume = np.zeros_like(volume)
    p_factor = np.mean(volume)/np.max(volume)
    vmin, vmax = int(p_factor * 255), 255
    c_volume = circle_cut(volume,
                            inner_radius=inner_radius,
                            edge_radius=edge_radius)
    c_volume = np.where(c_volume <= vmin, vmin, c_volume)

    for i in range(volume.shape[-1]):
        temp_slice = c_volume[:, :, i]
        # temp_slice = circle_cut(temp_slice,
        #                         inner_radius = inner_radius,
        #                         edge_radius= edge_radius)

        temp = despecking(temp_slice, sigma=2, size=5)
        # temp_slice = np.where(temp <= vmin, vmin, temp)

        low_p, high_p = np.percentile(temp, (low, high))
        temp_slice = exposure.rescale_intensity(temp,
                                          in_range=(low_p, high_p))
        # temp = closing(temp_slice, diamond(20))
        new_volume[:, :, i] = closing(temp_slice, diamond(20))

    new_volume = np.where(new_volume < np.mean(new_volume), 0, 255)

    return convert(new_volume, 0, 255, np.float64)


def clean_small_object(volume):
    new_volume = np.zeros_like(volume)
    for i in range(volume.shape[-1]):
        c_slice = volume[:,:,i]
        label_im, nb_labels = ndimage.label(c_slice)
        sizes = ndimage.sum(c_slice, label_im, range(nb_labels + 1))

        mask_size = sizes < np.max(sizes) * 0.5
        remove_pixel = mask_size[label_im]

        label_im[remove_pixel] = 0
        new_volume[:, :, i] = label_im
    return new_volume

def obtain_inner_edge(volume):

    iedge_volume = np.zeros_like(volume)
    for i in range(volume.shape[-1]):
        c_slice = volume[:,:,i]
        contours = measure.find_contours(c_slice)
        #1 is the inner edge, 0 is the outer edge
        edge_arr = np.zeros_like(c_slice)
        try:
            for j in range(len(contours[1]) - 1):
                x, y = contours[1][j]
                edge_arr[int(x), int(y)] = 255
        except:
            pass

        iedge_volume[:,:,i] = edge_arr
    return iedge_volume
