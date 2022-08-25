# -*- coding: utf-8 -*-
# @Time    : 2022-07-29 09:54
# @Author  : young wang
# @FileName: proc.py
# @Software: PyCharm
import numpy as np
import os
from tools.proc import clean_removal
from tools.pos_proc import imag2uint
from tools.OssiviewBufferReader import OssiviewBufferReader

def load_from_oct_file_reversed(oct_file, clean = False):
    """
    read .oct file uising OssiviewBufferReader
    export an array in the shape of [512,512,330]
    the aarry contains pixel intensity data(20log) in float16 format
    """
    obr = OssiviewBufferReader(oct_file)
    data_fp16 = np.squeeze(obr.data)
    data = np.flip(imag2uint(data_fp16),axis=2)

    if clean:
        data = clean_removal(data)
    else:
        pass
    return data

def frame_index_reverse(volume, dir, index, shift=0):
    peak_loc = []

    if dir == 'x':
        slice = volume[index, :, :]
        for i in range(slice.shape[0]):
            a_line = slice[i, :]
            peaks = np.where(a_line == 255)
            if len(peaks[0]) >= 1:
                peak_loc.append((i, int(peaks[0][0] - shift)))
            else:
                pass
    elif dir == 'y':
        slice = volume[:, index, :]
        for i in range(slice.shape[0]):
            a_line = slice[i, :]
            peaks = np.where(a_line == 255)
            if len(peaks[0]) >= 1:
                peak_loc.append((i, int(peaks[0][0] - shift)))
            else:
                pass
    else:
        print('please enter the correct direction')
    return peak_loc

def surface_index_reverse(volume, shift=0):
    '''get the index for the peaks in the each volume'''

    peak_loc = []
    for i in range(volume.shape[0]):
        slice = volume[i, :, :]
        for j in range(slice.shape[0]):
            a_line = slice[j, :]
            peaks = np.where(a_line == 255)
            if len(peaks[0]) >= 1:
                peak_loc.append((i, j, int(peaks[0][0] - shift)))
            else:
                pass

    return peak_loc