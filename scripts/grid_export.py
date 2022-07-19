# -*- coding: utf-8 -*-
# @Time    : 2022-07-18 21:11
# @Author  : young wang
# @FileName: grid_export.py
# @Software: PyCharm


import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.auxiliary import load_from_oct_file
from tools.preprocessing import filter_mask,\
    surface_index,sphere_fit,frame_index,max_slice

if __name__ == '__main__':

    data_sets = glob.glob('../data/1mW/1mm grid/*.oct')

    data = load_from_oct_file(data_sets[0])
    p_factor = 0.65
    vmin, vmax = int(p_factor*255), 255
    index = max_slice(data)
    # plt.imshow(np.rot90(data[:, 256, :]), 'gray', vmin=vmin, vmax=vmax)
    # plt.show()
    #
    # # vmin = 10
    # top, middle , bottom = index, int(index + 5), int(index + 5)
    # top_slice = filter_mask(data[:,:,top], vmin, vmax)
    # mid_slice = filter_mask(data[:,:,middle], vmin, vmax)
    # bot_slice = filter_mask(data[:,:,bottom], vmin, vmax)
    pad = 15
    stack = data[:,:,0:int(index+pad)]
    plt.imshow(stack[256, :, :])
    plt.show()

    top_slice = np.amax(stack, axis=2)
    # stack = data[:,:,index::]
    idx = 256
    #plot the segemnetation of slice 256 in xz direction for verification
    xz_slc = frame_index(data, 'x', idx)
    x, y = zip(*xz_slc)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.imshow(top_slice, 'gray', vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
