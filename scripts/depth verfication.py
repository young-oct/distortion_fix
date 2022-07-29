# -*- coding: utf-8 -*-
# @Time    : 2022-07-14 12:11
# @Author  : young wang
# @FileName: depth verfication.py
# @Software: PyCharm

import glob
import matplotlib.pyplot as plt
from tools.proc import max_slice,filter_mask
from tools.pre_proc import load_from_oct_file

if __name__ == '__main__':

    data = glob.glob('../data/2022.07.12_1mm(3dprint)/trial 2/*.oct')

    # data_decon = load_from_oct_file(data[0])
    data_ori = load_from_oct_file(data[-1])

    index = max_slice(data_ori)
    p_factor = 0.55
    vmin, vmax = int(255 * p_factor), 255

    top, middle , bottom = index, int(index + 5), int(index + 5)
    top_slice = filter_mask(data_ori[:,:,top], vmin, vmax)
    mid_slice = filter_mask(data_ori[:,:,middle], vmin, vmax)
    bot_slice = filter_mask(data_ori[:,:,bottom], vmin, vmax)

    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    ax[0].imshow(top_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[0].set_axis_off()
    ax[0].set_title('top', size=20)

    ax[1].imshow(mid_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[1].set_axis_off()
    ax[1].set_title('middle', size=20)

    ax[2].imshow(bot_slice, 'gray', vmin=vmin, vmax=vmax)
    ax[2].set_axis_off()
    ax[2].set_title('bottom', size=20)

    plt.tight_layout()
    plt.show()

    # mask = binary_mask(mip_slice,vmin, vmax )
    # mask = ndimage.median_filter(mask, size=3)
    # mask = ndimage.gaussian_filter(mask, sigma=0.2)
