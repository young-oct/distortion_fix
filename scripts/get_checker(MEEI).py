# -*- coding: utf-8 -*-
# @Time    : 2022-08-16 16:17
# @Author  : young wang
# @FileName: get_checker(MEEI).py
# @Software: PyCharm


import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import max_slice,mip_stack,\
    filter_mask,surface_index,sphere_fit,frame_index
from tools.pos_proc import image_export
from os import path

if __name__ == '__main__':

    data_sets = natsorted(glob.glob('../data/MEEI/checkerboard/geometric/*.oct'))

    dst_folder = '../data/MEEI/checkerboard/Bmode/'
    p_factor = 0.65
    vmin, vmax = int(p_factor*255), 255
    data = []
    for i in range(len(data_sets)):
        if 2 < i < 6:
            data.append(load_from_oct_file(data_sets[i], clean=False))

    for vol in data:
        idx = (max_slice(vol))
        sample_slice = np.amax(vol[:, :, int(idx-30)::], axis=2)

        file_name = path.join(dst_folder,str(idx)+'.png')
        image_export(sample_slice,vmin = vmin, vmax = vmax, filename =file_name)
        plt.show()

