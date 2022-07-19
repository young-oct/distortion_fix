# -*- coding: utf-8 -*-
# @Time    : 2022-07-18 21:11
# @Author  : young wang
# @FileName: grid_export.py
# @Software: PyCharm


import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.auxiliary import load_from_oct_file
from tools.preprocessing import filter_mask,surface_index,sphere_fit,frame_index

if __name__ == '__main__':

    data_sets = glob.glob('../data/1mW/1mm grid/*.oct')

    data = load_from_oct_file(data_sets[0])
    p_factor = 0.725
    vmin, vmax = int(p_factor*255), 255