# -*- coding: utf-8 -*-
# @Time    : 2022-08-15 01:38
# @Author  : young wang
# @FileName: test.py
# @Software: PyCharm
import numpy as np
from numba import njit

import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file,pre_volume,\
    clean_small_object,obtain_inner_edge
from tools.proc import wall_index
import matplotlib
from tools.plot import line_fit_plot
from tools.proc import line_fit
from natsort import natsorted

@njit
def circle_cut(vol: float,cut_ori = (256,256), inner_radius =40, edge_radius = 250) -> float:
    x, y = cut_ori

    assert vol.ndim == 3

    for i in range(vol.shape[-1]):
        for j in range(vol.shape[0]):
            for k in range(vol.shape[1]):
                radius = np.sqrt((j-x) ** 2 + (k-y) ** 2)

                inner_criteria = radius - inner_radius
                edge_criteria = radius - edge_radius

                if inner_criteria < 0 or edge_criteria > 0:
                    vol[j,k,i] = 0
                else:
                    pass
    return vol

if __name__ == '__main__':
    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )
    data_sets = natsorted(glob.glob('../data/MEEI/FOV/square/original/*.oct'))

    data = load_from_oct_file(data_sets[-1])

    c_map = circle_cut(data)
    plt.imshow(c_map[:,:,0])
    plt.show()