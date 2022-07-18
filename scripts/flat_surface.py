# -*- coding: utf-8 -*-
# @Time    : 2022-07-18 11:43
# @Author  : young wang
# @FileName: flat_surface.py
# @Software: PyCharm


from matplotlib.patches import Circle
from scipy import ndimage, misc
import glob
from matplotlib import rcParams

import numpy as np
import cv2 as cv
from matplotlib.patches import Circle
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# from tools.circle_fit import sphere_fit
from tools.auxiliary import folder_creator, arrTolist, listtoarr, load_from_oct_file
from tools.preprocessing import filter_mask,surface_index,sphere_fit
from scipy import optimize


if __name__ == '__main__':

    data_sets = glob.glob('../data/1mW/flat surface/*.oct')

    data = load_from_oct_file(data_sets[-1])
    p_factor = 0.65
    vmin, vmax = int(p_factor*255), 255

    xz_mask = np.zeros_like(data)

    # perform points extraction in the xz direction
    for i in range(data.shape[0]):
        xz_mask[i,:,:] = filter_mask(data[i,:,:],vmin = vmin, vmax = vmax)

    xz_pts = surface_index(xz_mask, dir = 'x')
    print('done with extracting points from the xz plane')

    yz_mask = np.zeros_like(data)

    for i in range(data.shape[1]):
        yz_mask[:,i,:] = filter_mask(data[:,i,:],vmin = vmin, vmax = vmax)

    yz_pts = surface_index(yz_mask, dir = 'y')

    print('done with extracting points from the yz plane')

    xz = sphere_fit(xz_pts)
    print('done with calculating radius and origin for the xz plane')
    yz = sphere_fit(yz_pts)
    print('done with calculating radius and origin for the yz plane')

    fig,ax = plt.subplots(1,2, figsize = (16,9))
    ax[0].imshow(np.rot90(xz_mask[256,:,:]), cmap='gray', vmin =vmin, vmax = vmax)
    ax[0].set_title('slice from the xz direction',size = 20)
    ax[1].imshow(np.rot90(yz_mask[:,256,:]),cmap='gray', vmin =vmin, vmax = vmax)
    ax[1].set_title('slice from the yz direction', size = 20)
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    fig.add_subplot(121)

    xz.plot() # the xz direction
    fig.add_subplot(122)

    yz.plot()  # the yz direction
    plt.tight_layout()
    plt.show()

