# -*- coding: utf-8 -*-
# @Time    : 2022-07-18 11:43
# @Author  : young wang
# @FileName: flat_surface.py
# @Software: PyCharm

import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.auxiliary import load_from_oct_file
from tools.preprocessing import filter_mask,surface_index,sphere_fit,frame_index

if __name__ == '__main__':

    data_sets = glob.glob('../data/1mW/flat surface/*.oct')

    data = load_from_oct_file(data_sets[0])
    p_factor = 0.725
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

    idx = 256
    fig,ax = plt.subplots(1,2, figsize = (16,9))
    ax[0].imshow(xz_mask[idx,:,:], cmap='gray', vmin =vmin, vmax = vmax)
    #plot the segemnetation of slice 256 in xz direction for verification
    xz_slc = frame_index(xz_mask, 'x', idx)
    x, y = zip(*xz_slc)
    ax[0].plot(y, x, linewidth=5,alpha = 0.5, color = 'r')
    ax[0].set_title('slice from the xz direction',size = 20)

    ax[1].imshow(yz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title('slice from the yz direction', size = 20)
    #plot the segemnetation of slice 256 in yz direction for verification

    yz_slc = frame_index(yz_mask, 'y', idx)
    x, y = zip(*yz_slc)
    ax[1].plot(y, x, linewidth=5,alpha = 0.5,color = 'r')

    plt.tight_layout()
    plt.show()

    xz.plot() # the xz direction
    plt.tight_layout()
    plt.show()

    yz.plot()  # the yz direction
    plt.tight_layout()
    plt.show()

