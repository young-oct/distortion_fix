# -*- coding: utf-8 -*-
# @Time    : 2022-07-18 11:43
# @Author  : young wang
# @FileName: flat_surface.py
# @Software: PyCharm

import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from tools.auxiliary import load_from_oct_file
from tools.preprocessing import filter_mask,surface_index,sphere_fit,frame_index

if __name__ == '__main__':

    data_sets = natsorted(glob.glob('../data/1mW/flat surface/*.oct'))

    data = load_from_oct_file(data_sets[-1])
    p_factor = 0.725
    vmin, vmax = int(p_factor*255), 255

    xz_mask = np.zeros_like(data)

    # perform points extraction in the xz direction
    for i in range(data.shape[0]):
        xz_mask[i,:,:] = filter_mask(data[i,:,:],vmin = vmin, vmax = vmax)

    xz_pts = surface_index(xz_mask, dir = 'x')
    # print('done with extracting points from the xz plane')

    yz_mask = np.zeros_like(data)

    for i in range(data.shape[1]):
        yz_mask[:,i,:] = filter_mask(data[:,i,:],vmin = vmin, vmax = vmax)

    yz_pts = surface_index(yz_mask, dir = 'y')

    # print('done with extracting points from the yz plane')

    xz = sphere_fit(xz_pts,centre = None, fixed_origin = False)
    oc_xz = xz.origin

    # print('done with calculating radius and origin for the xz plane')
    yz = sphere_fit(yz_pts,centre = None, fixed_origin = False)
    oc_yz = yz.origin
    # print('done with calculating radius and origin for the yz plane')

    idx = 256
    fig = plt.figure(figsize=(16,9))
    ax1 = fig.add_subplot(221)
    ax1.imshow(xz_mask[idx,:,:], cmap='gray', vmin =vmin, vmax = vmax)
    #plot the segemnetation of slice 256 in xz direction for verification
    xz_slc = frame_index(xz_mask, 'x', idx)
    x, y = zip(*xz_slc)
    ax1.plot(y, x, linewidth=5,alpha = 0.5, color = 'r')
    ax1.set_title('slice from the xz direction',size = 20)

    ax2 = fig.add_subplot(222)

    ax2.imshow(yz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title('slice from the yz direction', size = 20)
    #plot the segemnetation of slice 256 in yz direction for verification
    #
    yz_slc = frame_index(yz_mask, 'y', idx)
    x, y = zip(*yz_slc)
    ax2.plot(y, x, linewidth=5,alpha = 0.5,color = 'r')

    ax3 = fig.add_subplot(223, projection='3d')
    xz.plot(ax3)

    ax4 = fig.add_subplot(224, projection='3d')
    yz.plot(ax4)

    plt.suptitle('last dataset', fontsize=18)

    plt.tight_layout()
    plt.show()
    print('done with estimating the origins')

    for j in range(len(data_sets)-1):
    # for j in range(2):

        data = load_from_oct_file(data_sets[j])
        p_factor = 0.725
        vmin, vmax = int(p_factor*255), 255

        xz_mask = np.zeros_like(data)

        # perform points extraction in the xz direction
        for i in range(data.shape[0]):
            xz_mask[i,:,:] = filter_mask(data[i,:,:],vmin = vmin, vmax = vmax)

        xz_pts = surface_index(xz_mask, dir = 'x')
        # print('done with extracting points from the xz plane')

        yz_mask = np.zeros_like(data)

        for i in range(data.shape[1]):
            yz_mask[:,i,:] = filter_mask(data[:,i,:],vmin = vmin, vmax = vmax)

        yz_pts = surface_index(yz_mask, dir = 'y')

        # print('done with extracting points from the yz plane')

        xz = sphere_fit(xz_pts,centre = oc_xz, fixed_origin = True)
        # print('done with calculating radius and origin for the xz plane')
        yz = sphere_fit(yz_pts,centre = oc_yz, fixed_origin = True)
        # print('done with calculating radius and origin for the yz plane')

        idx = 256
        fig = plt.figure(figsize=(16,9))
        ax1 = fig.add_subplot(221)
        ax1.imshow(xz_mask[idx,:,:], cmap='gray', vmin =vmin, vmax = vmax)
        #plot the segemnetation of slice 256 in xz direction for verification
        xz_slc = frame_index(xz_mask, 'x', idx)
        x, y = zip(*xz_slc)
        ax1.plot(y, x, linewidth=5,alpha = 0.5, color = 'r')
        ax1.set_title('slice from the xz direction',size = 20)

        ax2 = fig.add_subplot(222)

        ax2.imshow(yz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)
        ax2.set_title('slice from the yz direction', size = 20)
        #plot the segemnetation of slice 256 in yz direction for verification
        #
        yz_slc = frame_index(yz_mask, 'y', idx)
        x, y = zip(*yz_slc)
        ax2.plot(y, x, linewidth=5,alpha = 0.5,color = 'r')

        ax3 = fig.add_subplot(223, projection='3d')
        xz.plot(ax3)

        ax4 = fig.add_subplot(224, projection='3d')
        yz.plot(ax4)

        txt_str = 'index %d' % j
        plt.suptitle(txt_str, fontsize=18)

        plt.tight_layout()
        plt.show()
        print('done with extracting points from the xz,yz plane for depth at '
              'index %d' % j)



