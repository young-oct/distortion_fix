# -*- coding: utf-8 -*-
# @Time    : 2022-07-22 13:44
# @Author  : young wang
# @FileName: axial_discrepancy.py
# @Software: PyCharm

import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from tools.auxiliary import load_from_oct_file
from tools.preprocessing import filter_mask,surface_index,frame_index,plane_fit
import pyransac3d as pyrsc

def heatmap(data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

if __name__ == '__main__':

    data_sets = natsorted(glob.glob('../data/1mW/flat surface(correctd)/*.oct'))

    data = load_from_oct_file(data_sets[-1])
    p_factor = 0.75
    vmin, vmax = int(p_factor*255), 255

    xz_mask = np.zeros_like(data)

    # perform points extraction in the xz direction
    for i in range(data.shape[0]):
        xz_mask[i,:,:] = filter_mask(data[i,:,:],vmin = vmin, vmax = vmax)

    xz_pts = surface_index(xz_mask, dir = 'x')

    yz_mask = np.zeros_like(data)

    for i in range(data.shape[1]):
        yz_mask[:,i,:] = filter_mask(data[:,i,:],vmin = vmin, vmax = vmax)

    yz_pts = surface_index(yz_mask, dir = 'y')

    fig = plt.figure(figsize=(16, 9))

    idx = 256
    ax = fig.add_subplot(221)
    ax.imshow(xz_mask[idx,:,:], cmap='gray', vmin =vmin, vmax = vmax)

    xz_slc = frame_index(xz_mask, 'x', idx)
    x, y = zip(*xz_slc)
    ax.plot(y, x, linewidth=5,alpha = 0.5, color = 'r')
    ax.set_title('slice %d from the xz direction'% idx,size = 16)

    ax = fig.add_subplot(222)

    ax.imshow(yz_mask[idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    yz_slc = frame_index(yz_mask, 'y', idx)
    x, y = zip(*yz_slc)
    ax.plot(y, x, linewidth=5,alpha = 0.5,color = 'r')
    ax.set_title('slice %d from the yz direction'% idx,size = 16)
    #plot the segemnetation of slice 256 in yz direction for verification

    ax = fig.add_subplot(223, projection='3d')
    xp,yp,zp = zip(*xz_pts)
    ax.scatter(xp, yp, zp,s = 0.5, alpha = 0.1)
    ax.set_title('raw points cloud in xz direction',size = 20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    ax.set_zlim([0, 330])

    ax = fig.add_subplot(224, projection='3d')
    xp,yp,zp = zip(*yz_pts)
    ax.scatter(xp, yp, zp,s = 0.5, alpha = 0.1, c = 'r')
    ax.set_title('raw points cloud in yz direction',size = 15)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    ax.set_zlim([0, 330])

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(16, 9))
    # z_low, z_high = 30, 70
    ax = fig.add_subplot(331, projection='3d')
    xp,yp,zp = zip(*yz_pts)
    ax.scatter(xp, yp, zp,s = 0.5, alpha = 0.5, c = 'r')

    ideal_plane = pyrsc.Plane()
    pts = np.asarray(yz_pts)

    idx_x = np.setdiff1d(np.arange(0,512),pts[:,0])
    idx_y = np.setdiff1d(np.arange(0,512),pts[:,1])


    best_eq, best_inliers = ideal_plane.fit(pts, 0.01)

    a, b, c, d = best_eq[0], best_eq[1], -best_eq[2], best_eq[3]

    xx, yy = np.meshgrid(np.arange(0, 512, 1), np.arange(0, 512, 1))
    z_ideal = (d - a * xx - b * yy) / c

    surf = ax.plot_wireframe(xx, yy, z_ideal,alpha = 0.2)

    ax.set_title('raw points cloud in yz direction \n'
                 '& ideal plane',size = 20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    z_mean = np.mean(z_ideal)
    z_low, z_high = int(z_ideal - 30), int(z_ideal + 30)
    ax.set_zlim([z_low, z_high])

    ax = fig.add_subplot(332, projection='3d')
    ax.set_title('raw points cloud in yz direction \n'
                 '& linearly fitted plane',size = 15)

    l_plane = plane_fit(yz_pts,order=1).zc
    ax.scatter(xp, yp, zp,s = 0.5, alpha = 0.5, c = 'r')

    plane_fit(yz_pts,order=1).plot(ax,z_low,z_high)

    ax = fig.add_subplot(335, projection='3d')
    dl_map = l_plane - z_ideal
    surf = ax.plot_wireframe(xx, yy, dl_map,alpha = 0.2)

    ax = fig.add_subplot(338)
    im, cbar = heatmap(dl_map, ax=ax,
                       cmap="hot", cbarlabel='depth variation')

    ax = fig.add_subplot(333, projection='3d')
    ax.set_title('raw points cloud in yz direction \n'
                 '& quadratically fitted plane',size = 15)

    q_plane = plane_fit(yz_pts,order=2).zc
    ax.scatter(xp, yp, zp,s = 0.5, alpha = 0.5, c = 'r')
    plane_fit(yz_pts,order=2).plot(ax,z_low,z_high)

    ax = fig.add_subplot(336, projection='3d')
    dq_map = q_plane - z_ideal
    surf = ax.plot_wireframe(xx, yy, dq_map,alpha = 0.2)

    ax = fig.add_subplot(339)
    im, cbar = heatmap(dq_map, ax=ax,
                       cmap="hot", cbarlabel='depth variation')

    plt.tight_layout()
    plt.show()

    print('the standard deviation obtained from the linear plane is %.2f' % np.std(dl_map))
    print('the standard deviation obtained from the quadratic plane is %.2f' % np.std(dq_map))
