# -*- coding: utf-8 -*-
# @Time    : 2022-08-15 02:30
# @Author  : young wang
# @FileName: test.py
# @Software: PyCharm
import copy
import time
import glob
import numpy as np
from skimage.filters import threshold_otsu
import discorpy.prep.preprocessing as prep

import matplotlib.pyplot as plt
from skimage.morphology import closing,disk,dilation,square,black_tophat,white_tophat
from tools.pos_proc import convert
from tools.pre_proc import load_from_oct_file
from tools.proc import wall_index,despecking,gaussian_filter,binary_mask
import matplotlib
from scipy import ndimage
from skimage import exposure


from tools.plot import line_fit_plot
from tools.proc import line_fit
from natsort import natsorted
from numba import njit
from skimage import exposure,measure
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)
import copy
@njit
def circle_cut(vol: float,cut_ori = (256,256), inner_radius =40, edge_radius = 250) -> float:
    x, y = cut_ori
    # fill_pix = np.median(vol)

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


@njit
def select_wall(vol: float,pmax: float,pmin: float) -> float:
    max_val = np.max(vol)*pmax
    min_val = np.max(vol)*pmin

    for i in range(vol.shape[-1]):
        for j in range(vol.shape[0]):
            for k in range(vol.shape[1]):
                val = vol[j,k, i]
                if min_val < val <= max_val:
                    vol[j, k, i] = 255
                else:
                    pass

    return vol

def pre_volume(volume,p_factor = 0.6):

    new_volume = np.zeros_like(volume)

    vmin, vmax = int(p_factor * 255), 255
    s_vol = np.where(volume <= vmin, 0, 255)

    for i in range(s_vol.shape[-1]):
        temp_img = s_vol[:,:,i]
        temp_img = despecking(temp_img, sigma=1, size=3)
        new_volume[:,:,i] = closing(temp_img, diamond(5))
    return new_volume


def clean_small_object(volume):
    new_volume = np.zeros_like(volume)
    for i in range(volume.shape[-1]):
        c_slice = volume[:,:,i]
        label_im, nb_labels = ndimage.label(c_slice)
        sizes = ndimage.sum(c_slice, label_im, range(nb_labels + 1))

        mask_size = sizes < np.mean(sizes)
        remove_pixel = mask_size[label_im]

        label_im[remove_pixel] = 0
        new_volume[:, :, i] = label_im
    return new_volume

def obtain_inner_edge(volume):

    iedge_volume = np.zeros_like(volume)
    for i in range(volume.shape[-1]):
        c_slice = volume[:,:,i]
        contours = measure.find_contours(c_slice)
        #1 is the inner edge, 0 is the outer edge
        edge_arr = np.zeros_like(c_slice)
        if len(contours) > 1:
            for j in range(len(contours[1])):
                x, y = contours[1][j]
                edge_arr[int(x), int(y)] = 255
        elif len(contours) == 1:
            for j in range(len(contours[-1])):
                x, y = contours[-1][j]
                edge_arr[int(x), int(y)] = 255
                edge_arr[int(x), int(y)] = 255
        else:
            pass

        iedge_volume[:,:,i] = edge_arr
    return iedge_volume

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets_ls = ['../data/MEEI/FOV/circle/raw/original/*.oct',
        '../data/MEEI/FOV/circle/geometric/original/*.oct']
    datas = []
    for i in range(len(data_sets_ls)):

        data_sets = natsorted(glob.glob(data_sets_ls[i]))
        temp = load_from_oct_file(data_sets[-1])
        datas.append(temp)
        datas.append(temp)
        datas.append(temp)

    index_list = [25,165,300,25,165,300]
    raw = copy.deepcopy(datas)
    original_list = []
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for n, (ax, vol, idx) in enumerate(zip(axs.flat, datas, index_list)):
        img = vol[:,:,idx]
        original_list.append(img)
        ax.imshow(img, 'gray', vmin=np.max(img) * 0.5, vmax=np.max(img))

        ax.set_title(str(idx))
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    cir_list = []
    cir_image = []
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for n, (ax, vol, idx) in enumerate(zip(axs.flat, raw,
                                                  index_list)):
        temp_vol = circle_cut(vol)
        cir_list.append(temp_vol)

        e_temp = temp_vol[:,:,idx]
        cir_image.append(e_temp)
        ax.imshow(e_temp, 'gray', vmin = 0.5*np.max(e_temp),
                  vmax =np.max(e_temp))

        ax.set_title(str(idx))
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    sel_list_vol = []
    sel_lst_img = []
    start = time.perf_counter()
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for n, (ax, vol, idx) in enumerate(zip(axs.flat, cir_list,
                                                  index_list)):

        s_vol = pre_volume(vol, p_factor=0.6)
        sel_list_vol.append(s_vol)
        img = s_vol[:,:,idx]
        sel_lst_img.append(img)
        ax.imshow(img, 'gray', vmin = 0.5*np.max(img),
                  vmax =np.max(img))

        ax.set_title(str(idx))
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    end = time.perf_counter()
    print("Elapsed (after compilation) = %.2fs" % (end - start))

    rel_list_vol = []
    rel_lst_img = []
    start = time.perf_counter()
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for n, (ax, vol, idx) in enumerate(zip(axs.flat, sel_list_vol,
                                                  index_list)):
        # img = closing(img, diamond(5))
        vol = np.where(vol < np.max(vol)*0.6, 0, 255)
        # img = closing(img, diamond(5))
        rel_list_vol.append(vol)

        img = vol[:,:,idx]
        rel_lst_img.append(img)
        ax.imshow(img, 'gray')

        ax.set_title(str(idx))
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    end = time.perf_counter()
    print("Elapsed (after compilation) = %.2fs" % (end - start))

    small_vol = []
    small_img = []
    start = time.perf_counter()
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for n, (ax, vol, idx) in enumerate(zip(axs.flat, rel_list_vol,
                                                  index_list)):

        vol = clean_small_object(vol)
        small_vol.append(vol)
        img = vol[:,:,idx]
        small_img.append(img)
        ax.imshow(img, 'gray')

        ax.set_title(str(idx))
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    end = time.perf_counter()
    print("Elapsed (after compilation) = %.2fs" % (end - start))

    edge_vol = []
    edge_img = []
    start = time.perf_counter()
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for n, (ax, vol, idx) in enumerate(zip(axs.flat, small_vol,
                                                  index_list)):

        vol = obtain_inner_edge(vol)
        edge_vol.append(vol)
        img = vol[:,:,idx]
        small_img.append(img)
        ax.imshow(img, 'gray')

        ax.set_title(str(idx))
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    end = time.perf_counter()
    print("Elapsed (after compilation) = %.2fs" % (end - start))

    for k in range(len(edge_vol)):

        vol = edge_vol[k]
        depth_profile = []
        for i in range(vol.shape[-1]):
            edge = vol[:, :, i]
            v_loc, h_loc = wall_index(edge, distance = 100, height = 0.6)
            depth_profile.append((i, v_loc, h_loc))

        pts1, pts2, pts3, pts4 = [], [], [], []
        for i in range(len(depth_profile)):
            idx, vpt, hpt = depth_profile[i]
            try:
                pts1.append((idx, vpt[0][-1]))
                pts2.append((idx, vpt[-1][-1]))

                pts3.append((idx, hpt[0][0]))
                pts4.append((idx, hpt[-1][0]))
            except:
                pass

        title_list = ['point 1', 'point 2', 'point 3', 'point 4']
        y_list =['y index', 'y index','x index','x index']
        fig, axs = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
        piont_list = [pts1, pts2, pts3, pts4]
        for n, (ax, pts, ylt, title) in enumerate(zip(axs.flat, piont_list, y_list,title_list)):
            line_fit_plot(pts, ylt, ax, order=1)
            ax.set_title(title)
        fig.suptitle('absolute distance map')
        plt.show()

        pt1_rel,pt2_rel,pt3_rel,pt4_rel = [], [],[],[]

        pt1_ref = pts1[0][-1]
        pt2_ref = pts2[0][-1]

        pt3_ref = pts3[0][-1]
        pt4_ref = pts4[0][-1]

        for i in range(len(pts1)):
            pt1_rel.append((i, np.abs(pts1[i][-1] - pt1_ref)))
            pt2_rel.append((i, np.abs(pts2[i][-1] - pt2_ref)))
            pt3_rel.append((i, np.abs(pts3[i][-1] - pt3_ref)))
            pt4_rel.append((i, np.abs(pts4[i][-1] - pt4_ref)))

        fig, axs = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
        piont_list_rel = [pt1_rel, pt2_rel, pt3_rel, pt4_rel]
        for n, (ax, pts, ylt, title) in enumerate(zip(axs.flat, piont_list_rel, y_list,title_list)):
            line_fit_plot(pts, ylt, ax, order=1)
            ax.set_title(title)
        fig.suptitle('relative distance map')
        plt.show()

        print(len(pts1) / vol.shape[-1])

