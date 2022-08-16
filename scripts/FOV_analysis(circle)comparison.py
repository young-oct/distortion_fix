# -*- coding: utf-8 -*-
# @Time    : 2022-08-14 15:25
# @Author  : young wang
# @FileName: FOV_analysis(circle)comparison.py
# @Software: PyCharm

import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file,pre_volume,\
    clean_small_object,obtain_inner_edge
from tools.proc import wall_index
import matplotlib
from tools.pos_proc import image_export
from tools.plot import line_fit_plot
from tools.proc import line_fit
from natsort import natsorted


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    start = time.perf_counter()

    dset_lst = ['../data/MEEI/FOV/circle/raw/original/*.oct',
                '../data/MEEI/FOV/circle/geometric/original/*.oct']

    dataset = []
    edge_vol = []
    for i in range(len(dset_lst)):

        data_sets = natsorted(glob.glob(dset_lst[i]))
        data = load_from_oct_file(data_sets[-1])
        dataset.append(data)

        volume = pre_volume(data, p_factor = 0.6)

        re_volume = clean_small_object(volume)
        edge_volume = obtain_inner_edge(re_volume)
        edge_vol.append(edge_volume)

    index_list = [50, 165, 300, 50, 165, 300]
    title_list = ['raw_top', 'raw_middle', 'raw_bottom',
                  'geo_top','geo_middle','geo_bottom']

    fig, axs = plt.subplots(2, int(len(index_list)/2), figsize=(16, 9))
    for n, (ax, idx, title) in enumerate(zip(axs.flat,
                                         index_list, title_list)):

        edge_img = edge_vol[n//3][:,:,idx]
        img = dataset[n//3][:,:,idx]
        ax.imshow(img, 'gray', vmin = 0.55*np.max(img), vmax = np.max(img))
        ax.contour(edge_img[:,:-1], levels=[-.1, .1], colors='red',
                   linestyles='--',alpha =0.5)

        v_loc, h_loc = wall_index(edge_img, distance = 100, height = 0.6)
        vpts_list = ['pt1', 'pt2']
        hpts_list = ['pt3', 'pt4']

        for n, (vpts, vpts_t) in enumerate(zip(v_loc, vpts_list)):
            ax.plot(vpts[0], vpts[1], 'o', ms=10, c='blue')
            ax.text(vpts[0] * 1.1, vpts[1] * 0.9, vpts_t,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transData, size=25, color='red')
        for n, (hpts, vpts_t) in enumerate(zip(h_loc, hpts_list)):
            ax.plot(hpts[0], hpts[1], 'o', ms=10, c='red')
            ax.text(hpts[0] * 0.9, hpts[1] * 0.9, vpts_t,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transData, size=25, color='red')
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    for k in range(len(edge_vol)):

        fig_name = dset_lst[k].split('/')[-3]
        e_vol = edge_vol[k]

        depth_profile = []
        for i in range(e_vol.shape[-1]):
            edge = e_vol[:, :, i]
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
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        piont_list = [pts1, pts2, pts3, pts4]
        for n, (ax, pts, ylt, title) in enumerate(zip(axs.flat, piont_list, y_list,title_list)):
            line_fit_plot(pts, ylt, ax, order=1)
            ax.set_title(title)
        fig.suptitle(fig_name + ': absolute distance map')
        plt.tight_layout()

        plt.show()

        pt1_rel,pt2_rel,pt3_rel,pt4_rel = [], [],[],[]

        ref_idex = 0
        pt1_ref = pts1[ref_idex][-1]
        pt2_ref = pts2[ref_idex][-1]

        pt3_ref = pts3[ref_idex][-1]
        pt4_ref = pts4[ref_idex][-1]

        for i in range(len(pts1)):
            pt1_rel.append((i, np.abs(pts1[i][-1] - pt1_ref)))
            pt2_rel.append((i, np.abs(pts2[i][-1] - pt2_ref)))
            pt3_rel.append((i, np.abs(pts3[i][-1] - pt3_ref)))
            pt4_rel.append((i, np.abs(pts4[i][-1] - pt4_ref)))

        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        piont_list_rel = [pt1_rel, pt2_rel, pt3_rel, pt4_rel]
        for n, (ax, pts, ylt, title) in enumerate(zip(axs.flat, piont_list_rel, y_list,title_list)):
            line_fit_plot(pts, ylt, ax, order=1)
            ax.set_title(title)
        fig.suptitle(fig_name +': relative distance map')
        plt.tight_layout()
        plt.show()

        print(fig_name + ':'+ str(100*len(pts1) / e_vol.shape[-1]) + '% of z direction')

    end = time.perf_counter()
    print("Elapsed (after compilation) = %.2fs" % (end - start))

    img_list = [dataset[0][256,:,:],dataset[-1][256,:,:]]
    img_tit_lst = ['before correction', 'geometric correction']
    fig, axs = plt.subplots(1, 2, figsize=(16, 9),constrained_layout=True)
    for n, (ax, img, title) in enumerate(zip(axs.flat, img_list, img_tit_lst)):
        ax.imshow(np.rot90(img), 'gray', vmin = 0.6* np.max(img), vmax =  np.max(img))
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle('cross section comparison')
    plt.show()

