# -*- coding: utf-8 -*-
# @Time    : 2022-08-13 15:03
# @Author  : young wang
# @FileName: FOV_circle.py
# @Software: PyCharm

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

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 13.5,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/MEEI/FOV/circle/geometric/original/*.oct'))
    for k in range(len(data_sets)):
        data = load_from_oct_file(data_sets[k])
        volume = pre_volume(data, low=1, inner_radius=50,
                            edge_radius=230)

        re_volume = clean_small_object(volume)
        edge_vol = obtain_inner_edge(re_volume)

        index_list = [50, 165, 229]
        title_list = ['top', 'middle', 'bottom']

        fig, axs = plt.subplots(1, len(index_list), figsize=(16, 9))
        for n, (ax, idx, title) in enumerate(zip(axs.flat, index_list, title_list)):
            edge = edge_vol[:, :, idx]
            img = volume[:, :, idx]
            masked = np.ma.masked_where(edge == 1, edge)
            ax.imshow(img, 'gray', vmin=np.max(img) * 0.5, vmax=np.max(img))
            ax.imshow(masked, 'hot', alpha=0.9)

            v_loc, h_loc = wall_index(edge, distance = 100, height = 0.6)
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

        depth_profile = []
        for i in range(volume.shape[-1]):
            edge = edge_vol[:, :, i]
            v_loc, h_loc = wall_index(edge, distance = 100, height = 0.6)
            depth_profile.append((i, v_loc, h_loc))

        pts1, pts2, pts3, pts4 = [], [], [], []
        for i in range(len(depth_profile)):
            idx, vpt, hpt = depth_profile[i]
            try:
                pts1.append((idx, vpt[0][-1]))
                pts2.append((idx, vpt[-1][-1]))
                #
                pts3.append((idx, hpt[0][0]))
                pts4.append((idx, hpt[-1][0]))
            except:
                pass
        print(len(pts1) / volume.shape[-1])

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
        print('done with %d out of %d' % (int(k + 1), len(data_sets)))

