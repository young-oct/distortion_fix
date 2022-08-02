# -*- coding: utf-8 -*-
# @Time    : 2022-08-02 17:06
# @Author  : young wang
# @FileName: plot.py
# @Software: PyCharm

import numpy as np
import matplotlib as plt

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
    ax.set_axis_off()

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title('std: %.2f' % np.std(data), y=0, pad=-14)
    ax.xaxis.set_label_position('top')
    # ax.set_xlabel('std: %.2f' % np.std(data))
    # ax.set_title('Manual y', y=1.0, pad=-14)

    return im, cbar

def angle_est(x, y, origin, radius, ax):
    xmin_idx, xmax_idx = np.argmin(x), np.argmax(x)
    ymin, ymax = y[xmin_idx], y[xmax_idx]
    xc, yc = origin[0], origin[1]

    ax.plot(xmin_idx, ymin, color='green', label='x1', marker='8', ms=10)
    ax.plot(xmax_idx, ymax, color='green', label='x2', marker='8', ms=10)

    angle_1 = np.degrees(np.arcsin(abs(xc - xmin_idx) / radius))
    angle_2 = np.degrees(np.arcsin(abs(xc - xmax_idx) / radius))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    textstr = '\n'.join((
        r'$\theta_2=%.2f$' % (angle_1,),
        r'$\theta_2=%.2f$' % (angle_2,)))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax.annotate('x1', xy=(xmin_idx, ymin), xycoords='data',
                xytext=(xmin_idx - radius / 4, ymin + radius / 4), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )

    ax.annotate('x2', xy=(xmax_idx, ymax), xycoords='data',
                xytext=(xmax_idx + radius / 4, ymax + radius / 4), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )

    return ax