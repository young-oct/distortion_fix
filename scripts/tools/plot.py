# -*- coding: utf-8 -*-
# @Time    : 2022-08-02 17:06
# @Author  : young wang
# @FileName: plot.py
# @Software: PyCharm

import numpy as np

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