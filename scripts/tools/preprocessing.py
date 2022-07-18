# -*- coding: utf-8 -*-
# @Time    : 2022-03-25 8:18 a.m.
# @Author  : young wang
# @FileName: preprocessing.py
# @Software: PyCharm
"""preprocessing module for geometric correction"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np
import cv2 as cv
from scipy.signal import find_peaks
from matplotlib import rcParams
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def clean(data, top=5, radius=230):
    '''

    :param data: oct 3d data 512x512x330
    :param top: top index to be removed
    :param radius: radius to remove the artfact as a result of scanning
    :return: oct 3d data 512x512x330
    '''

    data[:, :, 0:top] = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.sqrt((i - 256) ** 2 + (j - 256) ** 2) >= radius:
                data[i, j, :] = 0

    return data


def imag2uint(data, lt=0, ut=255):
    '''
    convert pixel data from the 255 range to unit16 range(0-65535)

    :param data:  oct 3d data 512x512x330
    :param lt: lower threshold of pixel values to be removed
    :param ut: upper threshold of pixel values to be removed
    :return: oct 3d data 512x512x330
    '''

    # remove the low and high bounds of the pixel intensity data points
    data = np.clip(data, lt, np.max(data))
    # pixel intensity normalization
    # for detail, please see wiki page
    # https://en.wikipedia.org/wiki/Normalization_(image_processing)

    data = (data - np.min(data)) * ut / (np.max(data) - np.min(data))

    return np.uint16(np.around(data, 0))


def despecking(frame, sigma=0.8, size=3):
    """
    :param frame: 512x 330 or 330x512 oct b mode frame
    :param sigma: sigma for gaussian filter
    :param size: median filter kernel size """

    frame = gaussian_filter(frame, sigma=sigma)

    return median_filter(frame, size=size)


def binary_mask(slice, vmin, vmax):
    ret, msk = cv.threshold(slice, vmin, vmax, cv.THRESH_BINARY)
    krn = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    return cv.bitwise_and(cv.dilate(msk, krn, iterations=100), msk)


def filter_mask(slice, vmin, vmax):
    mask = binary_mask(slice, vmin, vmax)
    mask = median_filter(mask, size=10)
    mask = gaussian_filter(mask, sigma=0.2)
    return mask


class sphere_fit:
    # def __init__(self, pts, dir):
    def __init__(self, pts):

        self.x, self.y, self.z = zip(*pts)
        # if dir == 'x':
        #     self.x, self.y, self.z = zip(*pts)
        #
        # elif dir == 'y':
        #     self.y, self.x, self.z = zip(*pts)
        #
        # else:
        #     print("please enter direction")
        # self.dir = dir
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        self.A = self.form_A()
        self.f = self.form_f()
        self.r, self.o = self.cal_sphere()

    def form_A(self):
        A = np.zeros((len(self.x), 4))
        A[:, 0] = self.x * 2
        A[:, 1] = self.y * 2
        A[:, 2] = self.z * 2
        A[:, 3] = 1

        return A

    def form_f(self):
        #   Assemble the f matrix
        f = np.zeros((len(self.x), 1))
        f[:, 0] = (self.x * self.x) + (self.y * self.y) + (self.z * self.z)
        return f

    #
    def cal_sphere(self):
        c, residules, _, _ = np.linalg.lstsq(self.A, self.f, rcond=None)
        radius = np.sqrt((c[0] * c[0]) + (c[1] * c[1]) + (c[2] * c[2]) + c[3])
        origin = (c[0], c[1], c[2])
        return radius, origin

    def plot(self):
        r = self.r
        x0, y0, z0 = self.o[0], self.o[1], self.o[2]
        # if self.dir == 'x':
        #     x0, y0, z0 = self.o[0], self.o[1], self.o[2]
        # elif self.dir == 'y':
        #     y0, x0, z0 = self.o[0], self.o[1], self.o[2]
        # else:
        #     print("please enter direction")
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v) * r
        y = np.sin(u) * np.sin(v) * r
        z = np.cos(v) * r
        x = x + x0
        y = y + y0
        z = z + z0
        #
        # #   3D plot of Sphere
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.x, self.y, self.z, zdir='z', s=0.1, c='b',alpha=0.3, rasterized=True)
        ax.plot_wireframe(x, y, z, color="r",)
        origin = np.asarray(self.o).flatten()
        ax.set_title('radius = %.2f \n origin(x,y,z) is %s' % (self.r, origin))
        return fig

def surface_index(volume, dir):
    peak_loc = []
    if dir == 'x':
        for i in range(volume.shape[0]):
            slice = volume[i, :, :]
            for j in range(slice.shape[0]):
                a_line = slice[j, :]
                peaks, _ = find_peaks(a_line)
                if len(peaks) != 0:

                    peak_loc.append((i, j, peaks[0]))
                else:
                    pass
    elif dir == 'y':
        for i in range(volume.shape[1]):
            slice = volume[:, i, :]
            for j in range(slice.shape[0]):
                a_line = slice[j, :]
                peaks, _ = find_peaks(a_line)
                if len(peaks) != 0:
                    peak_loc.append((j, i, peaks[0]))
                else:
                    pass
    else:
        print("please enter direction")
    return peak_loc
