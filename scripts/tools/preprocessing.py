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
    def __init__(self, pts, centre = None, fixed_origin = True):

        self.x, self.y, self.z = zip(*pts)
        self.fixed_origin = fixed_origin

        if fixed_origin:
            self.centre = centre
            self.x = np.array(self.x) - self.centre[0]
            self.y = np.array(self.y) - self.centre[1]
            self.z = np.array(self.z) - self.centre[2]

        else:
            self.x = np.array(self.x)
            self.y = np.array(self.y)
            self.z = np.array(self.z)

        self.A = self.form_A()
        self.B = self.form_B()

        self.radius, self.origin = self.cal_sphere()

    def form_A(self):
        A = np.zeros((len(self.x), 4))
        A[:, 0] = self.x * 2
        A[:, 1] = self.y * 2
        A[:, 2] = self.z * 2
        A[:, 3] = 1

        return A

    def form_B(self):
        #   Assemble the f matrix
        B = np.zeros((len(self.x), 1))
        B[:, 0] = (self.x * self.x) + (self.y * self.y) + (self.z * self.z)
        return B

    #
    def cal_sphere(self):
        c, residules, _, _ = np.linalg.lstsq(self.A, self.B, rcond=None)
        radius = np.sqrt((c[0] * c[0]) + (c[1] * c[1]) + (c[2] * c[2]) + c[3])
        if self.fixed_origin:
            origin = (c[0]+self.centre[0], c[1]+self.centre[1], c[2]+self.centre[2])
        else:
            origin = (c[0], c[1], c[2])

        return radius, origin

    def plot(self, ax):
        x0, y0, z0 = self.origin[0], self.origin[1], self.origin[2]

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v) * self.radius
        y = np.sin(u) * np.sin(v) * self.radius
        z = np.cos(v) * self.radius
        if self.fixed_origin:

            x = x + x0 - self.centre[0]
            y = y + y0 - self.centre[1]
            z = z + z0 - self.centre[2]
        else:
            x = x + x0
            y = y + y0
            z = z + z0

        ax.scatter(self.x, self.y, self.z, zdir='z', s=0.1, c='b',alpha=0.3, rasterized=True)
        ax.plot_wireframe(x, y, z, color="r",)
        origin = np.asarray(self.origin).flatten()
        ax.set_title('radius = %.2f \n origin(x,y,z) is %s' % (self.radius, origin))
        return ax


class plane_fit:

    def __init__(self, pts, order):

        self.x, self.y, self.z = zip(*pts)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)

        self.order = order

        self.A = self.form_A()
        self.B = self.form_B()
        self.xx, self.yy = np.arange(0, 512, 1),np.arange(0, 512, 1)

        self.x_plane, self.y_plane = np.meshgrid(self.xx, self.yy)
        self.zc = self.cal_plane()

    def form_A(self):
        # the  equation for a linear plane is: ax+by+c = z s.t Ax = B
        # A = [x0, y0, 1,......xn, yn, 1].T
        if self.order == 1:
            A = np.c_[self.x, self.y, np.ones(self.x.shape[0])]

        elif self.order == 2:
            # the  equation for a quadratic(n=2) plane is: ax^2+bxy+cy^2+dx+ey+f = z s.t Ax = B
            # A = [x0^2, x0y0, y0^2, x0, y0, 1,......xn^2, xnyn, yn^2, xn, yn, 1].T
            A = np.c_[self.x ** 2, self.x * self.y,  self.y ** 2,
                      self.x, self.y , np.ones(self.x.shape[0])]
        elif self.order == 3:
            # the  equation for a cubic(n=3) plane is:
            # ax^3+ by^3 + cx^2*y + dx*y^2 + ex^2 + fy^2 + gx*y+ hx+iy + j= z s.t Ax = B
            # A = [x0^3, y0^3, x0^2*y0, + x0*y0^2 + x0^2 + y0^2 +
            # x0y0 + x0 + y0 + 1, ...... xn ^ 3, yn ^ 3, xn ^ 2 * yn, + xn * yn ^ 2 + xn ^ 2 +
            # yn ^ 2 + xnyn + xn + yn + 1].T
            A = np.c_[self.x ** 3, self.y ** 3, (self.x**2) * self.y,self.x * (self.y**2),
                      self.x ** 2, self.y ** 2, self.x * self.y +
                      self.x, self.y , np.ones(self.x.shape[0])]
        else:
            pass

        return A


    def form_B(self):

        B = self.z

        return B

    def cal_plane(self):

        c, _, _, _ = np.linalg.lstsq(self.A, self.B, rcond=None)  # coefficients

        # x = [a, b, c].T
        if self.order == 1:
            zc = c[0] * self.x_plane + c[1] * self.y_plane + c[2]

        # x = [a, b, c, d, e, f].T
        elif self.order == 2:
            zc = c[0]*self.x_plane**2 + \
                 c[1]*self.x_plane*self.y_plane+\
                 c[2]*self.y_plane**2 +c[3]*self.x_plane+\
                 c[4]*self.y_plane+c[5]

        # x = [a, b, c, d, e, f, h, i, j].T
        elif self.order == 3:

            zc = c[0]*self.x_plane**3 + c[1]*self.y_plane**3 + c[2]*(self.x_plane**2) * self.y_plane + \
                 c[3] * self.x_plane * (self.y_plane ** 2) + c[4]*self.x_plane**2 + c[5]*self.y_plane**2 + \
                 c[6]* self.x_plane * self.y_plane + c[7]+self.x_plane+ c[8]*self.y_plane+c[9]

            pass
        return zc


    def plot(self, ax, low = 0, high = 330):

        ax.scatter(self.x, self.y, self.z, zdir='z', s=0.1, c='b',alpha=0.3, rasterized=True)
        ax.plot_surface(self.x_plane, self.y_plane, self.zc, color='r',alpha=0.5)

        ax.set_zlabel('z')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([0, 512])
        ax.set_ylim([0, 512])
        ax.set_zlim([low, high])
        return ax



def frame_index(volume, dir, index):
    if dir == 'x':
        slice = volume[index,:,:]
    elif dir == 'y':
        slice = volume[:,index,:]
    else:
        print("please enter direction")
    '''get the index for the peaks in the each slice'''
    peak_loc = []
    for i in range(slice.shape[0]):
        a_line = slice[i, :]
        peaks, _ = find_peaks(a_line)
        if len(peaks) != 0:

            peak_loc.append((i, peaks[0]))
        else:
            pass
    return peak_loc


def surface_index(volume, dir):
    '''get the index for the peaks in the each volume'''

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

def max_slice(volume):
    # take a volume and find the index of the maximum intensity
    # slice
    assert volume.ndim == 3

    slice = np.sum(volume, axis=0)
    line = np.sum(slice, axis=0)

    return np.argmax(line)

def mip_stack(volume, index, thickness):
    assert volume.ndim == 3

    low_b, high_b = int(index - thickness), int(index + thickness)

    if low_b >= 0 or high_b <= volume.shape[-1]:
        return np.amax(volume[:, :, low_b::high_b], axis=2)

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img