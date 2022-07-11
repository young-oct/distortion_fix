# -*- coding: utf-8 -*-
# @Time    : 2022-03-25 8:59 a.m.
# @Author  : young wang
# @FileName: geocrt.py
# @Software: PyCharm

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import math
from numba import jit


@jit(nopython=True)
def getPolarco(f_zmax = 1.7, degree =10.5):
    '''obtian correct polar coordinates from the distorted image

    since X and Y correction can be done independently with respect to Z,
    here we replace X_dim, Y_dim mentioend in Josh's proposal as i_dim
    for detailed math, see johs's proposal
    we can do this because
    (1) i_dim = X_dim = Y_dim = 512
    (2) azimuth and elevation are roughly the same 10 degrees (according to Dan)
    (3) 3D geometirc correction can be decomposed into two independent 2D correction
    please see "Real-time correction of geometric distortion artifact
     in large-volume optical coherence tomography paper'''

    i_dim, zdim, zmax = 512, 330, int(330 *f_zmax)

    _iz = np.zeros((i_dim, zdim, 2))  # construct iz plane
    i0, z0 = int(i_dim / 2), zmax  # i0 is half of the i dimension

    i_phi = math.radians(degree)  # converting from degree to radiant

    ki = i_dim / (2 * i_phi)  # calculate pixel scaling factor for i dimension
    # kz = 1.5 # calculate pixel scaling factor for z dimension, it should be Zmax/D, this is
    # a magic number kind works,
    kz = 1
    for i in range(i_dim):
        for z in range(zdim):  # pixel coordinates conversion
            _iz[i, z, :] = [
                (z + kz * z0) * math.sin((i - i0) / ki) * math.cos((i - i0) / ki) + i0,
                (z + kz * z0) * math.cos((i - i0) / ki) * math.cos((i - i0) / ki) - kz * z0]

        # _iz.reshape(i_dim * zdim, 2): numpy stores arrays in row-major order
        # This means that the resulting two-column array will first contain all the x values,
        # then all the y values rather than containing pairs of (x,y) in each row
    _iz = _iz.reshape(i_dim * zdim, 2)
    return _iz

@jit(nopython=True)
def valueRemap(dis_image):
    """remap the data to match with the correct orientation"""

    _v = np.zeros(dis_image.shape)
    for i in range(dis_image.shape[0]):
        for z in range(dis_image.shape[1]):  # pixel coordinates conversion

            _v[i, z] = dis_image[i, -z]  # store the pixel date temporally and flip along the colume
            # axis
    return np.ravel(_v)

def polar2cart(tri, xq, zq, values):
    values = valueRemap(values)

    """interpolate values from the target grid points"""

    # initilize interpolator
    interpolator = LinearNDInterpolator(tri, values)

    # interpolate values from from with respect to the targeted
    # cartisan coordinates
    valueUpdate = interpolator(xq, zq)

    return np.fliplr(valueUpdate)
    # return valueUpdate

def iniTri(polrcoordinate):
    '''initialize triangulation'''
    return Delaunay(polrcoordinate)
