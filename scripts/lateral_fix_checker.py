# -*- coding: utf-8 -*-
# @Time    : 2022-07-31 20:27
# @Author  : young wang
# @FileName: lateral_fix_checker.py
# @Software: PyCharm


import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure
from skimage import restoration
import cv2 as cv
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk, black_tophat, square, skeletonize)

from natsort import natsorted
import discorpy.losa.loadersaver as io
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post
from skimage.morphology import disk, dilation, square, erosion, binary_erosion, binary_dilation, \
    binary_closing, binary_opening, closing

from skimage import feature


def locate_points(image, radius=15, ratio=0.5, sen=0.05, nosie_le=1.5):
    binary_map = prep.binarization(image)

    img = lprep.convert_chessboard_to_linepattern(binary_map)

    # Calculate slope and distance between lines
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(img, radius=radius, sensitive=sen)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(img, radius=radius, sensitive=sen)

    # Extract reference-points
    list_points_hor_lines = lprep.get_cross_points_hor_lines(img, slope_ver, dist_ver,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr="bright", radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)

    list_points_ver_lines = lprep.get_cross_points_ver_lines(img, slope_hor, dist_hor,
                                                             ratio=ratio, norm=True, offset=0,
                                                             bgr="bright", radius=radius,
                                                             sensitive=nosie_le, denoise=True,
                                                             subpixel=True)
    return list_points_hor_lines, list_points_ver_lines


def locate_center(image, cut=False, radius=100, sigma=4):
    edges = feature.canny(image, sigma=sigma)
    edges = edges.astype(np.float32)

    M = cv.moments(edges)
    # calculate x,y coordinate of center
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    if cut:
        for k in range(image.shape[0]):
            for j in range(image.shape[1]):
                x = k - cX
                y = j - cY
                r = np.sqrt(x ** 2 + y ** 2)
                if r > radius:
                    image[k, j] = 0
                else:
                    pass
        return image, (cX, cY)
    else:
        return image, (cX, cY)


if __name__ == '__main__':


    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/2022.07.31/original/*.oct'))

    data = []
    for i in range(len(data_sets)):
        data.append(load_from_oct_file(data_sets[i]))

    p_factor = 0.4

    # for i in range(len(data)):
    volume = data[4]

    # access the frame index of where axial location of the checkerboard
    index = surface_index(volume)[-1][-1]
    pad = 30
    stack = volume[:, :, int(index - pad ):int(index)]

    top_slice = np.amax(stack, axis=2)

    # de-speckling for better feature extraction
    top_slice = despecking(top_slice, sigma=0.5, size=3)
    vmin, vmax = int(p_factor * 255), 255

    top_slice = np.where(top_slice <= vmin,vmin, top_slice)
    # create binary image of the top surface
    bi_img = prep.binarization(top_slice)

    bi_img = binary_dilation(bi_img,square(9, dtype=bool))
    top_slice = despecking(top_slice, sigma=1, size=10)

    # bi_img = -bi_img

    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])

    # bi_img = bi_img.astype('uint8')

    # bi_img = cv.filter2D(src=bi_img, ddepth=-1, kernel=kernel)
    img_list = [top_slice, bi_img]

    title_lst = ['original image','binary image']

    # img_list = [top_slice,bi_img]
    vmin, vmax = int(p_factor * 255), 255
    fig, axs = plt.subplots(1, 2, figsize=(16, 9),constrained_layout = True)

    for n, (ax, image,title) in enumerate(zip(axs.flat, img_list,title_lst)):
        ax.imshow(image, 'gray', vmin=np.min(image), vmax=np.max(image))

        ax.set_title(title)
        ax.set_axis_off()
    plt.show()

    CHECKERBOARD = (3, 3)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    gray = bi_img.astype('uint8')

    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD,
                                             cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK +
                                            cv.CALIB_CB_NORMALIZE_IMAGE)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9),constrained_layout = True)

    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #
        imgpoints.append(corners2)
    #
        # cv.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)
        # plt.imshow(img)
        # plt.imshow(bi_img)
        # cv.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)
        for pts in corners2:
            print(pts)
            ax.plot(pts[:,0],pts[:,1] ,marker = 'o', ms = 10)

    #
        # cv.drawChessboardCorners(bi_img, (5,5), corners2, ret)
    ax.imshow(bi_img)
    plt.show()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (512, 512), 1, (512, 512))

    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx,(512, 512), 5)
    dst = cv.remap(top_slice, mapx, mapy, cv.INTER_LINEAR)
    # dst = cv.undistort(gray, mtx, dist, None, newcameramtx)
