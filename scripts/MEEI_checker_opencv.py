# -*- coding: utf-8 -*-
# @Time    : 2022-08-12 10:46
# @Author  : young wang
# @FileName: MEEI_checker_opencv.py
# @Software: PyCharm


import glob
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure
import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, disk, black_tophat, square,
                                binary_closing,binary_opening,skeletonize)
from natsort import natsorted
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post
from skimage.morphology import square
from skimage import feature
from scipy.ndimage import map_coordinates
from tools.pos_proc import export_map,export_list
import numpy as np
import cv2 as cv
import glob
from itertools import combinations_with_replacement

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 15,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    comb = combinations_with_replacement(np.arange(3, 10), 2)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img = cv.imread('/Users/youngwang/Desktop/distortion_fix/validation/target.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    thresh = 50
    true_board = []
    im_bw = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)[1]
    im_bw = gaussian_filter(im_bw, sigma=0.25)
    im_bw = opening(im_bw, square(5))
    comb_list = list(comb)
    for i in range(len((comb_list))):
        # print(i)
        checkboard = comb_list[i]
        ret, corners = cv.findChessboardCorners(im_bw, checkboard, None)
        if ret:
            print(checkboard)
            true_board.append(checkboard)
    print('done')
    plt.imshow(im_bw,'gray')
    plt.show()

    check_board = true_board[-1]
    objp = np.zeros((check_board[0] * check_board[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:check_board[1],0:check_board[0]].T.reshape(-1,2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(im_bw, check_board, None)
    print(ret)
    # If found, add object points, image points (after refining them)
    fig,ax = plt.subplots(1,1, figsize = (16,9))
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(im_bw,corners, (3,3), (-1,-1), criteria)
        imgpoints.append(corners)

        for pt in corners.squeeze():
            ax.plot(pt[1], pt[0], 'o', markersize=10, color='red')
    ax.imshow(im_bw, 'gray', vmin=np.mean(im_bw)*0.9, vmax=np.max(im_bw))
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

