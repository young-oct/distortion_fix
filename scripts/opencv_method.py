# -*- coding: utf-8 -*-
# @Time    : 2022-08-16 14:59
# @Author  : young wang
# @FileName: opencv_method.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
from skimage.morphology import square, \
    closing, dilation, erosion, disk, diamond, opening
from tools.proc import median_filter

if __name__ == '__main__':
    # termination criteria

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    checksize = list(combinations_with_replacement(np.arange(3, 10), 2))

    img = cv.imread('../validation/Artboard 1.png')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    gray = gray
    ax[0].imshow(gray, 'gray')

    c_gray = closing(gray, disk(3))
    c_gray = median_filter(c_gray, 5)

    c_gray = c_gray.astype(img.dtype)
    ax[1].imshow(c_gray, 'gray')
    plt.show()
    tru_chk = []
    for checker in checksize:
        ret, corners = cv.findChessboardCorners(c_gray, checker, None)

        if ret:
            print(checker)
            tru_chk.append(checker)
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            ax.imshow(c_gray, 'gray')
            for pts in corners.squeeze():
                ax.plot(pts[0],pts[1], 'o',ms = 10, color = 'red')
            # ax.set_axis_off()
            plt.show()

    print('done')

    # chckboard = tru_chk[-1]
    # objp = np.zeros((chckboard[0]*chckboard[1],3), np.float32)
    # objp[:, :2] = np.mgrid[0:chckboard[1],
    #               0:chckboard[0]].T.reshape(-1, 2)
    # # Arrays to store object points and image points from all the images.
    # objpoints = []  # 3d point in real world space
    # imgpoints = []  # 2d points in image plane.
    #
    # # Find the chess board corners
    # ret, corners = cv.findChessboardCorners(gray, chckboard, None)
    # # If found, add object points, image points (after refining them)
    # if ret == True:
    #     objpoints.append(objp)
    #     corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #     imgpoints.append(corners)
    #
    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,
    #                                                   gray.shape[::-1], None, None)
    #
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 1,gray.shape[::-1])
    # mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, gray.shape[::-1], 5)
    # dst = cv.remap(c_gray, mapx, mapy, cv.INTER_LINEAR)
    #
    # plt.imshow(dst)
    # plt.show()
