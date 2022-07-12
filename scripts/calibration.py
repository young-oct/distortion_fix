# -*- coding: utf-8 -*-
# @Time    : 2022-07-11 11:14
# @Author  : young wang
# @FileName: calibration.py
# @Software: PyCharm

import glob
import numpy as np
import cv2 as cv
from matplotlib.patches import Circle
from os.path import join
import time
from skimage import exposure

import matplotlib.pyplot as plt

from tools.auxiliary import folder_creator, arrTolist, listtoarr, load_from_oct_file

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

if __name__ == '__main__':

    # data = glob.glob('../data/distorted/*.jpeg')
    # data = glob.glob('../data/distorted/*.oct')
    # data = glob.glob('../data/distorted/*.png')
    # data = glob.glob('../data/2022.07.12_1.25mm(paper)/*.oct')
    # image = glob.glob('../data/2022.07.12_1.25mm(paper)/*.png')
    data = glob.glob('../data/2022.07.12_1mm(3dprint)/trial 2/*.oct')
    image = glob.glob('../data/2022.07.12_1mm(3dprint)/trial 2/*.png')

    # Define the dimensions of checkerboard
    height, width = 4, 4
    checked_board = (height, width)

    gray = cv.imread(image[-1])
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, checked_board, None)
    print(ret)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    # # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv.drawChessboardCorners(gray, checked_board, corners2, ret)

        corners2 = np.squeeze(corners2)
        for corner in corners2:
            coord = (int(corner[0]), int(corner[1]))
            circ = Circle(coord, 10, edgecolor='r', fill=False, linewidth=1)
            ax.add_patch(circ)

    plt.title('screenshot demo', size=20)
    plt.axis('off')
    plt.imshow(gray, 'gray')
    plt.tight_layout()
    plt.show()

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv.filter2D(src=gray, ddepth=-1, kernel=kernel)

    # load .oct files deconvolved and original files
    data_decon = load_from_oct_file(data[0])
    data_ori = load_from_oct_file(data[-1])

    # constructed maximum intensity projections of each oct volume
    #
    ori_mip = np.amax(data_ori, axis=2)
    dec_mip = np.amax(data_decon, axis=2)

    # ori_mip = np.sum(data_ori, axis=2)
    # ori_mip /= np.max(ori_mip) * 255


    #
    # dec_mip = np.sum(data_decon, axis=2)
    # dec_mip /= np.max(dec_mip)* 255


    p_factor = 0.65
    vmin, vmax = int(255 * p_factor), 255

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    ax[0, 0].imshow(ori_mip, 'gray', vmin=vmin, vmax=vmax)
    ax[0, 0].set_axis_off()
    ax[0, 0].set_title('mip of the original volume', size=20)

    # plot histogram to find the cut-off for thresholding
    ax[1, 0].hist(np.ravel(ori_mip), density=True)

    ax[0, 1].imshow(dec_mip, 'gray', vmin=vmin, vmax=vmax)
    ax[0, 1].set_title('mip of the deconvolved volume', size=20)
    ax[0, 1].set_axis_off()

    # plot histogram to find the cut-off for thresholding
    ax[1, 1].hist(np.ravel(dec_mip), density=True)

    plt.show()

    # thresholding images to create binary images
    bn_ori = np.clip(ori_mip, vmin, vmax)
    bn_dec = np.clip(dec_mip, vmin, vmax)

    # converted to the corrected data format for processing
    temp = convert(bn_dec,0, 255,np.uint8)

    # image_inverted = np.array(256 - temp, dtype = np.uint8)



    # plt.imshow(temp)
    # plt.show()


    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    # temp = cv.filter2D(src=bn_dec, ddepth=-2, kernel=kernel)
    #
    # # dst = cv.cornerHarris(np.float32(bn_ori), 2,3,0.04)
    # # dst = cv.dilate(dst,None)
    # # np.float32(bn_ori)[dst > 0.01*np.max(np.float32(bn_ori))] = [0,255]
    #
    ret, corners = cv.findChessboardCorners(temp, checked_board, flags=cv.CALIB_CB_ADAPTIVE_THRESH
                                                + cv.CALIB_CB_EXHAUSTIVE)
    # ret, corners = cv.findChessboardCorners(bn_dec.astype('uint8'), checked_board, None)
    print(ret)

    # gray = np.float32(bn_ori)
    # dst = cv.cornerHarris(gray, 2, 3, 0.04)
    # # result is dilated for marking the corners, not important
    # dst = cv.dilate(dst, None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # gray[dst > 0.01 * dst.max()] = [0, 0, 255]


    #
    #
    # image = load_from_oct_file(data[-1])
    # temp = np.sum(image,axis=2) - np.mean(image, axis=2)
    # img = temp/temp.max()
    # img *= 255
    #
    # from skimage import feature

    from skimage import filters

    # edge_roberts = filters.roberts(img)
    # edge_sobel = filters.sobel(img)
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    # image_sharp = cv.filter2D(src=tmp, ddepth=-1, kernel=kernel)
    #
    # edges2 = feature.canny(im, sigma=12)
    # # tmp = np.clip(edges2,10,60)
    # plt.imshow(edges2,'gray')
    # plt.show()
    #
    # # hlow = 120

    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    # image_sharp = cv.filter2D(src=img, ddepth=-2, kernel=kernel)
    # tmp = np.where(img > hlow, hlow, img)

    # data_mpi = []
    # #
    # for i in range(1):
    #     image = load_from_oct_file(data[i])
    #     img_mpi = np.(image, axis=2).astype('uint8')
    # plt.imshow(img_mpi, 'gray')
    # plt.show()
    # #
    # #     # create the mip image from original image
    #     img_mpi = np.amax(image, axis=2).astype('uint8')
    #     img_mpi = np.amax(image, axis=2)
    #     data_mpi.append(img_mpi)
    # plt.imshow(img_mpi, 'gray', vmin=190, vmax = 255)
    # plt.show()
    # #
    # print('done')
    #
    # # Define the dimensions of checkerboard
    # height,width = 4,4
    # checked_board = (height, width)
    # # termination criteria
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # objp = np.zeros((height * width, 3), np.float32)
    # objp[:, :2] = np.mgrid[0:height, 0:width].T.reshape(-1, 2)
    #
    # square_size = 0.2 # physical square size is 0.2mm
    # objp = objp * square_size
    #
    # # Arrays to store object points and image points from all the images.
    # objpoints = []  # 3d point in real world space
    # imgpoints = []  # 2d points in image plane.
    #
    # # pics = []
    # # low = 215
    # # # for img in data_mpi:
    # # tmp = np.where(data_mpi[0] <= low, 0,255).astype('uint8')
    # #     # pics.append(tmp)
    # # plt.imshow(tmp,'gray', vmin = low, vmax = 255)
    # # plt.show()
    #
    # # termination criteria
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # objp = np.zeros((6 * 7, 3), np.float32)
    # objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    # # Arrays to store object points and image points from all the images.
    # objpoints = []  # 3d point in real world space
    # imgpoints = []  # 2d points in image plane.
    #
    # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # objp = np.zeros((2 * 2, 3), np.float32)
    # objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    # objpoints = []  # 3d point in real world space
    # imgpoints = []  # 2d points in image plane.
    # # #
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # #
    # images = glob.glob('../data/distorted/*.jpeg')
    # for fname in images:
    #     img = cv.imread(fname)
    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #
    #     # temp = img_mpi
    #     # temp = np.where(temp < 190, 0, 255).astype('uint8')

    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    # image_sharp = cv.filter2D(src=tmp, ddepth=-1, kernel=kernel)
    # plt.imshow(image_sharp, 'gray')
    # plt.show()

    # gray = tmp.astype('uint8')
    # ret, corners = cv.findChessboardCorners(gray, (3, 3), None)
    # print(ret)
    # #
    #     corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #
    #     cv.drawChessboardCorners(img, (3,3), corners2, ret)
    #     plt.imshow(img)
    #     plt.show()
    #     # cv.waitKey(500)
    #
