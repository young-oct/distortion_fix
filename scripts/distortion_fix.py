# -*- coding: utf-8 -*-
# @Time    : 2022-07-13 11:08
# @Author  : young wang
# @FileName: distortion_fix.py
# @Software: PyCharm
import glob
import numpy as np
import cv2 as cv
from matplotlib.patches import Circle
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from tools.auxiliary import folder_creator, arrTolist, listtoarr, load_from_oct_file
from tools.dicom_converter import oct_to_dicom
from os.path import join


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

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

def binary_mask(slice, vmin, vmax):
    ret, msk = cv.threshold(slice, vmin, vmax, cv.THRESH_BINARY)
    krn = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    return cv.bitwise_and(cv.dilate(msk, krn, iterations=100), msk)

if __name__ == '__main__':

    data = glob.glob('../data/2022.07.13_1mm(3dprint)/trial 5/*.oct')

    # Define the dimensions of checkerboard
    height, width = 4, 4
    checked_board = (height, width)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)

    # Initialize enpty list to accumulate coordinates
    objpoints = []  # 3d world coordinates
    imgpoints = []  # 2d image coordinates

    objp = np.zeros((1,
                     checked_board[0] * checked_board[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checked_board[0], 0:checked_board[1]].T.reshape(-1, 2)

    fig, ax = plt.subplots(1,len(data), figsize = (16, 9))

    for i in range(len(data)):

        data_ori = load_from_oct_file(data[i])

        index = max_slice(data_ori)

        # constructed maximum intensity projections from a stack with
        # certain thickness
        pad = 10
        mip_slice = mip_stack(data_ori, index, pad)

        p_factor = 0.65
        vmin, vmax = int(255 * p_factor), 255

        mask = binary_mask(mip_slice,vmin, vmax)
        mask = ndimage.median_filter(mask, size=3)
        mask = ndimage.gaussian_filter(mask, sigma=0.1)

        res = np.uint8(mask)
        ret, corners = cv.findChessboardCorners(res, checked_board,
                                                flags=cv.CALIB_CB_ADAPTIVE_THRESH +
                                                      cv.CALIB_CB_FAST_CHECK +
                                                      cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret:

            # If found, add object points, image points (after refining them)
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(res, corners, (13, 13), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv.drawChessboardCorners(res, checked_board, corners2, ret)

            corners2 = np.squeeze(corners2)
            for corner in corners2:
                coord = (int(corner[0]), int(corner[1]))
                circ = Circle(coord, 5, edgecolor='r', fill=True, linewidth=1, facecolor='r')
                ax[i].add_patch(circ)

        ax[i].set_title('trial '+ str(i), size = 20)
        ax[i].set_axis_off()
        ax[i].imshow(res,'gray', vmin= vmin, vmax = vmax )
    plt.tight_layout()
    plt.show()

    h, w = res.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, res.shape[::-1], None, None)

    val_data = load_from_oct_file('../data/validation/2022-Jul-13_02.04.07_PM_Bin_Capture.oct')
    index = max_slice(val_data)

    # constructed maximum intensity projections from a stack with
    # certain thickness
    pad = 10
    val_slice = mip_stack(val_data, index, pad)

    # undistort
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv.remap(val_slice, mapx, mapy, cv.INTER_LINEAR)

    fig,ax = plt.subplots(1,2, figsize = (16,9))
    ax[0].set_title('distorted validation image', size=25)
    ax[0].set_axis_off()
    ax[0].imshow(val_slice, 'gray', vmin=vmin, vmax=vmax)

    ax[1].set_title('undistorted validation image', size=25)
    ax[1].set_axis_off()
    ax[1].imshow(dst, 'gray', vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.show()

    #create dicom stacks for comparison
    dicom_path = join('../', 'validation dicom')
    resolutionx, resolutiony, resolutionz = 0.026, 0.026, 0.030

    folder_creator(dicom_path)

    file_path = 'distorted'
    f_path = join(dicom_path,file_path)
    folder_creator(f_path)

    patient_info = {'PatientName': 'RESEARCH',
                    'PatientBirthDate': '19600507',
                    'PatientSex': 'F',
                    'PatientAge': '63Y',
                    'PatientID': '202207070001',
                    'SeriesDescription': file_path,
                    'StudyDescription': 'OCT 3D'}

    oct_to_dicom(val_data, resolutionx=resolutionx,
                 resolutiony=resolutiony,resolutionz = resolutionz,
                 dicom_folder=f_path,
                 **patient_info)
    #
    print('Done creating dicom stacks for distorted validation dataset')

    val_undis = np.zeros_like(val_data)
    for i in range(val_data.shape[-1]):
        val_undis[:,:,i] = cv.remap(val_data[:,:,i], mapx, mapy, cv.INTER_LINEAR)

    print('Done undistorting validation dataset')

    file_path = 'undistorted'
    f_path = join(dicom_path,file_path)
    folder_creator(f_path)

    patient_info = {'PatientName': 'RESEARCH',
                    'PatientBirthDate': '19600507',
                    'PatientSex': 'F',
                    'PatientAge': '63Y',
                    'PatientID': '202107070001',
                    'SeriesDescription': file_path,
                    'StudyDescription': 'OCT 3D'}

    oct_to_dicom(val_undis, resolutionx=resolutionx,
                 resolutiony=resolutiony,resolutionz = resolutionz,
                 dicom_folder=f_path,
                 **patient_info)
    #
    print('Done creating dicom stacks for undistorted validation dataset')