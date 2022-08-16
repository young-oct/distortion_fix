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

    for checker in checksize:

        ret, corners = cv.findChessboardCorners(c_gray, checker, None)

        if ret:
            print(checker)
    print('done')

