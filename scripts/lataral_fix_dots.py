# -*- coding: utf-8 -*-
# @Time    : 2022-07-31 21:28
# @Author  : young wang
# @FileName: lataral_fix_dots.py
# @Software: PyCharm



import glob
from matplotlib.patches import Circle

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


#
# def find_dots(image):
#     # target = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     target = image.astype('uint8')
#     detected_circles = cv.HoughCircles(target,
#                                         cv.HOUGH_GRADIENT, 1, 20, param1=50,
#                                         param2=30, minRadius=1, maxRadius=40)
#     cir_loc = []
#     # Draw circles that are detected.
#     if detected_circles is not None:
#
#         # Convert the circle parameters a, b and r to integers.
#         detected_circles = np.uint16(np.around(detected_circles))
#
#         for pt in detected_circles[0, :]:
#             a, b, = pt[0], pt[1]
#             cir_loc.append((a,b))
#
#     return cir_loc

if __name__ == '__main__':


    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    data_sets = natsorted(glob.glob('../data/2022.07.31/enhanced/*.oct'))

    data = []
    for i in range(len(data_sets)):
        data.append(load_from_oct_file(data_sets[i]))

    p_factor = 0.4

    # for i in range(len(data)):
    volume = data[0]

    # access the frame index of where axial location of the checkerboard
    index = surface_index(volume)[-1][-1]
    pad = 30
    stack = volume[:, :, int(index - pad ):int(index)]

    top_slice = np.amax(stack, axis=2)

    # de-speckling for better feature extraction
    top_slice = despecking(top_slice, sigma=1, size=1)
    vmin, vmax = int(p_factor * 255), 255

    top_slice = np.where(top_slice <= vmin,vmin, top_slice)
    # create binary image of the top surface
    bi_img = prep.binarization(top_slice)

    # Calculate the median dot size and distance between them.

    (dot_size, dot_dist) = prep.calc_size_distance(bi_img)
    # Remove non-dot objects
    s_img = prep.select_dots_based_size(bi_img, dot_size)
    s_img = prep.select_dots_based_ratio(s_img, ratio=0.4)

    title_lst = ['original image','binary image', 'segmented image']

    img_list = [top_slice,bi_img,s_img]
    vmin, vmax = int(p_factor * 255), 255
    # fig, axs = plt.subplots(1, 3, figsize=(16, 9), constrained_layout = True)
    fig, axs = plt.subplots(1, 3, figsize=(16, 9),constrained_layout = True)

    for n, (ax, image,title) in enumerate(zip(axs.flat, img_list,title_lst)):
        ax.imshow(image, 'gray', vmin=np.min(image), vmax=np.max(image))

        ax.set_title(title)
        ax.set_axis_off()
    plt.show()

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ########################################Blob Detector##############################################

    from skimage.feature import blob_dog, blob_log, blob_doh
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.feature import canny
    from skimage.draw import circle_perimeter

    edges = canny(s_img, sigma=3, low_threshold=10, high_threshold=50)

    hough_radii = np.arange(20, 35, 2)
    hough_res = hough_circle(edges, hough_radii)
    # keypoints = blob_dog(s_img,max_sigma=30, num_sigma=10, threshold=.1) # Detect blobs.
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=3)
    edges = canny(s_img, sigma=0.1, low_threshold=10, high_threshold=50)

    # for dot in keypoints:
    #     circ = Circle(dot, 5, edgecolor='r', fill=True, linewidth=1, facecolor='r')
    #     plt.add_patch(circ)

    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=s_img.shape)
        s_img[circy, circx] = (220, 20, 20)

    plt.tight_layout()
    plt.show()