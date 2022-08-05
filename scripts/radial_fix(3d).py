# -*- coding: utf-8 -*-
# @Time    : 2022-08-05 15:51
# @Author  : young wang
# @FileName: radial_fix(3d).py
# @Software: PyCharm

import glob
import matplotlib.pyplot as plt
from natsort import natsorted
from tools.pre_proc import folder_creator,load_from_oct_file
from tools.pos_proc import oct_to_dicom
from os.path import join
import discorpy.post.postprocessing as post
import numpy as np

if __name__ == '__main__':

    data_sets = natsorted(glob.glob('../data/2022.08.01/validation/*.oct'))

    xcenter, ycenter = 297.843961688918, 271.40700819595907
    list_fact = [1.03579574e+00, -9.56134997e-04,  7.48450764e-06, -2.45661329e-08,
        1.67663262e-11]
    data = load_from_oct_file(data_sets[0])
    #
    start_index, stop_index = 0, int(data.shape[-1] - 1)
    c_data = post.unwarp_chunk_slices_backward(data, xcenter, ycenter, list_fact,
                                 start_index, stop_index)
    # # plt.imshow(data[:,:,100])
    # plt.show()
    c_data = np.interp(c_data,
                        (c_data.min(),
                        c_data.max()),
                        (0, 255)).astype(data.dtype)
    # #create dicom stacks for comparison
    dicom_path = join('../', 'data','validation dicom')
    resolutionx, resolutiony, resolutionz = 0.034, 0.034, 0.034
    #
    folder_creator(dicom_path)

    file_path = 'radial fix'
    f_path = join(dicom_path,file_path)
    folder_creator(f_path)
    #
    patient_info = {'PatientName': 'validation',
                    'PatientBirthDate': '19600507',
                    'PatientSex': 'M',
                    'PatientAge': '64Y',
                    'PatientID': '202107070001',
                    'SeriesDescription': file_path,
                    'StudyDescription': 'OCT 3D'}

    oct_to_dicom(c_data, resolutionx=resolutionx,
                 resolutiony=resolutiony,resolutionz = resolutionz,
                 dicom_folder=f_path,
                 **patient_info)
    # #
    print('Done creating dicom stacks for distorted validation dataset')
