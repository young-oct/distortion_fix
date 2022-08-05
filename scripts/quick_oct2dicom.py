# -*- coding: utf-8 -*-
# @Time    : 2022-07-22 11:58
# @Author  : young wang
# @FileName: quick_oct2dicom.py
# @Software: PyCharm

import glob
from natsort import natsorted
from tools.pre_proc import folder_creator,load_from_oct_file
from tools.pos_proc import oct_to_dicom
from os.path import join

if __name__ == '__main__':

    # data_sets = natsorted(glob.glob('../data/comparsion/*.oct'))
    data_sets = natsorted(glob.glob('/Users/youngwang/Desktop/corrected-datasets-v2/*.oct'))

    data = load_from_oct_file(data_sets[2])
    #create dicom stacks for comparison
    dicom_path = join('../', 'data','validation dicom')
    resolutionx, resolutiony, resolutionz = 0.034, 0.034, 0.034

    folder_creator(dicom_path)

    file_path = 'update'
    f_path = join(dicom_path,file_path)
    folder_creator(f_path)

    patient_info = {'PatientName': 'validation',
                    'PatientBirthDate': '19600507',
                    'PatientSex': 'M',
                    'PatientAge': '64Y',
                    'PatientID': '202107070001',
                    'SeriesDescription': file_path,
                    'StudyDescription': 'OCT 3D'}

    oct_to_dicom(data, resolutionx=resolutionx,
                 resolutiony=resolutiony,resolutionz = resolutionz,
                 dicom_folder=f_path,
                 **patient_info)
    #
    print('Done creating dicom stacks for distorted validation dataset')

