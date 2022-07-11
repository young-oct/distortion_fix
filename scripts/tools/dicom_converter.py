# -*- coding: utf-8 -*-
# @Time    : 2022-03-25 9:22 a.m.
# @Author  : young wang
# @FileName: dicom_converter.py
# @Software: PyCharm

import pydicom

from os.path import join

from os.path import isfile
from pydicom.uid import generate_uid

def oct_to_dicom(data, resolutionx, resolutiony,resolutionz,
                 dicom_folder,**patient_info):
    """
    convert pixel array [512,512,330] to DICOM format
    using MRI template, this template will be deprecated
    in the future once OCT template is received
    """
    dss = []

    template_file = '../template data/template.dcm'
    ds = pydicom.dcmread(template_file)

    # SeriesInstanceUID refers to each series, and should be
    # unqie for each sesssion, and generate_uid() provides an unique
    # identifier
    ds.SeriesInstanceUID = generate_uid()

    all_files_exist = False
    dicom_prefix = 'oct'
    # looping through all 330 slice of images with [512(row) x 512(column)]
    for i in range(data.shape[2]):
        # UID used for indexing slices
        ds.SOPInstanceUID = generate_uid()

        # update row and column numbers to be 512
        ds.Rows = data.shape[0]
        ds.Columns = data.shape[1]

        # define the bottom(assuming the middle plane to be zero,
        # that -165 * 30um(axial resolution) = -4.95 mm)
        # DICOM assumes physical dimension to be in mm unit
        bottom = -4.95
        # elevate the z by its axial resolution at a time
        z = bottom + (i * resolutionz)
        # update meta properties

        # 1cm / 512 = 0.02 mm, needs to check with rob
        # this spacing should be calculated as radiant/pixel then mm to pixel

        ds.PixelSpacing = [resolutionx, resolutiony]  # pixel spacing in x, y planes [mm]
        ds.SliceThickness = resolutionz  # slice thickness in axial(z) direction [mm]
        ds.SpacingBetweenSlices = resolutionz # slice spacing in axial(z) direction [mm]
        ds.SliceLocation = '%0.2f' % z  # slice location in axial(z) direction
        ds.InstanceNumber = '%0d' % (i + 1,)  # instance number, 330 in total
        ds.ImagePositionPatient = [z, 0, 0]  # patient physical location
        ds.Manufacturer = 'Audioptics Medical Inc'
        ds.InstitutionName = 'Audioptics Medical'
        ds.InstitutionAddress = '1344 Summer St., #55, Halifax, NS, Canada'

        ds.PatientName = patient_info['PatientName']
        ds.PatientBirthDate = patient_info['PatientBirthDate']
        ds.PatientSex = patient_info['PatientSex']
        ds.PatientAge = patient_info['PatientAge']
        ds.PatientID = patient_info['PatientID']
        ds.SeriesDescription = patient_info['SeriesDescription']
        ds.StudyDescription = patient_info['StudyDescription']
        ds.StationName = 'Unit 1'
        ds.PhysiciansOfRecord = ''
        ds.PerformingPhysicianName = ''
        ds.InstitutionalDepartmentName = ''
        ds.ManufacturerModelName = 'Mark II'
        ds.PatientAddress = ''

        # setting the dynamic range with WindowCenter and WindowWidth
        # lowest_visible_value = window_center — window_width / 2
        # highest_visible_value = window_center + window_width / 2

        ds.WindowCenter = '215'
        ds.WindowWidth = '225'

        # # set highest and lowest pixel values
        ds.LargestImagePixelValue = 255
        ds.SmallestImagePixelValue = 0
        dicom_file = join(dicom_folder, "%s%04d.dcm" % (dicom_prefix, i))

        pixel_data = data[:, :, i]
        ds.PixelData = pixel_data.tobytes()
        ds.save_as(dicom_file)
        dss.append(ds)
        all_files_exist = all_files_exist and isfile(dicom_file)
    return all_files_exist
