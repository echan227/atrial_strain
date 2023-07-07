# import cc3d
import imageio
import logging
import os
import random
import shutil
import string
import subprocess
import warnings
from datetime import datetime, timedelta, date
from time import sleep

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from dateutil.parser import parse
from matplotlib.colors import ListedColormap
from nibabel.nifti1 import unit_codes, xform_codes, data_type_codes
from pydicom.dicomio import read_file

def convert_dicom_to_nifti(source_dicom_dirs, out_image_file, sequences_SA, logger):
    X, Y, dx, dy, SpacingBetweenSlices, SliceThickness, z_locs, img_pos, axis_x, axis_y, tt, inst_nb, series_nb, HR, SequenceName= \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    data = []
    SOPInstanceUID = []
    for source_dicom_dir in source_dicom_dirs:
        dicom_data = read_file(source_dicom_dir)
        series_nb.append(int(dicom_data.SeriesNumber))
        inst_nb.append(dicom_data.InstanceNumber)
        X.append(dicom_data.Columns)
        Y.append(dicom_data.Rows)
        dx.append(float(dicom_data.PixelSpacing[0]))
        dy.append(float(dicom_data.PixelSpacing[1]))

        # Determine the z 
        if hasattr(dicom_data, 'SpacingBetweenSlices'):
            SpacingBetweenSlices.append(float(dicom_data.SpacingBetweenSlices))
        else:
            SpacingBetweenSlices.append(1)

        if hasattr(dicom_data, 'SliceThickness'):
            SliceThickness.append(float(dicom_data.SliceThickness))
        else:
            SliceThickness.append(1)
            logger.error('Cannot find attribute SliceThickness.')

        if hasattr(dicom_data, 'SOPInstanceUID'):
            SOPInstanceUID.append(dicom_data.SOPInstanceUID)
        else:
            logger.error('Cannot find attribute SOPInstanceUID.')

        # Image position
        if hasattr(dicom_data, 'SliceLocation'):
            z_locs.append(float(dicom_data.SliceLocation))  # SliceLocation: z-location
        else:
            logger.error('Cannot find attribute SliceLocation.')

        # Heart Rate
        if hasattr(dicom_data, 'HeartRate'):
            HR.append(float(dicom_data.HeartRate))      
        if hasattr(dicom_data, 'NominalInterval'):
            if float(dicom_data.NominalInterval) != 0:
                HR.append(60000.0 / float(dicom_data.NominalInterval))     
            else:
                logger.error('Cannot find attribute HeartRate or NominalInterval')
                HR.append(0)
        else:
            logger.error('Cannot find attribute HeartRate or NominalInterval')

        pos_ul = np.array([float(x) for x in
                           dicom_data.ImagePositionPatient])
        pos_ul[:2] = -pos_ul[:2]
        img_pos.append(pos_ul)

        if hasattr(dicom_data, 'RescaleSlope'):
            rslope = float(dicom_data.RescaleSlope)
        else:
            rslope = 1
        if hasattr(dicom_data, 'RescaleIntercept'):
            rinter = float(dicom_data.RescaleIntercept)
        else:
            rinter = 0

        if hasattr(dicom_data, 'SequenceName'):
            SequenceName = dicom_data.SequenceName

        # Image orientation
        ax = np.array([float(x) for x in dicom_data.ImageOrientationPatient[:3]])
        ay = np.array([float(x) for x in dicom_data.ImageOrientationPatient[3:]])
        ax[:2] = -ax[:2]
        ay[:2] = -ay[:2]
        axis_x.append(ax)
        axis_y.append(ay)
        tt.append(int(dicom_data.TriggerTime))
        try:
            pixelarray = dicom_data.pixel_array * rslope + rinter
        except:
            logger.error('Cannot find a valid pixel_array.')

        data.append(pixelarray)

    ##########################################
    # CREATE NIFTI VOLUME
    ##########################################
    X = np.array(X)
    Y = np.array(Y)
    dx = np.array(dx)
    dy = np.array(dy)
    SliceThickness = np.array(SliceThickness)
    SpacingBetweenSlices = np.array(SpacingBetweenSlices)
    series_nb = np.array(series_nb)
    axis_x = np.array(axis_x)
    axis_y = np.array(axis_y)
    inst_nb = np.array(inst_nb).astype(int)
    data = np.array(data)
    SOPInstanceUID = np.array(SOPInstanceUID)
    img_pos = np.array(img_pos)
    tt = np.array(tt)
    z_locs = np.array(z_locs)
    if len(z_locs) == 0:
        z_locs = img_pos[:, 0]

    if len(np.unique(z_locs)) == 1:
        dz = np.unique(SliceThickness)[0]
    else:
        if len(np.unique(SpacingBetweenSlices)) != 1:
            logger.error('Size of SpacingBetweenSlices changes over dicoms!')
        elif np.unique(SpacingBetweenSlices)[0] == 1:
            dz = int(np.mean(np.diff(np.unique(z_locs))))
        else:
            dz_aux = np.unique(SpacingBetweenSlices)[0]
            dz_aux2 = int(np.mean(np.diff(np.unique(z_locs))))
            if np.diff([dz_aux, dz_aux2])[0] < 0.5:
                dz = dz_aux
            else:
                dz = dz_aux
                # raise Warning('Problem with dz')
    # Generate Tags
    if len(np.unique(X)) != 1:
        logger.error('Size of X changes over dicoms!')
    else:
        X = np.unique(X)[0]

    if len(np.unique(Y)) != 1:
        logger.error('Size of Y changes over dicoms!')
    else:
        Y = np.unique(Y)[0]

    if len(np.unique(dx)) != 1:
        logger.error('Size of dx changes over dicoms!')
    else:
        dx = np.unique(dx)[0]

    if len(np.unique(dy)) != 1:
        logger.error('Size of dy changes over dicoms!')
    else:
        dy = np.unique(dy)[0]
    HR = np.array(HR)
    HR = HR[HR!=0]
    if len(np.unique(HR)) != 1:
        HR = np.mean(np.unique(HR))
    else:
        HR = np.unique(HR)[0]

    axis_x = np.squeeze(np.unique(axis_x, axis=0))
    axis_y = np.squeeze(np.unique(axis_y, axis=0))

    if 'ThroughPlane_Flow_Breath_Hold_P' in sequences_SA:
        np.savetxt(out_image_file[:-7] + '_venc.txt', [np.unique(SequenceName)[0].split('_')[-1]], fmt = '%s')

    if len(axis_x) == 3 and len(axis_y) == 3:
        data = np.transpose(data)
        _, idx = np.unique(z_locs, return_index=True)
        z_locs_unique = np.sort(z_locs[np.sort(idx)])
        Z = len(z_locs_unique)

        # Check if slices repeated in same position
        ind = np.where(np.diff(z_locs_unique) < 1)[0]  # TODO: smaller than 1mm, why?
        if len(ind) > 0:
            for ix in ind:  # There can be more than one slices repeated --> ind=1 or ind>1
                z_locs_repeated = z_locs_unique[np.where(np.abs(z_locs_unique - z_locs_unique[ix]) < 1)[0]]
                z_locs_repeated_idx = []
                for z_loc_repeated in z_locs_repeated:
                    z_locs_repeated_idx.append(np.where(z_locs == z_loc_repeated)[0])
                z_locs_repeated_idx = np.array(z_locs_repeated_idx)
                indx_to_reject = np.where(series_nb[np.concatenate(z_locs_repeated_idx)].min() == series_nb)[0]
                data = np.delete(data, indx_to_reject, axis=2)
                img_pos = np.delete(img_pos, indx_to_reject, axis=0)
                inst_nb = np.delete(inst_nb, indx_to_reject)
                z_locs = np.delete(z_locs, indx_to_reject)
                SOPInstanceUID = np.delete(SOPInstanceUID, indx_to_reject)
                series_nb = np.delete(series_nb, indx_to_reject)
                tt = np.delete(tt, indx_to_reject)
                _, idx = np.unique(z_locs, return_index=True)
                z_locs_unique = z_locs[np.sort(idx)]
                z_locs_unique = np.sort(z_locs_unique)
                Z = len(z_locs_unique)
        img_pos = np.unique(img_pos, axis=0)
        if Z > 1:
            if img_pos.shape[0] == Z:
                z_diff = np.diff(img_pos, axis=0)
                axis_z = np.mean(z_diff, axis=0)
                if (abs(z_diff - axis_z) > 10 * np.std(z_diff, axis=0)).any():
                    logger.error('z-spacing between slices varies by more than 10 standard deviations!')
                axis_z = axis_z / np.linalg.norm(axis_z)  # divide by the norm to normalise 3-D vector
            else:
                logger.error('z-locations do not correspond with slice locations!')
        else:
            axis_z = np.cross(axis_x, axis_y)

        if np.max(np.unique(inst_nb)) == len(source_dicom_dirs):
            T = data.shape[2] / Z
            if T.is_integer():
                T = int(T)
                repeated_slices = False
            else:
                T = int(T)
                repeated_slices = True
        else:
            T = int(np.unique(inst_nb).shape[0])
            if T * Z == data.shape[2]:
                repeated_slices = False
            else:
                repeated_slices = True
                
        if repeated_slices:
            ind = []
            for z_loc_idx, z_loc in enumerate(z_locs_unique):
                a = np.where(z_locs == z_loc)[0]
                unique_series_nb = np.unique(series_nb[a])
                if len(unique_series_nb) > 1:
                    for ii in unique_series_nb:
                        if ii != np.max(unique_series_nb):
                            ind.append(np.where([series_nb == ii])[1])
                elif len(tt[a]) > T:
                    aux = inst_nb[a]
                    a = a[aux.argsort()]
                    ind.append(a[:len(aux)//2])
            ind = np.concatenate(ind)
            tt = np.delete(tt, ind)
            inst_nb = np.delete(inst_nb, ind)
            data = np.delete(data, ind, axis=2)
            SOPInstanceUID = np.delete(SOPInstanceUID, ind)
            z_locs = np.delete(z_locs, ind)
            T = data.shape[2] / Z
            if T.is_integer():
                T = int(T)
                repeated_slices = False
            else:
                logger.error('Repeated slices not possible to solve')

        # Affine matrix which converts the voxel coordinate to world coordinate NOTE: currently sform affine
        affine = np.eye(4)
        affine[:3, 0] = axis_y
        affine[:3, 1] = axis_x
        affine[:3, 2] = -1 * axis_z
        affine[:3, 3] = img_pos[-1, :]
        volume = np.zeros((Y, X, Z, T), dtype='float32')
        dcm_files = np.zeros((Z, T), dtype=object)

        trigger_time_final = np.zeros((Z, T), dtype=object)
        if T != 1 or Z != 1:
            # Divide data by slices
            for z_loc_idx, z_loc in enumerate(z_locs_unique):
                a = np.where(z_locs == z_loc)[0]
                if len(inst_nb[a]) != T:
                    logger.error('Repeated slices not possible to solve')
                trigTime = tt[a]
                inNb = inst_nb[a]
                if (trigTime.argsort() == inNb.argsort()).all():
                    auxData = np.squeeze(data[:, :, a])
                    auxData = auxData[:, :, trigTime.argsort()]
                    volume[:, :, Z - z_loc_idx - 1, :] = np.transpose(auxData, (1, 0, 2))
                    auxData = np.squeeze(SOPInstanceUID[a])
                    auxData = auxData[trigTime.argsort()]
                    dcm_files[Z - z_loc_idx - 1, :] = auxData
                    trigger_time_final[Z - z_loc_idx - 1, :] = np.sort(trigTime)
                elif len(trigTime) / len(np.unique(trigTime)) == 3:
                    print('This is a flow sequence: {}'.format(sequences_SA))
                    T = len(np.unique(trigTime))
                    Z = 3
                    inNb_1 = inNb.argsort()[0:T]
                    tt_1 = trigTime[inNb.argsort()[0:T]]
                    inNb_2 = inNb.argsort()[T:2 * T]
                    tt_2 = trigTime[inNb.argsort()[T:2 * T]]
                    inNb_3 = inNb.argsort()[2 * T:]
                    tt_3 = trigTime[inNb.argsort()[2 * T:]]
                    if (tt_3 == tt_2).all() and (tt_3 == tt_1).all():
                        volume = np.zeros((Y, X, Z, T), dtype='float32')
                        dcm_files = np.zeros((Z, T), dtype=object)

                        # Flow 1
                        auxData = np.squeeze(data[:, :, inNb_1])
                        auxData = auxData[:, :, tt_1.argsort()]
                        volume[:, :, 0, :] = np.transpose(auxData, (1, 0, 2))
                        auxData = np.squeeze(SOPInstanceUID[inNb_1])
                        auxData = auxData[tt_1.argsort()]
                        dcm_files[0, :] = auxData

                        # Flow 2
                        auxData = np.squeeze(data[:, :, inNb_2])
                        auxData = auxData[:, :, tt_2.argsort()]
                        volume[:, :, 1, :] = np.transpose(auxData, (1, 0, 2))
                        auxData = np.squeeze(SOPInstanceUID[inNb_2])
                        auxData = auxData[tt_2.argsort()]
                        dcm_files[1, :] = auxData

                        # Flow 3
                        auxData = np.squeeze(data[:, :, inNb_3])
                        auxData = auxData[:, :, tt_3.argsort()]
                        volume[:, :, 2, :] = np.transpose(auxData, (1, 0, 2))
                        auxData = np.squeeze(SOPInstanceUID[inNb_3])
                        auxData = auxData[tt_3.argsort()]
                        dcm_files[2, :] = auxData

                        dt = np.unique(np.diff(tt))
                        dt = dt[dt > 0]
                        dt = int(round(np.mean(dt)))
                        toffset = 0.0

                        for indx in range(3):
                            imgobj = nib.nifti1.Nifti1Image(np.expand_dims(volume[:, :, indx, :], 2),
                                                            affine)  # Create images from numpy arrays

                            # header info: http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
                            hdr = {
                                'pixdim': np.array([1.0, dx, dy, dz, dt, 1.0, 1.0, 1.0], np.float32),
                                'toffset': toffset,
                                'slice_duration': dt,
                                'xyzt_units': unit_codes['mm'] | unit_codes['msec'],
                                'qform_code': xform_codes['aligned'],
                                'sform_code': xform_codes['scanner'],
                                'datatype': data_type_codes.code[np.float32]
                            }
                            for k, v in hdr.items():
                                if k in imgobj.header:
                                    imgobj.header[k] = v

                            imgobj.to_filename('{}_{}.nii.gz'.format(out_image_file.strip('.nii.gz'), indx))

                        return [Y, X, Z, T, dcm_files, volume, affine, hdr, repeated_slices]

                elif (trigTime.argsort() == inNb.argsort()[::-1]).all():
                    auxData = np.squeeze(data[:, :, a])
                    auxData = auxData[:, :, trigTime.argsort()]
                    volume[:, :, Z - z_loc_idx - 1, :] = np.transpose(auxData, (1, 0, 2))
                    auxData = np.squeeze(SOPInstanceUID[a])
                    auxData = auxData[trigTime.argsort()]
                    dcm_files[Z - z_loc_idx - 1, :] = auxData
                    trigger_time_final[Z - z_loc_idx - 1, :] = np.sort(trigTime)

                else:
                    logger.error('Mismatch between trigger time and instance number.')

        else:
            logger.error('T = {} and Z = {} - Not a SAX sequence'.format(T, Z))

        trigger_time_final = np.mean(trigger_time_final,axis=0).astype(int)
        np.savetxt(out_image_file[:-7] + '_tt.txt', trigger_time_final)

        if len(np.unique(tt)) > 1:
            dt = np.unique(np.diff(trigger_time_final, axis=0))
            dt = dt[dt > 0]
            dt = int(round(np.mean(dt)))
            toffset = 0.0
        elif len(np.unique(tt)) == 1 or T == 1:
            # toffset = np.unique(tt)[0]
            # dt = 0
            logger.error('Only one trigger time found.')

        if np.abs(HR-60000.0 / (dt * T)) > 5:
            logger.error('Heart rate from dicom and dt have a difference bigger than 5bmp')
            np.savetxt(out_image_file[:-7] + '_HR.txt', [HR])
        else:
            np.savetxt(out_image_file[:-7] + '_HR.txt', [HR])

        imgobj = nib.nifti1.Nifti1Image(volume, affine)

        # header info: http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
        hdr = {
            'pixdim': np.array([1.0, dx, dy, dz, dt, 1.0, 1.0, 1.0], np.float32),
            # NOTE: The srow_* vectors are in the NIFTI_1 header.  Note that no use is made of pixdim[] in this method.
            'toffset': toffset,
            'slice_duration': dt,
            'xyzt_units': unit_codes['mm'] | unit_codes['msec'],
            'qform_code': xform_codes['aligned'],
            # Code-Labels for qform codes https://nipy.org/nibabel/nifti_images.html
            'sform_code': xform_codes['scanner'],
            # Code-Labels for sform codes https://nipy.org/nibabel/nifti_images.html
            'datatype': data_type_codes.code[np.float32]
        }
        for k, v in hdr.items():
            if k in imgobj.header:
                imgobj.header[k] = v

        imgobj.to_filename(out_image_file)

    else:
        logger.error('Problem converting dicoms: wrong size for axis_x and axis_y')

    return [Y, X, Z, T, dcm_files, volume, affine, hdr, repeated_slices]


# User-defined exceptions
class DirNotEmpty(Exception):
    """Raised when folder is not empty"""

    def __init__(self, non_empty_dir):
        self.message = 'This folder is not empty: {}'.format(non_empty_dir)
        super().__init__(self.message)


class CsvNotUpdated(Exception):
    """Raised when csv does not contain updated information"""
    pass


def convert_dates(study_dates_raw):
    study_dates_new = []
    for date_obj in study_dates_raw:
        if date_obj == '-1':
            pass
        elif not (isinstance(date_obj, datetime)):  # if it doesn't have datetime format...
            date_obj = parse(str(date_obj), dayfirst=True)  # date_obj must have datetime format
            date_obj = date_obj.strftime('%Y%m%d')  # convert to YYYYMMDD format
        study_dates_new.append(date_obj)

    return study_dates_new


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str


def check_csv_file(csv_file):
    error_bool = False
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)  # read csv_file
            if df.empty:
                print("\nFile {} only contains headers. Removing.\n".format(os.path.basename(csv_file)))
                os.remove(csv_file)
                error_bool = True
        except pd.errors.EmptyDataError:
            print("\nFile {} is empty. Removing.\n".format(os.path.basename(csv_file)))
            os.remove(csv_file)
            error_bool = True
    else:
        print("\nFile {} does not exist.\n".format(os.path.basename(csv_file)))
        error_bool = True

    return error_bool


def discard_study_found(study_found_csv_file, patient_ID_list, study_date_list, ethics_list, study_dates_bool):
    patient_IDs = patient_ID_list
    study_dates = study_date_list
    ethics = ethics_list
    patient_IDs_csv = []
    study_dates_csv = []
    study_instance_UIDs_csv = []
    study_dates_found_csv = []
    study_description_csv = []
    ethics_csv = []

    return_bool = check_csv_file(study_found_csv_file)

    if return_bool:
        return [patient_IDs, study_dates, ethics, patient_IDs_csv, study_dates_csv, study_instance_UIDs_csv,
                study_dates_found_csv, study_description_csv, ethics_csv]
    else:
        df = pd.read_csv(study_found_csv_file)

    # Get PatientID and StudyDate from patient_info_csv_file.csv
    patient_ID_list = np.array(patient_ID_list)
    study_date_list = np.array(study_date_list)
    # Get PatientID and StudyDate from study_found.csv
    patient_IDs_csv = df['PatientID'].astype(str).to_numpy()
    study_dates_csv = df['StudyDate'].astype(str).to_numpy()

    loc_remove = []

    # For every patient in the list check if it was already found
    for loc, patient_ID in enumerate(patient_ID_list):
        patient_remove = np.where(patient_ID == patient_IDs_csv)  # Get location(s) of PatientID within patient_IDs_csv
        if patient_remove[0].size > 0:
            # Check if dates match
            for idx in patient_remove[0].tolist():
                if not study_dates_bool:
                    study_date_list[loc] = '-1'

                if study_date_list[loc] == study_dates_csv[idx]:
                    loc_remove.append(loc)  # If they do, save index to later remove from patient info list
                    continue

    # Remove PatientIDs and StudyDates which were already found
    patient_IDs = np.delete(patient_ID_list, loc_remove).tolist()
    study_dates = np.delete(study_date_list, loc_remove).tolist()
    ethics = np.delete(ethics, loc_remove).tolist()
    # Load info from study_found.csv
    patient_IDs_csv = df['PatientID'].tolist()
    study_dates_csv = df['StudyDate'].tolist()
    study_instance_UIDs_csv = df['StudyInstanceUID'].tolist()
    study_dates_found_csv = df['StudyDate_found'].tolist()
    study_description_csv = df['StudyDescription'].tolist()
    ethics_csv = df['Ethics'].tolist()

    return [patient_IDs, study_dates, ethics, patient_IDs_csv, study_dates_csv, study_instance_UIDs_csv,
            study_dates_found_csv, study_description_csv, ethics_csv]


def get_date_range(date_earliest, date_latest):
    if date_earliest > date_latest:
        raise Exception('The first date must be older or equal to the second date')
    study_dates_range = np.arange(date_latest, date_earliest - timedelta(days=1), timedelta(days=-1)).astype(datetime)
    study_dates_range = convert_dates(study_dates_range)

    return study_dates_range


def find_duplicates_csv(csv_file):
    df = pd.read_csv(csv_file)  # read csv
    patient_IDs_csv = df['PatientID'].astype(str).to_numpy()
    study_dates_csv = df['StudyDate_found'].astype(str).to_numpy()
    study_instance_UIDs_csv = df['StudyInstanceUID'].astype(str).to_numpy()

    patient_IDs_dates_csv = patient_IDs_csv + '/' + study_dates_csv

    loc_duplicates = []
    loc_remove = []

    for loc, (patient_ID_date, UID) in enumerate(zip(patient_IDs_dates_csv, study_instance_UIDs_csv)):
        if UID == '-1':
            loc_remove.append(loc)
            continue

        # Find matches between current UID and UID list
        loc_UID = np.where(UID == study_instance_UIDs_csv)[
            0]  # This will return reference loc and found locs of matching UIDs

        # Find matches between current patient_ID/study_date list
        loc_UID = np.unique(np.append(loc_UID, np.where(patient_ID_date == patient_IDs_dates_csv)[
            0]))  # This will return reference loc and found locs of matching patientIDs/dates

        if loc_UID.size > 1:
            # Save loc_duplicates in separate arrays
            loc_duplicates.append(loc_UID)
            # Save loc_remove in a list
            loc_remove.extend(loc_UID[1:].tolist())

    loc_duplicates_flat = [item for sublist in loc_duplicates for item in sublist]  # Flatten loc_duplicates
    df_duplicates = df.loc[loc_duplicates_flat, :]
    df_no_duplicates = df.drop(loc_remove)

    return loc_duplicates, df_duplicates, df_no_duplicates


def find_patient_date_folders(path_DICOM_formatted):
    patient_folders = get_list_of_dirs(path_DICOM_formatted, full_path=False)

    patient_date_folders = []
    for patient_folder in patient_folders:
        patient_folder_full_path = os.path.join(path_DICOM_formatted, patient_folder)
        date_folders = get_list_of_dirs(patient_folder_full_path, full_path=False)

        for date_folder in date_folders:
            patient_date_folder = os.path.join(patient_folder, date_folder)
            patient_date_folders = np.append(patient_date_folders, patient_date_folder)

    return patient_date_folders


def find_study_downloaded(study_instance_UIDs_dirs, DICOM_formatted_ws_dir, DICOM_formatted_no_ws_dir, patient_IDs_csv,
                          study_dates_csv, study_instance_UIDs_csv):
    # Check 1: find duplicate cases in Workspace/ and No_Workspace.
    patient_date_folders_ws = find_patient_date_folders(DICOM_formatted_ws_dir)
    patient_date_folders_no_ws = find_patient_date_folders(DICOM_formatted_no_ws_dir)
    patient_date_folders_both = list(
        set(patient_date_folders_ws).intersection(patient_date_folders_no_ws))  # cases which appear in both folders
    if len(patient_date_folders_both) > 0:
        raise Exception('At least one case appears both in Workspace/ and No_Workspace/. Please check.')

        # Check which found cases were already downloaded
    patient_date_folders = np.append(patient_date_folders_ws, patient_date_folders_no_ws)
    loc_download = []
    for loc, (patient_ID, study_date, study_instance_UID) in enumerate(
            zip(patient_IDs_csv, study_dates_csv, study_instance_UIDs_csv)):
        # Check if already downloaded in DICOM_dir
        if study_instance_UIDs_dirs.size == 0:
            pass
        elif (study_instance_UID == study_instance_UIDs_dirs).any():
            loc_download.append(loc)
            continue

        # Check if already downloaded in DICOM_formatted_ws_dir or DICOM_formatted_no_ws_dir
        if patient_date_folders.size == 0:
            pass
        elif ((patient_ID + os.path.sep + study_date) == patient_date_folders).any():
            loc_download.append(loc)
            continue

    return loc_download


def subfolders_count(_dir):
    one_folder_dirs = 0

    dirs = [d for d in os.listdir(_dir) if os.path.isdir(_dir + '/' + d)]  # Get all subfolders within dir
    for d in dirs:
        subdirs = [subd for subd in os.listdir(_dir + '/' + d) if os.path.isdir(_dir + '/' + d + '/' + subd)]
        if len(subdirs) == 0:
            warnings.warn('No folders within: ' + d)
        elif len(subdirs) > 1:
            warnings.warn('Multiple folders: ' + d + '/' + str(subdirs))
        else:
            one_folder_dirs += 1

    print("{}/{} folders contain exactly 1 subfolder".format(one_folder_dirs, len(dirs)))


def remove_small_subfolders(_dir, folder_size_min):
    folder_sizes = []

    # Remove small folders directly under dir (level 1)
    subfolder_dirs = get_list_of_dirs(_dir)  # get all subfolder names ignoring hidden subfolders
    for subfolder_dir in subfolder_dirs:
        folder_size = subprocess.check_output(['du', '-shm', subfolder_dir]).split()[0].decode(
            'utf-8')  # Size for each subfolder given in megabites
        folder_sizes.append(int(folder_size))

        if int(folder_size) <= folder_size_min:
            print("Removing {} ({}MB)".format(os.path.basename(subfolder_dir), folder_size))
            shutil.rmtree(subfolder_dir)
            continue

        # Remove small subfolders under subfolder_dirs (level 2)
        subsubfolder_dirs = get_list_of_dirs(subfolder_dir)  # get all subfolder names ignoring hidden subfolders
        for subsubfolder_dir in subsubfolder_dirs:
            folder_size = subprocess.check_output(['du', '-shm', subsubfolder_dir]).split()[0].decode(
                'utf-8')  # size for each subfolder given in megabites
            if int(folder_size) <= folder_size_min:
                print("Removing {} ({}MB)".format(os.path.basename(subsubfolder_dir), folder_size))
                shutil.rmtree(subsubfolder_dir)

    if folder_sizes:
        min_size = min(folder_sizes)
        max_size = max(folder_sizes)
    else:
        min_size, max_size = [-1, -1]

    return [min_size, max_size]


def subfolders_remove_empty(_dir):
    for root, dirs, _ in os.walk(_dir):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                # print('Skipping', os.path.join(root, name))
                pass
            else:
                print('Removing', os.path.join(root, name))


def move_folders(path_folder_parent, path_folder_target):
    # Find all folder names within given folder, ignoring hidden folders and all files
    folders = sorted([fname for fname in os.listdir(path_folder_parent) if
                      (not fname.startswith('.') and os.path.isdir(os.path.join(path_folder_parent, fname)))])

    for folder in folders:
        # Move to target folder
        os.system('mv {} {}'.format(os.path.join(path_folder_parent, folder), path_folder_target))


def compare_two_folders(folder1, folder2):
    folder1_contents = sorted([fname for fname in os.listdir(folder1) if
                               (not fname.startswith('.') and os.path.isdir(os.path.join(folder1, fname)))])
    folder2_contents = sorted([fname for fname in os.listdir(folder2) if
                               (not fname.startswith('.') and os.path.isdir(os.path.join(folder2, fname)))])

    print('Number of subfolders in {}: {}\n'.format(folder1, len(folder1_contents)))
    print('Number of subfolders in {}: {}\n'.format(folder2, len(folder2_contents)))

    folder1_missing_folders = list(set(folder2_contents).difference(folder1_contents))
    folder2_missing_folders = list(set(folder1_contents).difference(folder2_contents))

    print('Subfolders present in {} but missing in {}: {}\n'.format(folder2, folder1, folder1_missing_folders))
    print('Subfolders present in {} but missing in {}: {}\n'.format(folder1, folder2, folder2_missing_folders))


def find_workspace_downloaded(str_old, str_ext, str_datetime_len):
    if str_old[-len(str_ext):] != str_ext:
        print('{} is not a {} file\n'.format(str_old, str_ext))
        return -1

    for count_str, char in enumerate(str_old):
        if char.isalpha():  # letters
            continue

        elif char.isnumeric():  # numbers
            count_str += str_datetime_len
            str_new = str_old[count_str:-len(str_ext)]
            return str_new

        else:
            if ['\'', ',', '-'].count(char) != 1:  # invalid str_old if char is different from: [' , -]
                print('Found unexpected non-alphanumeric character before DateTime for: {}\n'.format(str_old))
                return -1
            continue


def store_anonymised_metadata(tags_to_store, anon_index_str, source_dir):
    metadata = []  # dicom data to be stored in csv

    series_dirs = get_list_of_dirs(source_dir)
    for idx, sn_dir in enumerate(series_dirs):
        dicom_files = get_list_of_files(sn_dir, ext_str='.dcm')
        for dicom_file in dicom_files:
            # Read DICOM file
            dicom_data = read_file(os.path.join(sn_dir, dicom_file))

            metadata_tmp = ['A_S' + anon_index_str]
            for tag in tags_to_store.keys():
                if tag in dicom_data:
                    value = str(dicom_data[tag].value)
                    if value:
                        metadata_tmp.append(value)
                    else:
                        metadata_tmp.append(-1)
                else:
                    metadata_tmp.append(-1)
            metadata.append(metadata_tmp)
    return metadata


def anonymise_UID_date(UID, study_date):
    UID_split = UID.split('.')
    UID_anon = []
    for UID_date in UID_split:
        if study_date in UID_date:
            UID_date = UID_date.replace(study_date, str(int(study_date) - 1000))  # subtract one year from UID_date
        UID_anon.append(UID_date)
    UID_anon = '.'.join(UID_anon)
    return UID_anon


def get_replace_tag(tag, value, replace_tag, study_date):
    if tag in [(0x0002, 0x0003), (0x0008, 0x0018), (0x0008, 0x1155), (
            0x0020,
            0x000e)]:  # MediaStorageSOPInstanceUID, SOPInstanceUID,  ReferencedSOPInstanceUID, SeriesInstanceUID
        replace_tag = anonymise_UID_date(value, study_date)
    return replace_tag


def anonymise_elements(dicom_data, TAGS_DICOM_TO_ANONYMIZE, study_date):
    for tag, replace_tag in TAGS_DICOM_TO_ANONYMIZE.items():
        if tag in dicom_data:
            if dicom_data[tag].VR != 'SQ':  # anonymise current element
                dicom_data[tag].value = get_replace_tag(tag, dicom_data[tag].value, replace_tag, study_date)
            else:  # anonymise sequence of elements
                for i in range(len(dicom_data[tag].value)):
                    seq_temp = anonymise_elements(dicom_data[tag][i], TAGS_DICOM_TO_ANONYMIZE, study_date)
                    for seq_tag in dicom_data[tag][i].keys():
                        dicom_data[tag][i][seq_tag] = seq_temp[seq_tag]
    return dicom_data


def perform_anonymisation(source_dir, target_folder, anon_index_str, TAGS_DICOM_TO_ANONYMIZE, study_date):
    TAGS_DICOM_TO_ANONYMIZE[(0x0010, 0x0020)] = anon_index_str  # Patient ID
    TAGS_DICOM_TO_ANONYMIZE[(0x0010, 0x21c0)] = 4  # Pregnancy Status, 4: unknown

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    non_dcm_counter = 0  # counter for non-DICOM files
    series_names = get_list_of_dirs(source_dir, full_path=False)
    for sn in series_names:
        sn_dir = os.path.join(source_dir, sn)
        if study_date in sn:
            sn = sn.replace(study_date, str(int(study_date) - 1000))  # subtract one year
        sn_dir_anon = os.path.join(target_folder, sn)
        if not os.path.exists(sn_dir_anon):
            os.mkdir(sn_dir_anon)

        dicom_files = get_list_of_files(sn_dir, full_path=False)
        for dicom_file in dicom_files:
            if dicom_file.endswith('.dcm'):
                dicom_data = read_file(os.path.join(sn_dir, dicom_file))

                # Anonymise elements in header
                for tag, replace_tag in TAGS_DICOM_TO_ANONYMIZE.items():
                    if tag in dicom_data.file_meta:
                        dicom_data.file_meta[tag].value = get_replace_tag(tag, dicom_data.file_meta[tag].value,
                                                                          replace_tag, study_date)

                # Anonymise elements in dataset
                dicom_data = anonymise_elements(dicom_data, TAGS_DICOM_TO_ANONYMIZE, study_date)

                # Copy anonymised DICOM to sn_dir_anon/
                dname_ano = '{}.dcm'.format(dicom_data.SOPInstanceUID)
                with open(os.path.join(sn_dir_anon, dname_ano), 'wb') as outfile:
                    dicom_data.save_as(outfile)

            else:
                # Anonymise file name and copy to sn_dir_anon/
                source_file = os.path.join(sn_dir, dicom_file)
                file_name_w_ext = os.path.basename(source_file)
                _, file_extension = os.path.splitext(file_name_w_ext)
                target_folder_non_dcm = sn_dir_anon
                target_file = os.path.join(target_folder_non_dcm, 'anonymised_' + str(non_dcm_counter) + file_extension)
                _ = shutil.copy(source_file, target_file)
                non_dcm_counter += 1


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger(os.path.basename(log_path))
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    sleep(1)  # ensure that filenames based on DateTime are not identical

    return logger


class custom_color_map:
    def __init__(self):
        self.newcmap = None
        no_color = np.array([1, 1, 1, 0])
        red = np.array([220 / 256, 20 / 256, 60 / 256, 1])
        cyan = np.array([1 / 256, 256 / 256, 256 / 256, 1])
        blue = np.array([1 / 256, 1 / 256, 256 / 256, 1])
        self.newcolors = np.vstack([no_color, red, cyan, blue])

    def choose_colours(self, labels):
        self.newcolors = self.newcolors[labels]
        self.newcmap = ListedColormap(self.newcolors)
        return self.newcmap


def plot_for_offset(image, segmentation, sl, fr):
    cmap = custom_color_map().choose_colours(np.unique(segmentation).astype(int).tolist())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(image, cmap='gray', interpolation='none')
    # ax.imshow(segmentation, cmap='inferno', interpolation='none', alpha=0.4, origin='lower')
    ax.imshow(segmentation, cmap=cmap, interpolation='none', alpha=0.4, origin='lower')
    ax.set(title='Slice {}, frame {}'.format(sl, fr))

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return image


def save_segmentation_gif(img_filename, seg_filename, save_filename, variation_axis, variation_location, fps=2,
                          quality_control=False):
    """Save a gif file containing all images + segmentations along the z or t axis.
    This will allow us to visualise e.g. misaligned slices/frames.
    Args:
        img_filename: (string) 4-D img name
        seg_filename: (string) corresponding 4-D segmentation name
        save_filename: (string) absolute path of save file
        variation_axis: (string) axis along which we move (z or t)
        variation_location: (int) fixed location in the opposite axis
        fps: (integer, default=2) frames per second
        quality_control: do QC
    """
    img_data = nib.load(img_filename)
    seg_data = nib.load(seg_filename)
    img = img_data.get_fdata()
    seg = seg_data.get_fdata()

    if quality_control:
        if variation_axis == 't':
            new_img = []
            new_seg = []
            frame_idx = []
            for frame in range(seg.shape[3]):
                seg_temp = seg[:, :, variation_location, frame]
                img_temp = img[:, :, variation_location, frame]
                if np.sum(seg_temp) == 0:
                    frame_idx.append(frame)
                    new_seg.append(seg_temp)
                    new_img.append(img_temp)
                else:  # frames with segmentations are shown 4x longer
                    frame_idx.extend([frame, frame, frame, frame])
                    new_seg.extend([seg_temp, seg_temp, seg_temp, seg_temp])
                    new_img.extend([img_temp, img_temp, img_temp, img_temp])

            save_filename = os.path.join('{}_t_var_sl_{}.gif'.format(save_filename, variation_location))
            imageio.mimsave(save_filename,
                            [plot_for_offset(new_img[i], new_seg[i], variation_location, frame_idx[i]) for i in
                             range(len(new_img))], fps=fps)

    else:
        if variation_axis == 't':
            save_filename = os.path.join('{}_t_var_sl_{}.gif'.format(save_filename, variation_location))
            imageio.mimsave(save_filename, [
                plot_for_offset(img[:, :, variation_location, i], seg[:, :, variation_location, i], variation_location,
                                i) for i in range(img.shape[3])], fps=fps)
        elif variation_axis == 'z':
            save_filename = os.path.join('{}_z_var_fr_{}.gif'.format(save_filename, variation_location))
            if len(img.shape) == 3:
                imageio.mimsave(save_filename,
                                [plot_for_offset(img[:, :, i], seg[:, :, i], i, variation_location) for i in
                                 range(img.shape[2])], fps=fps)
            elif len(img.shape) == 4:
                imageio.mimsave(save_filename, [
                    plot_for_offset(img[:, :, i, variation_location], seg[:, :, i, variation_location], i,
                                    variation_location) for i in range(img.shape[2])], fps=fps)


def save_segmentation_png(image_filename, seg_filename, save_filename):
    """Save a png file containing a single image + segmentation.
    This will be useful for quick visualisation during debugging.
    Args:
        image_filename: (string) image name
        seg_filename: (string) corresponding segmentation name
        save_filename: (string) absolute path of save file
    """
    image_data = nib.load(image_filename)
    seg_data = nib.load(seg_filename)
    image = image_data.get_fdata()
    seg = seg_data.get_fdata()

    if len(image.shape) > 2:
        print('WARNING: image shape > 2.')
        if len(image.shape) == 3:
            if image.shape[0] == 1:
                print('WARNING: reshaping image by ignoring first dimension.')
                image = image[0, :, :]
                seg = seg[0, :, :]
            elif image.shape[2] == 1:
                print('WARNING: reshaping image by ignoring last dimension.')
                image = image[:, :, 0]
                seg = seg[:, :, 0]
        else:
            print('WARNING: could not reshape image. Return.')
            return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(image, cmap='gray', interpolation='none')
    ax.imshow(seg, cmap='inferno', interpolation='none', alpha=0.33, origin='lower')
    plt.savefig(save_filename + '.png')


def dice_sim_coef(prediction, reference):
    """Dice similarity coefficient
    """
    if not (prediction.shape == reference.shape):
        raise Exception("Shape mismatch between prediction and ground truth.")

    prediction = np.atleast_1d(prediction.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if np.count_nonzero(reference) == 0:
        dc = -1.0
        return dc
        # raise Exception("Missing ground truth label.")

    tp = np.count_nonzero(prediction & reference)
    fp = np.count_nonzero(prediction & ~reference)
    fn = np.count_nonzero(~prediction & reference)

    try:
        dc = 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        # dc = -1.0
        raise Exception("Empty prediction and ground truth.")

    return dc


def detect_misaligned_segmentations(seg_filename, save_file, label_idx, similarity_threshold):
    # 1. Load segmentation
    seg_data = nib.load(seg_filename)
    seg = seg_data.get_fdata()

    # 2. Calculate dice along t
    Z = seg.shape[2]
    T = seg.shape[3]
    dice_along_t = np.full((Z, T - 1), -1.0)
    for sl in range(seg.shape[2]):
        for fr in range(seg.shape[3] - 1):
            seg_1 = seg[:, :, sl, fr] == label_idx
            seg_2 = seg[:, :, sl, fr + 1] == label_idx
            if np.sum(seg_1) == 0.0 or np.sum(seg_2) == 0.0:
                # print('WARNING: empty segmentation. Continue.')
                continue
            else:
                dice_along_t[sl, fr] = dice_sim_coef(seg_1, seg_2)
        if (dice_along_t[sl, :] < similarity_threshold).any() and (dice_along_t[sl, :] != -1.0).any():
            print('Potential issue for slice {}. What do we do?'.format(sl))

    # np.savetxt("dice_along_t.csv", dice_along_t, delimiter=",", fmt='%0.2f')
    fig, ax = plt.subplots(figsize=(10, 3))
    img_tmp = ax.imshow(dice_along_t, cmap='RdBu', vmin=-1, vmax=1, interpolation='none', origin='lower')
    plt.yticks(range(0, Z, 2))
    plt.xticks(range(0, T - 1, 2))
    cbar = fig.colorbar(img_tmp, ax=ax, extend='both')
    cbar.minorticks_on()
    ax.set(title='Study: {}, dice along t'.format(seg_filename.split(os.path.sep)[-2]), xlabel='frames',
           ylabel='slices')
    plt.savefig(save_file + "_dice_along_t.png")

    # 3. Calculate dice along z
    dice_along_z = np.full((Z - 1, T), -1.0)
    for fr in range(seg.shape[3]):
        for sl in range(seg.shape[2] - 1):
            seg_1 = seg[:, :, sl, fr] == label_idx
            seg_2 = seg[:, :, sl + 1, fr] == label_idx
            if np.sum(seg_1) == 0.0 or np.sum(seg_2) == 0.0:
                # print('WARNING: empty segmentation. Continue.')
                continue
            else:
                dice_along_z[sl, fr] = dice_sim_coef(seg_1, seg_2)
        if (dice_along_z[:, fr] < similarity_threshold).any() and (dice_along_z[:, fr] != -1.0).any():
            print('Potential issue for frame {}. What do we do?'.format(fr))

    # np.savetxt("dice_along_z.csv", dice_along_z, delimiter=",", fmt='%0.2f')
    fig, ax = plt.subplots(figsize=(10, 3))
    img_tmp = ax.imshow(dice_along_z, cmap='RdBu', vmin=-1, vmax=1, interpolation='none', origin='lower')
    plt.yticks(range(0, Z - 1, 2))
    plt.xticks(range(0, T, 2))
    cbar = fig.colorbar(img_tmp, ax=ax, extend='both')
    cbar.minorticks_on()
    ax.set(title='Study: {}, dice along z'.format(seg_filename.split(os.path.sep)[-2]), xlabel='frames',
           ylabel='slices')
    plt.savefig(save_file + "_dice_along_z.png")


def save_matrix_png(matrix_in, save_filename):
    _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix_in, cmap='inferno', interpolation='none', origin='lower')
    plt.savefig(save_filename + '.png')


def get_list_of_files(target_dir, full_path=True, ext_str=[], strip_ext=False):
    if full_path:
        if ext_str:
            files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if
                            (not f.startswith('.') and f.endswith(ext_str))])
        else:
            files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if not f.startswith('.')])
    elif not full_path:
        if ext_str:
            files = sorted([f for f in os.listdir(target_dir) if (not f.startswith('.') and f.endswith(ext_str))])
        else:
            files = sorted([f for f in os.listdir(target_dir) if not f.startswith('.')])
    if strip_ext and ext_str:
        files = [f.split(ext_str)[-2] for f in files]
    elif strip_ext:
        files = [os.path.splitext(f)[0] for f in files]
    return files


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.remove(path)


def remove_small_subfolders(_dir, folder_size_min):
    folder_sizes = []

    # Remove small folders directly under dir (level 1)
    subfolder_dirs = get_list_of_dirs(_dir)  # get all subfolder names ignoring hidden subfolders
    for subfolder_dir in subfolder_dirs:
        folder_size = subprocess.check_output(['du', '-shm', subfolder_dir]).split()[0].decode(
            'utf-8')  # Size for each subfolder given in megabites
        folder_sizes.append(int(folder_size))

        if int(folder_size) <= folder_size_min:
            print("Removing {} ({}MB)".format(os.path.basename(subfolder_dir), folder_size))
            shutil.rmtree(subfolder_dir)
            continue

        # Remove small subfolders under subfolder_dirs (level 2)
        subsubfolder_dirs = get_list_of_dirs(subfolder_dir)  # get all subfolder names ignoring hidden subfolders
        for subsubfolder_dir in subsubfolder_dirs:
            folder_size = subprocess.check_output(['du', '-shm', subsubfolder_dir]).split()[0].decode(
                'utf-8')  # size for each subfolder given in megabites
            if int(folder_size) <= folder_size_min:
                print("Removing {} ({}MB)".format(os.path.basename(subsubfolder_dir), folder_size))
                shutil.rmtree(subsubfolder_dir)

    if folder_sizes:
        min_size = min(folder_sizes)
        max_size = max(folder_sizes)
    else:
        min_size, max_size = [-1, -1]

    return [min_size, max_size]


def get_list_of_dirs(target_dir, full_path=True):
    if full_path:
        dirs = sorted([os.path.join(target_dir, d) for d in os.listdir(target_dir) if
                       (os.path.isdir(os.path.join(target_dir, d)) and not d.startswith('.'))])
    elif not full_path:
        dirs = sorted([d for d in os.listdir(target_dir) if
                       (os.path.isdir(os.path.join(target_dir, d)) and not d.startswith('.'))])

    return dirs


def save_nifti(affine, volume, hdr2, out_dir):
    imgobj = nib.nifti1.Nifti1Image(volume, affine)

    # header info: http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
    hdr = {
        'pixdim': hdr2['pixdim'],
        'toffset': hdr2['toffset'],
        'slice_duration': hdr2['slice_duration'],
        'xyzt_units': unit_codes['mm'] | unit_codes['msec'],
        'qform_code': xform_codes['aligned'],
        'sform_code': xform_codes['scanner'],
        'datatype': data_type_codes.code[np.float32]
    }
    for k, v in hdr.items():
        if k in imgobj.header:
            imgobj.header[k] = v

    nib.save(imgobj, out_dir)


def make_zip(source_dir, target_dir, password):
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    today = date.today().strftime('%Y%m%d')
    target_file = os.path.join(target_dir, today)
    file_zip = '{}.zip'.format(target_file)
    if not os.path.exists(file_zip):
        bash_zip = ['zip', '--password', password, file_zip, '-rj', source_dir]
        subprocess.run(bash_zip)


def mount_secure(utils_dir):
    if not os.path.isdir("/Volumes/Secure/"):
        bash_diskutil = "diskutil info Secure"
        diskutil_info = subprocess.run(bash_diskutil.split(), capture_output=True).stdout
        device_node = diskutil_info.decode("utf-8").split("Device Node:")[1].split("\n")[0].replace(' ', '')
        f = open(os.path.join(utils_dir, 'PACS_to_dicom', '.secure.txt'), 'r')
        secure_pass = f.read()
        bash_mount = "diskutil apfs unlockVolume {} -passphrase {}".format(device_node, secure_pass)
        subprocess.run(bash_mount.split())


def connect_to_nas(nas_root_dir, utils_dir):
    if os.path.isdir(nas_root_dir):
        if os.listdir(nas_root_dir):
            return
        else:
            os.system("diskutil unmountDisk {}".format(nas_root_dir))
    sleep(10)
    f = open(os.path.join(utils_dir, 'PACS_to_dicom', '.nas26.txt'), 'r')
    nas_pass = f.read()
    os.system("osascript -e 'mount volume \"smb://nas26/HF_AI\" \
        as user name \"joh15\" with password \"{}\"'".format(nas_pass))


def move_dicoms_to_nas(DICOM_anon_dir, nas_data_dir):
    source_dirs = get_list_of_dirs(DICOM_anon_dir)  # case dirs in trashcan
    source_cases = get_list_of_dirs(DICOM_anon_dir, full_path=False)  # case names in trashcan
    nas_cases = get_list_of_dirs(nas_data_dir, full_path=False)  # case names in nas

    # raise if duplicated
    if set(source_cases) & set(nas_cases):
        raise Exception('Some cases are already in the NAS')

    else:
        # copy to nas and remove from trashcan
        for source_dir, source_case in zip(source_dirs, source_cases):
            target_dir = os.path.join(nas_data_dir, source_case)
            print("Moving anonymised study {} to NAS".format(source_case))
            _ = shutil.copytree(source_dir, target_dir)
            shutil.rmtree(source_dir)


def get_z_regions(seg_fdata, lower_lim, upper_lim):
    if np.argmin(seg_fdata.shape) == 2:
        z_labels_bool = (seg_fdata != 0.0).any(axis=(0, 1))
    else:
        raise Exception("Axis order for 3D image is not (x,y,z).")
    sl_number = len(z_labels_bool)
    z_labels = np.full(sl_number, -1)
    z_labels_loc = np.where(z_labels_bool)[0]
    region_locs = np.array([0, lower_lim, upper_lim, 1.0])
    slice_locs_0_1 = np.linspace(0, 1, len(z_labels_loc))
    base_slices = slice_locs_0_1 < region_locs[1]
    apex_slices = slice_locs_0_1 > region_locs[2]
    mid_slices = ~(base_slices | apex_slices)

    z_labels[np.where(~z_labels_bool)[0]] = 0  # background slices
    z_labels[z_labels_loc[base_slices]] = 1  # basal slices
    z_labels[z_labels_loc[mid_slices]] = 2  # middle slices
    z_labels[z_labels_loc[apex_slices]] = 3  # apical slices

    return z_labels


def get_largest_cc(binary):
    """ Get the largest connected component in the foreground."""

    cc, n_cc = cc3d.connected_components(binary, return_N=True,
                                         connectivity=26)
    max_n, max_area = -1, -1
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_n, max_area = n, area
    largest_cc = (cc == max_n)
    return largest_cc


def calculate_ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def get_temporal_sequences(_source_ID_dir, _sequences, min_TT, logger):
    _temporal_seq_index = np.zeros((len(_sequences)))
    _tt_per_seq = []
    _dcm_params = np.zeros((len(_sequences), 10), dtype=object)
    seq_to_reject = np.zeros((len(_sequences)))
    axis = np.zeros((len(_sequences), 6))
    dcm_files_seq = []
    dcm_files_seq_all = []

    for s, sequence in enumerate(_sequences):
        try:
            if 'argus' in sequence.lower():
                seq_to_reject[s] = 1
                continue
            _sequence_dir = os.path.join(_source_ID_dir, sequence)
            dicom_files = get_list_of_files(_sequence_dir, full_path=False, ext_str='.dcm')
            tt = -1 * np.ones(len(dicom_files))
            x = -1 * np.ones(len(dicom_files))
            y = -1 * np.ones(len(dicom_files))
            dx = -1 * np.ones(len(dicom_files))
            dy = -1 * np.ones(len(dicom_files))
            dz = -1 * np.ones(len(dicom_files))
            SeriesNumber = -1 * np.ones(len(dicom_files))
            InstanceNumber = -1 * np.ones(len(dicom_files))
            SeriesDescription = -1 * np.ones(len(dicom_files), dtype=object)
            SliceLocation = -1 * np.ones(len(dicom_files))
            axis_aux = -1 * np.ones((len(dicom_files), 6))
            ProtocolName = -1 * np.ones(len(dicom_files), dtype=object)
            img_to_delete = []
            aux = get_list_of_files(_sequence_dir, full_path=True, ext_str='.dcm')
            aux = np.array(aux)
            for dicom_file_counter, dicom_file in enumerate(dicom_files):
                dicom_data = read_file(os.path.join(_sequence_dir, dicom_file))  # read metadata from DICOM file
                if hasattr(dicom_data, 'TriggerTime'):
                    # Trigger time
                    tt[dicom_file_counter] = float(dicom_data.TriggerTime)  # get trigger time for DICOM
                    if hasattr(dicom_data, 'Columns'):
                        x[dicom_file_counter] = dicom_data.Columns
                    if hasattr(dicom_data, 'Rows'):
                        y[dicom_file_counter] = dicom_data.Rows
                    if hasattr(dicom_data, 'PixelSpacing'):
                        dx[dicom_file_counter] = dicom_data.PixelSpacing[0]
                        dy[dicom_file_counter] = dicom_data.PixelSpacing[1]
                    if hasattr(dicom_data, 'SeriesNumber'):
                        SeriesNumber[dicom_file_counter] = dicom_data.SeriesNumber
                    if hasattr(dicom_data, 'InstanceNumber'):
                        InstanceNumber[dicom_file_counter] = dicom_data.InstanceNumber
                    if hasattr(dicom_data, 'SliceThickness'):
                        dz[dicom_file_counter] = float(dicom_data.SliceThickness)
                    elif hasattr(dicom_data, 'SpacingBetweenSlices'):
                        dz[dicom_file_counter] = float(dicom_data.SpacingBetweenSlices)
                    if hasattr(dicom_data, 'ProtocolName'):
                        ProtocolName[dicom_file_counter] = dicom_data.ProtocolName
                    if hasattr(dicom_data, 'SeriesDescription'):
                        SeriesDescription[dicom_file_counter] = dicom_data.SeriesDescription
                    if hasattr(dicom_data, 'SliceLocation'):
                        SliceLocation[dicom_file_counter] = dicom_data.SliceLocation
                    if hasattr(dicom_data, 'ImageOrientationPatient'):
                        ax = np.array([float(x) for x in dicom_data.ImageOrientationPatient])
                        axis_aux[dicom_file_counter, :] = ax
                else:
                    img_to_delete.append(dicom_file_counter)

            if len(img_to_delete) > 1:
                x = np.delete(x, np.array(img_to_delete))
                y = np.delete(y, np.array(img_to_delete))
                dx = np.delete(dx, np.array(img_to_delete))
                dy = np.delete(dy, np.array(img_to_delete))
                dz = np.delete(dz, np.array(img_to_delete))
                SeriesDescription = np.delete(SeriesDescription, np.array(img_to_delete))
                tt = np.delete(tt, np.array(img_to_delete))
                SeriesNumber = np.delete(SeriesNumber, np.array(img_to_delete))
                InstanceNumber = np.delete(InstanceNumber, np.array(img_to_delete))
                SliceLocation = np.delete(SliceLocation, np.array(img_to_delete))
                axis_aux = np.delete(axis_aux, np.array(img_to_delete), axis=0)
                ProtocolName = np.delete(ProtocolName, np.array(img_to_delete))
                aux = np.delete(aux, np.array(img_to_delete))
            if len(aux) == 0:
                seq_to_reject[s] = 1
                continue

            # Params
            # print(sequence)
            # 1. X
            if len(np.unique(x)) == 1:
                _dcm_params[s, 0] = np.unique(x)[0]
            else:
                seq_to_reject[s] = 1
                logger.error('X no available for seq {}'.format(sequence))
            # 2. Y
            if len(np.unique(y)) == 1:
                _dcm_params[s, 1] = np.unique(y)[0]
            else:
                seq_to_reject[s] = 1
                logger.error('Y no available for seq {}'.format(sequence))
            # 3. dx
            if len(np.unique(dx)) == 1:
                _dcm_params[s, 2] = np.unique(dx)[0]
            else:
                seq_to_reject[s] = 1
                logger.error('dx no available for seq {}'.format(sequence))
            # 4. dy
            if len(np.unique(dy)) == 1:
                _dcm_params[s, 3] = np.unique(dy)[0]
            else:
                seq_to_reject[s] = 1
                logger.error('dy no available for seq {}'.format(sequence))
            # 5. dz
            if len(np.unique(dz)) == 1:
                _dcm_params[s, 4] = np.unique(dz)[0]
            else:
                seq_to_reject[s] = 1
                logger.error('dz no available for seq {}'.format(sequence))
            # 6. Max InstanceNumber
            _dcm_params[s, 5] = np.max(InstanceNumber)
            # 7. SeriesDescription
            if len(np.unique(SeriesDescription)) == 1:
                _dcm_params[s, 6] = np.unique(SeriesDescription)[0]
            else:
                seq_to_reject[s] = 1
                logger.error('SeriesDescription no available for seq {}'.format(sequence))
            # 8. ProtocolName
            if len(np.unique(ProtocolName)) == 1:
                _dcm_params[s, 7] = np.unique(ProtocolName)[0]
            else:
                seq_to_reject[s] = 1
                logger.error('ProtocolName no available for seq {}'.format(sequence))
            # 9. Multi seq
            if len(np.unique(tt)) > 1 and np.sum(tt == np.min(tt)) > 1:
                _dcm_params[s, 8] = 1
            # 10. series_nb
            if len(np.unique(SeriesNumber)) == 1:
                _dcm_params[s, 9] = np.unique(SeriesNumber)[0]
            else:
                seq_to_reject[s] = 1
                logger.error('dz no available for seq {}'.format(sequence))

            axis_aux = np.squeeze(np.unique(axis_aux, axis=0))
            if len(axis_aux) == 6:
                axis[s, :] = axis_aux
            else:
                logger.error('Wrong size for axis for {}'.format(sequence))
                seq_to_reject[s] = 1

            tt_len = len(np.unique(tt))
            if tt_len > min_TT and seq_to_reject[s] != 1:
                _temporal_seq_index[s] = 1
                _tt_per_seq.append(list(tt))
                if len(aux[tt == tt.min()].tolist()) > 1:
                    dcm_files_seq.append([sequence, aux[tt == tt.min()].tolist()])
                else:
                    dcm_files_seq.append([sequence, [aux[tt == tt.min()]]])
                dcm_files_seq_all.append([sequence, [aux]])
        except:
            seq_to_reject[s] = 1
            logger.error('Sequence {0}: unknown issues'.format(sequence))
    inds = np.where(seq_to_reject == 1)[0]
    if len(inds) > 1:
        _sequences = np.delete(_sequences, inds, axis=0)
        _temporal_seq_index = np.delete(_temporal_seq_index, inds, axis=0)
        _dcm_params = np.delete(_dcm_params, inds, axis=0)
        axis = np.delete(axis, inds, axis=0)

    _sequences_non_temp = _sequences[np.logical_not(_temporal_seq_index.astype(bool))]
    _sequences_temporal = _sequences[_temporal_seq_index.astype(bool)]
    ind = np.where(_temporal_seq_index == 0)[0]
    _dcm_params = np.delete(_dcm_params, ind, axis=0)
    axis = np.delete(axis, ind, axis=0)
    if not len(dcm_files_seq_all) == len(_sequences_temporal):
        logger.error('Wrong number of sequences deleted')

    return _tt_per_seq, _sequences_temporal, _sequences_non_temp, _dcm_params, axis, dcm_files_seq, dcm_files_seq_all
