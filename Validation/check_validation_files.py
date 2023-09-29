"""
Check which files are available for validation for the two segmentation tasks (2ch/4ch)
"""

from glob import glob as glob
import numpy as np
import os

dir_2ch_ts = "/home/ec17/bioeng391-pc/data/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task303_la_2Ch_LARA/Results/"
dir_4ch_ts = "/home/ec17/bioeng391-pc/data/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task305_la_4Ch_LARA/Results/"
dir_2ch_tr = "/home/ec17/bioeng391-pc/data/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task303_la_2Ch_LARA/imagesTr/"
dir_4ch_tr = "/home/ec17/bioeng391-pc/data/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task305_la_4Ch_LARA/imagesTr/"

files_2ch_ts = glob(f"{dir_2ch_ts}/*.nii.gz")
files_2ch_tr = glob(f"{dir_2ch_tr}/*.nii.gz")
files_2ch = [os.path.basename(x) for x in files_2ch_ts] + [os.path.basename(x) for x in files_2ch_tr]
files_4ch_ts = glob(f"{dir_4ch_ts}/*.nii.gz")
files_4ch_tr = glob(f"{dir_4ch_tr}/*.nii.gz")
files_4ch = [os.path.basename(x) for x in files_4ch_ts] + [os.path.basename(x) for x in files_4ch_tr]

ids_2ch = np.unique([f[:7] for f in files_2ch])
ids_4ch = np.unique([f[:7] for f in files_4ch])

in2ch_not4ch = list(set(ids_2ch) - set(ids_4ch))
in4ch_not2ch = list(set(ids_4ch) - set(ids_2ch))
print(len(in2ch_not4ch))
print(len(in4ch_not2ch))
