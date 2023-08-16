from datetime import datetime
import os

import numpy as np
import pandas as pd

from common_utils.utils import set_logger
from compute_atrial_strain import compute_atria_params


UKBB_dir = "/motion_repository/UKBiobank/"
CMR_txt = os.path.join(UKBB_dir, "metadata/October2022/subjects_with_CMR.txt")
study_IDs = np.loadtxt(CMR_txt).astype(int)
study_IDs = np.unique(study_IDs)
data_dir = os.path.join(UKBB_dir, "data")

log_dir = '/media/ec17/WINDOWS_DATA/Flow_project/Atrial_strain/log'
time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
log_txt_file = os.path.join(log_dir, f"compute_atrial_strain_{time_file}.txt")
logger = set_logger(log_txt_file)
logger.info("Starting computing atrial parameters for lax\n")

# Options
# study_IDs = study_IDs[:3]
recalculate_points = False  # If False, use existing saved points
overwrite_summary_csv = False  # If False, append to existing

summary_csv_file = os.path.join(UKBB_dir, 'results/atrial_strain/atrial_strain_params.csv')
if overwrite_summary_csv and os.path.exists(summary_csv_file):
    os.remove(summary_csv_file)
if not overwrite_summary_csv and os.path.exists(summary_csv_file):
    already_run_IDs = pd.read_csv(summary_csv_file).values[:, 0]
else:
    already_run_IDs = []

error_subjects = []
for idx, study_ID in enumerate(study_IDs):
    logger.info('-'*40)
    logger.info(f"[{idx+1}/{len(study_IDs)}]: {study_ID}")

    if study_ID in already_run_IDs:
        logger.info('Already processed. Skipping.')
        continue

    subject_id_x = f"{str(study_ID)[:2]}xxxxx"
    subject_dir = os.path.join(data_dir, subject_id_x, str(study_ID))
    results_dir = os.path.join(
        subject_dir, 'results_nnUnet', 'results_atrial_strain')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    try:
        df, header = compute_atria_params(study_ID, subject_dir, results_dir, logger, recalculate_points)
        if df is not None:
            write_header = header if not os.path.exists(summary_csv_file) else False
            df.to_csv(summary_csv_file, mode='a', header=write_header, index=False)
    except:
        logger.error('Unexpected error. Logging ID')
        error_subjects.append(study_ID)
    
error_file = os.path.join(UKBB_dir, 'results/atrial_strain/error_subjects.txt')
with open(error_file, 'w') as f:
    for item in error_subjects:
        f.write("{}\n".format(item))
logger.info(f"Closing compute_atrial_strain_{time_file}.txt")
