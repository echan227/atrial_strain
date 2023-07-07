from datetime import datetime
import os

import numpy as np

from common_utils.utils import set_logger
from compute_atrial_strain import compute_atria_params


CMR_txt = (
    "/motion_repository/UKBiobank/metadata/October2022/subjects_with_CMR.txt")
study_IDs = np.loadtxt(CMR_txt).astype(int)
study_IDs = np.unique(study_IDs)
data_dir = "/motion_repository/UKBiobank/data/"

log_dir = '/media/ec17/WINDOWS_DATA/Flow_project/Atrial_strain/log'
time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
log_txt_file = os.path.join(log_dir, f"compute_atrial_strain_{time_file}.txt")
logger = set_logger(log_txt_file)
logger.info("Starting computing atrial parameters for lax\n")

study_IDs = study_IDs[:10]

for idx, study_ID in enumerate(study_IDs):
    print(f"[{idx+1}/{len(study_IDs)}]: {study_ID}")
    subject_id_x = f"{str(study_ID)[:2]}xxxxx"
    subject_dir = os.path.join(data_dir, subject_id_x, str(study_ID))
    results_dir = os.path.join(
        subject_dir, 'results_nnUnet', 'results_atrial_strain')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    compute_atria_params(study_ID, subject_dir, results_dir, logger)

logger.info(f"Closing compute_atrial_strain_{time_file}.txt")
