"""
Run inference to segment LARA, LVRV in both 2ch and 4ch views and then combine
"""

# imports - stdlib
import os
import subprocess
import nibabel as nib
import numpy as np
import logging
import argparse

# imports - 3rd party:
from common_utils.load_args import Params
from common_utils.utils import get_list_of_dirs, save_nifti, set_logger

# =============================================================================
# DEFAULT DIR
# =============================================================================
DEFAULT_JSON_FILE = '/home/br14/code/Python/Atrial_strain_emily/basic_opt.json'


def run_nnunet_inference(target_dir, task_folder, script_no, script_dir):
    target_imagesTs = os.path.join(target_dir, task_folder, "imagesTs")
    inference_dir = os.path.join(target_dir, task_folder, "Results")
    subprocess.run(f'bash {script_dir} -a {target_imagesTs} -b {inference_dir} -c {script_no}',
                   shell=True)


def main():
    parser = argparse.ArgumentParser(description='Segmentation inference')
    parser.add_argument("--json_config_path", type=str, default=DEFAULT_JSON_FILE,
                        help='path of configurations')
    args = parser.parse_args()
    if os.path.exists(args.json_config_path):
        cfg = Params(args.json_config_path).dict
    else:
        raise FileNotFoundError
    local_dir = cfg['DEFAULT_LOCAL_DIR']
    nifti_dir = os.path.join(local_dir, cfg['DEFAULT_SUBDIR_NIFTI'])
    target_dir = os.path.join(local_dir, cfg['DEFAULT_NNUNET_NIFTI'])
    log_dir = os.path.join(local_dir, cfg['DEFAULT_LOG_DIR'])
    log_filename = os.path.split(__file__)[-1][:-3]
    log_file = os.path.join(log_dir, log_filename)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    set_logger(log_file)

    script_dir = '/home/br14/code/Python/AI_centre/Clint/KCL_V2/Segmentation/inference_nnunet.sh' #os.path.join(os.getcwd(), 'Segmentation', 'inference_nnunet.sh')

    log_file.info('Starting inference for Task302_la_2Ch_LVRV')
    run_nnunet_inference(target_dir, 'Task302_la_2Ch_LVRV', 302, script_dir)
    log_file.info('Finished')
    log_file.info('-'*40)

    log_file.info('Starting inference for Task303_la_2Ch_LARA')
    run_nnunet_inference(target_dir, 'Task303_la_2Ch_LARA', 303, script_dir)
    log_file.info('Finished')
    log_file.info('-'*40)

    log_file.info('Starting inference for Task304_la_4Ch_LVRV')
    run_nnunet_inference(target_dir, 'Task304_la_4Ch_LVRV', 304, script_dir)
    log_file.info('Finished')
    log_file.info('-'*40)

    log_file.info('Starting inference for Task305_la_4Ch_LARA')
    run_nnunet_inference(target_dir, 'Task305_la_4Ch_LARA', 305, script_dir)
    log_file.info('Finished')
    log_file.info('-'*40)


    # Find studies to analyse
    # study_IDs = get_list_of_dirs(nifti_dir, full_path=False)

    # =============================================================================
    # SAX
    # =============================================================================
    # _doit1(nifti_dir, target_dir, study_IDs, 'SAX', 301, 'Task301_SAX', 'sa.nii.gz', 'sa_seg_nnUnet.nii.gz', script_dir)

    # =============================================================================
    # LA 2CH
    # =============================================================================
    # _doit2(nifti_dir, target_dir, study_IDs, 'la_2Ch', 302, 303, 'Task302_la_2Ch_LVRV', 'Task303_la_2Ch_LARA',
    #        'la_2Ch.nii.gz', 'la_2Ch_seg_LVRV_nnUnet.nii.gz', 'la_2Ch_seg_LARA_nnUnet.nii.gz',
    #        'la_2Ch', script_dir)

    # =============================================================================
    # LA 4CH
    # =============================================================================
    # _doit2(nifti_dir,
    #        target_dir,
    #        study_IDs,
    #        'la 4Ch',
    #        304, 305,
    #        'Task304_la_4Ch_LVRV', 'Task305_la_4Ch_LARA',
    #        'la_4Ch.nii.gz',
    #        'la_4Ch_seg_LVRV_nnUnet.nii.gz', 'la_4Ch_seg_LARA_nnUnet.nii.gz',
    #        'la_4Ch', script_dir)


if __name__ == '__main__':
    main()
