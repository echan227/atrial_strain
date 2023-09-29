import os
from datetime import datetime

import numpy as np
import pandas as pd


def create_csv(atrial_csv, survival_csv, output_csv):
    df_atrial = pd.read_csv(atrial_csv)
    atrial_header = list(df_atrial)
    df_atrial = df_atrial.values
    atrial_eids = df_atrial[:, 0].astype(int)
    atrial_eids = np.array(atrial_eids.astype(str))

    df_survival = pd.read_csv(survival_csv)
    survival_header = list(df_survival)
    df_survival = df_survival.values
    survival_eids = df_survival[:, 0].astype(int)
    survival_eids = list(survival_eids.astype(str))

    no_atrial = []
    survival_shape = df_survival.shape[1] - 1
    atrial_shape = df_atrial.shape[1] - 1
    output_data = np.zeros(
        (len(survival_eids), atrial_shape + survival_shape + 1))
    for study_counter, current_eid in enumerate(survival_eids):
        print(
            f"[{study_counter + 1}/{len(survival_eids)}]: {current_eid}")
        if current_eid in atrial_eids:
            eid_ind = np.where(current_eid == atrial_eids)[0][0]
            output_data[study_counter, 0] = current_eid
            output_data[study_counter, 1:survival_shape +
                        1] = df_survival[study_counter, 1:]
            output_data[study_counter, survival_shape+1:] = df_atrial[eid_ind, 1:]
        else:
            no_atrial.append(study_counter)
            print("No atrial data available")

    # Remove rows where no atrial data exists
    output_data = np.delete(output_data, no_atrial, axis=0)
    output_header = survival_header + atrial_header[1:]
    df_output = pd.DataFrame(output_data)
    df_output.to_csv(output_csv, header=output_header, index=False)


# =============================================================================
# MAIN
# =============================================================================
def main():
    UKBB_dir = "/motion_repository/UKBiobank/"
    atrial_dir = os.path.join(UKBB_dir, "results/atrial_strain")

    atrial_csv = os.path.join(atrial_dir, "atrial_strain_params.csv")
    survival_csv = os.path.join(UKBB_dir, "results/EF1/UKBB_survival.csv")
    output_csv = os.path.join(atrial_dir, "atrial_survival.csv")

    create_csv(atrial_csv, survival_csv, output_csv)


if __name__ == "__main__":
    import sys

    sys.path.append("/media/ec17/WINDOWS_DATA/Flow_project/Atrial_strain")
    main()
