import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_boxplot(x_values, title, atrial_dir):
    x_values = x_values[~np.isnan(x_values)]
    mean = x_values.mean()
    std = x_values.std()

    fig1, ax1 = plt.subplots()
    ax1.boxplot(x_values, showmeans=True)
    ax1.set_title(title)

    # line = bp['medians']
    # x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ={:.2f}'.format(mean, std)
    ax1.annotate(text, xy=(1.1, mean))
    fig1.savefig(os.path.join(atrial_dir, f'{title}_boxplot.png'))


def do_calculate(atrial_csv, atrial_dir):
    df = pd.read_csv(atrial_csv)
    df_header = np.array(df.columns)
    df = df.values

    columns = ['LA_vol_area_combo', 'LA_vol_SR_combo',
               'LA_strain_circum_combined_ES',
               'LA_strain_circum_combined_max',
               'LA_strain_long_combo_ES', 'LA_strain_long_combo_max',
               'RA_volume_area', 'RA_volume_SR',
               'RA_strain_circum_ES', 'RA_strain_circum_max',
               'RA_strain_long_ES', 'RA_strain_long_max',
               'LA_strainRate_circum_combo', 'LA_strainRate_longit_combo',
               'RA_strainRate_circum', 'RA_strainRate_longit']
    
    for current_column in columns:
        print(f'Processing {current_column}')
        x_ind = np.where(current_column == df_header)[0][0]
        x = df[:, x_ind]
        create_boxplot(x, current_column, atrial_dir)


def main():
    UKBB_dir = "/motion_repository/UKBiobank/"
    atrial_dir = os.path.join(UKBB_dir, "results/atrial_strain")

    atrial_csv = os.path.join(atrial_dir, "atrial_strain_params.csv")

    do_calculate(atrial_csv, atrial_dir)


if __name__ == "__main__":
    import sys

    sys.path.append("/media/ec17/WINDOWS_DATA/Flow_project/Atrial_strain")
    main()
