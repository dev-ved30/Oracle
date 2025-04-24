import os
import sncosmo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# <----- Hyperparameters for the model.....Unfortunately ----->
marker_style = 'o'
marker_size = 3
linewidth = 0.5
ztf_filter_marker_colors = {
    'g': 'seagreen', # g
    'r': 'deeppink', # r
    'i': 'dodgerblue' # i
}

# Paths for the raw lc data and the images
lc_dir = Path('./data/BTS/bts_lcs')
lc_image_dir = Path('./data/BTS/bts_lc_images')

all_file_names = os.listdir(lc_dir)
lc_image_dir.mkdir(exist_ok=True)

for file_name in all_file_names:

    object_df = pd.read_csv(f'./data/BTS/bts_lcs/{file_name}').sort_values(by='jd')
    ztf_object_id = file_name.split('_')[0]

    # Remove any entries with negative flux
    object_df = object_df[object_df['fnu_microJy'] > 0]
    object_df = object_df[object_df['fnu_microJy_unc'] > 0]

    if len(object_df) != 0:

        flux = object_df['fnu_microJy'].to_numpy()
        flux_err = object_df['fnu_microJy_unc'].to_numpy()
        jd = object_df['jd'].to_numpy()
        filters = np.array([pb.split('_')[1] for pb in object_df['passband'].to_numpy()])

        color = np.array([ztf_filter_marker_colors[f] for f in filters])
        marker_sizes = np.ones_like(jd, dtype=float) * marker_size

        # Create a figure and axes
        fig, ax = plt.subplots(1, 1)

        # Remove all spines (black frame)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in ztf_filter_marker_colors.keys():
            
            idx = np.where(filters == i)[0]
            ax.errorbar(jd[idx], flux[idx], yerr=flux_err[idx], linewidth=linewidth, color=ztf_filter_marker_colors[i])
            ax.scatter(jd[idx], flux[idx], marker=marker_style, sizes=marker_sizes, color=ztf_filter_marker_colors[i])

        # Save the figure as PNG with the desired DPI
        dpi = 100  # Dots per inch (adjust as needed)

        # Set the figure size in inches
        width_in = 2.56  # Desired width in inches (256 pixels / 100 dpi)
        height_in = 2.56  # Desired height in inches (256 pixels / 100 dpi)

        fig.set_size_inches(width_in, height_in)

        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        #plt.gca().invert_yaxis()

        plt.savefig(f"{lc_image_dir}/{ztf_object_id}.png", dpi=dpi)
        plt.close()


