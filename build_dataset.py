import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# <----- Hyperparameters for the model.....Unfortunately ----->
marker_style = 'o'
marker_size = 3
linewidth = 0.5
ztf_filter_marker_colors = {
    1: 'seagreen', # g
    2: 'deeppink', # r
    3: 'dodgerblue' # i
}

df = pd.read_csv('test_cand_v10_N100.csv')


for index, objectId in enumerate(np.unique(df['objectId'])):

    object_df = df[df['objectId'] == objectId].sort_values('jd')

    mag = object_df['magpsf'].to_numpy()
    mag_err = object_df['sigmapsf'].to_numpy()
    jd = object_df['jd'].to_numpy()
    color = np.array([ztf_filter_marker_colors[f] for f in object_df['fid'].to_numpy()])
    marker_sizes = np.ones_like(jd) * marker_size

    # Create a figure and axes
    fig, ax = plt.subplots(1, 1)

    for i in ztf_filter_marker_colors.keys():
        
        idx = np.where(object_df['fid'].to_numpy() == i)[0]
        ax.errorbar(jd[idx], mag[idx], yerr=mag_err[idx], linewidth=linewidth, color=ztf_filter_marker_colors[i])
        ax.scatter(jd[idx], mag[idx], marker=marker_style, sizes=marker_sizes, color=ztf_filter_marker_colors[i])

    # Save the figure as PNG with the desired DPI
    dpi = 100  # Dots per inch (adjust as needed)

    # Set the figure size in inches
    width_in = 2.56  # Desired width in inches (256 pixels / 100 dpi)
    height_in = 2.56  # Desired height in inches (256 pixels / 100 dpi)



    fig.set_size_inches(width_in, height_in)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(f"./data/BTS/LC/{index}.png", dpi=dpi)
    plt.close()


