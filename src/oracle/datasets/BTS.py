import io
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from oracle.models import Light_curve_classifier
from oracle.taxonomies import ORACLE_Taxonomy
from oracle.constants import ztf_filter_to_fid


# <----- Hyperparameters for the model.....Unfortunately ----->
marker_style = 'o'
marker_size = 3
linewidth = 0.5

ZTF_passband_to_wavelengths = {
    'g': (400 + 552) / (2 * 1000),
    'r': (552 + 691) / (2 * 1000),
    'i': (691 + 818) / (2 * 1000),
}

ZTF_fid_to_wavelengths = {
    ztf_filter_to_fid[k]: ZTF_passband_to_wavelengths[k] for k in ZTF_passband_to_wavelengths.keys()
}

# Mean wavelength to colors for plotting
ZTF_fid_to_color = {
    ztf_filter_to_fid['g']: np.array((255, 0, 0))/255,
    ztf_filter_to_fid['r']: np.array((0, 255, 0))/255,
    ztf_filter_to_fid['i']: np.array((0, 0, 255))/255,
}

flag_value = -9


from oracle.constants import ztf_filters, ztf_alert_image_order, ztf_alert_image_dimension


images = ['g_reference', 'g_science', 'g_difference', 'r_reference', 'r_science', 'r_difference', 'i_reference', 'i_science', 'i_difference']
time_dependent_feature_list = ['jd', 'magpsf', 'sigmapsf', 'fid']
book_keeping_feature_list = ['ZTFID', 'bts_class']

# Path to this file's directory
here = Path(__file__).resolve().parent

# Go up to the root, then into data/
# Path to this file's directory
here = Path(__file__).resolve().parent

# Go up to the root, then into data/ and then get the parquet file
parquet_path = str(here.parent.parent.parent / "data" / 'BTS' / 'val.parquet')

def reconstruct_images(examples):

    # Get the number of samples
    N_samples = len(examples['ZTFID'])

    for i in range(N_samples):

        for f in ztf_filters:
            for img_type in ztf_alert_image_order:

                img_data = examples[f"{f}_{img_type}"][i]

                if img_data == None:
                    examples[f"{f}_{img_type}"][i] = np.zeros(ztf_alert_image_dimension) * np.nan
                else:
                    examples[f"{f}_{img_type}"][i] = np.reshape(img_data, ztf_alert_image_dimension)

    return examples

def truncate_lcs_fractionally(examples, fraction=None):

    # Get the number of samples
    N_samples = len(examples['ZTFID'])

    # Fraction of their original length to augment the light curves to
    if fraction is None:
        # Randomly sample a fraction between 0.1 and 1.0
        fractions_array = np.random.uniform(0.1, 1.0, N_samples)
    else:
        # Use the provided fraction
        fractions_array = np.array([fraction]*N_samples)

    # Reconstruct the images
    examples = reconstruct_images(examples)

    all_bands = []

    for i in range(N_samples):

        # Apply the fraction limit on the light curve
        original_obs_count = len(examples['jd'][i])

        # Find the new length of the light curve
        new_obs_count = int(original_obs_count * fractions_array[i])
        if new_obs_count < 1:
            new_obs_count = 1
        
        # Truncate the light curve to its new length and apply other transforms
        examples['jd'][i] = np.array(examples['jd'][i])[:new_obs_count] - examples['jd'][i][0] # Subtract out time of first observation (could be a detection or a non detection)
        examples['magpsf'][i] =  np.array(examples['magpsf'][i])[:new_obs_count] 
        examples['sigmapsf'][i] = np.array(examples['sigmapsf'][i])[:new_obs_count]
        examples['fid'][i] =  np.array(examples['fid'][i])[:new_obs_count]
        bands = np.array([ZTF_fid_to_wavelengths[b] for b in examples['fid'][i]]) # Convert the pass band label (ugrizY) to the mean wavelength of the filter

        all_bands.append(bands)
    
    examples['band'] = all_bands

    return examples

def get_lc_plots(examples):

    # Get the number of samples
    N_samples = len(examples['ZTFID'])

    plots_array = []

    for i in range(N_samples):

        mag = examples['magpsf'][i]
        mag_err = examples['sigmapsf'][i]
        jd = examples['jd'][i]
        filters = examples['fid'][i]

        marker_sizes = np.ones_like(jd, dtype=float) * marker_size

        # Create a figure and axes
        fig, ax = plt.subplots(1, 1)

        # Remove all spines (black frame)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for fid in ZTF_fid_to_color.keys():
            
            idx = np.where(filters == fid)[0]
            ax.errorbar(jd[idx], mag[idx], yerr=mag_err[idx], linewidth=3, color=ZTF_fid_to_color[fid])
            ax.scatter(jd[idx], mag[idx], marker=marker_style, sizes=marker_sizes, color=ZTF_fid_to_color[fid])

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

        # Write the plot data to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        plt.close()

        # Go to the start of the buffer and read into an image
        buf.seek(0)
        im = Image.open(buf).convert('RGB')
        img_arr = np.array(im, dtype=np.float32)
        img_arr = np.permute_dims(img_arr, (2, 0, 1))

        # Add image array to the list
        plots_array.append(img_arr)
        
        # Close the buffer
        buf.close()

    return plots_array

def get_postage_stamp_plots(examples):

    # Get the number of samples
    N_samples = len(examples['ZTFID'])

    img_length = ztf_alert_image_dimension[0]
    plots_array = []

    for i in range(N_samples):

        canvas = np.zeros((256, 256)) * np.nan


        # for j, f in enumerate(ztf_filters):

        #     for k, img_type in enumerate(ztf_alert_image_order):

        #         # get the image
        #         canvas[(j)*img_length:(j+1)*img_length,(k)*img_length:(k+1)*img_length] = examples[f"{f}_{img_type}"][i]

        # Loop through all of the filters
        for j, f in enumerate(ztf_filters):
            canvas[:img_length,(j)*img_length:(j+1)*img_length] = examples[f"{f}_reference"][i]


        # Create a figure and axes
        fig, ax = plt.subplots(1, 1)

        ax.imshow(canvas, cmap='viridis')

        # Save the figure as PNG with the desired DPI
        dpi = 100  # Dots per inch (adjust as needed)

        # Set the figure size in inches
        width_in = 2.56  # Desired width in inches (256 pixels / 100 dpi)
        height_in = 2.56  # Desired height in inches (256 pixels / 100 dpi)

        fig.set_size_inches(width_in, height_in)

        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()

        # Write the plot data to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        plt.close()

        # Go to the start of the buffer and read into an image
        buf.seek(0)
        im = Image.open(buf).convert('RGB')
        img_arr = np.array(im, dtype=np.float32)
        img_arr = np.permute_dims(img_arr, (2, 0, 1))

        # Add image array to the list
        plots_array.append(img_arr)
        
        # Close the buffer
        buf.close()

    return plots_array



def add_fractionally_truncated_lc_plots(examples, fraction=None):

    # Truncate LC's before building the plots
    examples = truncate_lcs_fractionally(examples, fraction=None)

    # Add plots of the light curve for the vision transformer
    examples['plot'] = get_lc_plots(examples)

    # Add plots of the reference images in the g,r, and i bands. 
    examples['reference_images'] = get_postage_stamp_plots(examples)

    return examples


def get_BTS_dataset(split):

    # Load the whole dataset
    dataset = load_dataset("parquet", data_files=parquet_path, split='train')

    # Only select the useful columns
    dataset = dataset.select_columns(time_dependent_feature_list+images+book_keeping_feature_list)

    # Set the appropriate formatting for all the data
    dataset.set_format(type="torch")

    return dataset

def collate_BTS_lc_data(batch, includes_plots=True):

    ts_data = []
    postage_image_data = []
    plots = []
    labels = []
    ztfids = []
    lengths = []

    for b in batch:

        length = len(b['jd'])
        n_ts_features = len(time_dependent_feature_list)

        # Fill the array with the time series data
        array = np.zeros((length, n_ts_features), dtype=np.float32)
        array[:, 0] = b['jd']
        array[:, 1] = b['magpsf']
        array[:, 2] = b['sigmapsf']
        array[:, 3] = b['band']
        ts_data.append(array)

        lengths.append(length)
        labels.append(b['bts_class'])
        ztfids.append(b['ZTFID'])

        if includes_plots:
            plots.append(b['plot'])
            postage_image_data.append(b['reference_images'])

        
    ts_data = [torch.from_numpy(np.squeeze(x)) for x in ts_data]
    ts_data = pad_sequence(ts_data, batch_first=True, padding_value=flag_value)

    
    d = {
        'ts_data':ts_data,
        'lengths':lengths,
        'labels':labels,
        'SNID':ztfids
    }

    if includes_plots:
        plots = torch.from_numpy(np.squeeze(plots))
        postage_image_data = torch.from_numpy(np.squeeze(postage_image_data))
        d['plots'] = plots
        d['reference_images'] = postage_image_data

    return d

def show_batch(images, labels, n=16):
    # Get the first n images
    images = images[:n]

    # Unnormalize if needed (optional)
    # images = images * 0.5 + 0.5

    # Create a grid of images (4x4)
    grid_size = int(n ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        img = images[i]
        label = labels[i]
        if img.shape[0] == 1:  # grayscale
            img = img.squeeze(0)
            img = np.array(img,dtype=int)
            ax.imshow(img, cmap='gray')
        else:  # RGB
            img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            img = np.array(img,dtype=int)
            ax.imshow(img)

        ax.set_title(f"{label}", fontsize=8) 



    plt.tight_layout()
    plt.show()


def main():

    dataset = get_BTS_dataset('test')
    dataset = dataset.with_transform(add_fractionally_truncated_lc_plots)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_BTS_lc_data)

    taxonomy = ORACLE_Taxonomy()
    config = {
        "layer1_neurons": 512,
        "layer1_dropout": 0.3,
        "layer2_neurons": 128,
        "layer2_dropout": 0.2,
    }
    model_VT = Light_curve_classifier(config, taxonomy)
    model_VT.eval()

    for batch in dataloader:

        logits = model_VT(batch)
        print(model_VT.get_class_probabilities_df(batch))

        print(list(batch.keys()))
        

        print(batch['ts_data'].shape)
        print(batch['reference_images'].shape)
        print(batch['reference_images'])
        show_batch(batch['reference_images'], batch['labels'])

        
        # print(batch['static_data'].shape)
        # print(batch['labels'])
        # print('----------')

        break
        

if __name__ == '__main__':

    main()