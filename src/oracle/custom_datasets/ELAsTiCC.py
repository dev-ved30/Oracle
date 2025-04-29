import io
import torch

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from oracle.architectures import ORACLE1, ORACLE2_lite_swin
from oracle.taxonomies import ORACLE_Taxonomy
from oracle.constants import ELAsTiCC_to_Astrophysical_mappings

time_independent_feature_list = ['MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y']
time_dependent_feature_list = ['MJD', 'PHOTFLAG', 'FLUXCAL', 'FLUXCALERR', 'BAND']
book_keeping_feature_list = ['SNID', 'ELASTICC_class']

# Path to this file's directory
here = Path(__file__).resolve().parent

# Go up to the root, then into data/ and then get the parquet file
parquet_path = str(here.parent.parent.parent / "data" / 'ELasTiCC' / 'complete.parquet')

# <----- Hyperparameters for the model.....Unfortunately ----->
marker_style = 'o'
marker_size = 50
linewidth = 0.75

# Mean wavelengths for the LSST pass bands in micrometers
LSST_passband_to_wavelengths = {
    'u': (320 + 400) / (2 * 1000),
    'g': (400 + 552) / (2 * 1000),
    'r': (552 + 691) / (2 * 1000),
    'i': (691 + 818) / (2 * 1000),
    'z': (818 + 922) / (2 * 1000),
    'Y': (950 + 1080) / (2 * 1000),
}

# Mean wavelength to colors for plotting
LSST_passband_wavelengths_to_color = {
    LSST_passband_to_wavelengths['u']: np.array((0, 127, 255))/255,
    LSST_passband_to_wavelengths['g']: np.array((127, 0, 255))/255,
    LSST_passband_to_wavelengths['r']: np.array((0, 255, 127))/255,
    LSST_passband_to_wavelengths['i']: np.array((127, 255, 0))/255,
    LSST_passband_to_wavelengths['z']: np.array((255, 127, 0))/255,
    LSST_passband_to_wavelengths['Y']: np.array((255, 0, 127))/255,
}

max__elasticc_ts_len = 500

flag_value = -9

# Flag values for missing data of static feature according to elasticc
missing_data_flags = [-9, -99, -999, -9999, 999]

def replace_missing_value_flags(examples):

    # Get the number of samples
    N_samples = len(examples['SNID'])

    for i in range(N_samples):

        # Replace any missing value flag with -9
        for feature in time_independent_feature_list:
            if examples[feature][i] in missing_data_flags:
                examples[feature][i] = flag_value
    
    return examples

def mask_off_saturations(examples):

    # Get the number of samples
    N_samples = len(examples['SNID'])

    for i in range(N_samples):

        # Mask off the saturations
        PHOTFLAG = examples['PHOTFLAG'][i]
        saturation_mask =  (np.array(PHOTFLAG) & 1024) == 0 

        # Remove the saturations
        examples['MJD'][i] = np.array(examples['MJD'][i])[saturation_mask]
        examples['FLUXCAL'][i] = np.array(examples['FLUXCAL'][i])[saturation_mask] 
        examples['FLUXCALERR'][i] = np.array(examples['FLUXCALERR'][i])[saturation_mask]
        examples['BAND'][i] = np.array(examples['BAND'][i])[saturation_mask]
        examples['PHOTFLAG'][i] = np.array(examples['PHOTFLAG'][i])[saturation_mask]
    
    return examples

def truncate_lcs_fractionally(examples, fraction=None):

    # Get the number of samples
    N_samples = len(examples['SNID'])

    # Fraction of their original length to augment the light curves to
    if fraction is None:
        # Randomly sample a fraction between 0.1 and 1.0
        fractions_array = np.random.uniform(0.1, 1.0, N_samples)
    else:
        # Use the provided fraction
        fractions_array = np.array([fraction]*N_samples)

    for i in range(N_samples):

        # Apply the fraction limit on the light curve
        original_obs_count = len(examples['MJD'][i])

        # Find the new length of the light curve
        new_obs_count = int(original_obs_count * fractions_array[i])
        if new_obs_count < 1:
            new_obs_count = 1
        
        # Truncate the light curve to its new length and apply other transforms
        examples['MJD'][i] = np.array(examples['MJD'][i])[:new_obs_count] - examples['MJD'][i][0] # Subtract out time of first observation (could be a detection or a non detection)
        examples['FLUXCAL'][i] =  np.array(examples['FLUXCAL'][i])[:new_obs_count] 
        examples['FLUXCALERR'][i] = np.array(examples['FLUXCALERR'][i])[:new_obs_count]
        examples['BAND'][i] = np.array([LSST_passband_to_wavelengths[b] for b in np.array(examples['BAND'][i])[:new_obs_count]]) # Convert the pass band label (ugrizY) to the mean wavelength of the filter
        examples['PHOTFLAG'][i] = np.where((np.array(examples['PHOTFLAG'][i])[:new_obs_count] & 4096 != 0), 1, 0) # 1 for detections, 0 for non detections
    
    return examples

def add_lc_plots(examples):

    # Get the number of samples
    N_samples = len(examples['SNID'])

    lc_plots_array = []

    for i in range(N_samples):

        flux = examples['FLUXCAL'][i]
        flux_err = examples['FLUXCALERR'][i]
        jd = examples['MJD'][i]
        filters = examples['BAND'][i]

        # Create a figure and axes
        fig, ax = plt.subplots(1, 1)

        # Remove all spines (black frame)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for wavelength in LSST_passband_wavelengths_to_color.keys():
            
            idx = np.where(filters == wavelength)[0]
            ax.errorbar(jd[idx], flux[idx], yerr=flux_err[idx], linewidth=linewidth, color=LSST_passband_wavelengths_to_color[wavelength])
            ax.scatter(jd[idx], flux[idx], marker=marker_style, s=marker_size, color=LSST_passband_wavelengths_to_color[wavelength])

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
        lc_plots_array.append(img_arr)
        
        # Close the buffer
        buf.close()

    # Add plots of the light curves
    examples['lc_plots'] = lc_plots_array

    return examples

def replace_labels(examples, mapper: dict):

    # Get the number of samples
    N_samples = len(examples['SNID'])

    for i in range(N_samples):

        examples['ELASTICC_class'][i] = mapper[examples['ELASTICC_class'][i]]

    return examples

def collate_ELAsTiCC_lc_data(batch, includes_plots=True):

    ts_data = []
    static_data = []
    plots = []
    labels = []
    snids = []
    lengths = []

    for b in batch:

        length = len(b['MJD'])
        n_ts_features = len(time_dependent_feature_list)

        # Fill the array with the time series data
        array = np.zeros((length, n_ts_features), dtype=np.float32)
        array[:, 0] = b['MJD']
        array[:, 1] = b['FLUXCAL']
        array[:, 2] = b['FLUXCALERR']
        array[:, 3] = b['BAND']
        array[:, 4] = b['PHOTFLAG']
        ts_data.append(array)

        # Fill the array with the static data
        s = []
        for feature in time_independent_feature_list:
            s.append(b[feature])

        static_data.append(np.array(s, dtype=np.float32))
        lengths.append(length)
        labels.append(b['ELASTICC_class'])
        snids.append(b['SNID'])

        if includes_plots:
            plots.append(b['lc_plots'])

        
    ts_data = [torch.from_numpy(x) for x in ts_data]
    ts_data = pad_sequence(ts_data, batch_first=True, padding_value=flag_value)

    static_data = torch.from_numpy(np.array(static_data))
    
    d = {
        'ts_data':ts_data,
        'static_data':static_data,
        'lengths':lengths,
        'labels':labels,
        'SNID':snids
    }

    if includes_plots:
        plots = torch.from_numpy(np.array(plots))
        d['lc_plots'] = plots

    return d

def get_ELAsTiCC_dataset(split, parquet_path=parquet_path):

    # Load the whole dataset
    dataset = load_dataset("parquet", data_files=parquet_path, split='train')

    # Break the dataset into train, test, and val splits
    seed = 42
    ds_train_test = dataset.train_test_split(test_size=0.3, seed=seed)
    ds_test_val = ds_train_test['train'].train_test_split(test_size=0.1, seed=seed)

    # Create a new dataset dictionary
    dataset_splits = DatasetDict({
        'train': ds_test_val['train'],
        'validation': ds_test_val['test'],
        'test': ds_train_test['test']
    })

    # Select the correct split
    dataset = dataset_splits[split]

    # Only select the useful columns
    dataset = dataset.select_columns(time_independent_feature_list+time_dependent_feature_list+book_keeping_feature_list)

    # Set the appropriate formatting for all the data
    dataset.set_format(type="torch")

    return dataset

def show_batch(images, labels, n=16):

    # Get the first n images
    images = images[:n]

    # Create a grid of images (4x4)
    grid_size = int(n ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        img = images[i]
        label = labels[i]
        if img.shape[0] == 1:  # grayscale
            img = img.squeeze(0)
            img = img.numpy().astype(int) 
            ax.imshow(img, cmap='gray')
        else:  # RGB
            img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            img = img.numpy().astype(int) 
            ax.imshow(img)

        ax.set_title(f"{label}", fontsize=8) 

    plt.tight_layout()
    plt.show()

def main():

    dataset = get_ELAsTiCC_dataset('train')

    # Create a custom function with all the transforms
    def custom_transforms_function(examples):

        # Mask off any saturations
        examples = mask_off_saturations(examples)

        # Replace any missing values with a flag
        examples = replace_missing_value_flags(examples)

        # Truncate LC's before building the plots
        examples = truncate_lcs_fractionally(examples)

        # Add plots of the light curve for the vision transformer
        examples = add_lc_plots(examples)

        # Convert the labels from ELAsTiCC labels to astrophysically meaningful labels
        examples = replace_labels(examples, ELAsTiCC_to_Astrophysical_mappings)

        return examples

    dataset = dataset.with_transform(custom_transforms_function)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_ELAsTiCC_lc_data, shuffle=True)

    taxonomy = ORACLE_Taxonomy()
    model = ORACLE1(taxonomy)
    model.eval()

    model_VT = ORACLE2_lite_swin(taxonomy)
    model_VT.eval()

    for batch in dataloader:

        print(list(batch.keys()))

        logits = model(batch)
        
        print(model.predict_class_probabilities_df(batch))
        print(model_VT.predict_class_probabilities_df(batch))

        print(logits.shape)
        print(batch['lc_plots'].shape)
        print(batch['ts_data'].shape)
        print(batch['static_data'].shape)
        print(batch['labels'])

        show_batch(batch['lc_plots'], batch['labels'])
        break

if __name__ == '__main__':

    main()