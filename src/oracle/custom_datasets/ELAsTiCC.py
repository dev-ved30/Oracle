import io
import torch

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from oracle.constants import ELAsTiCC_to_Astrophysical_mappings

# <----- constant for the dataset ----->

batch_size = 512

img_height = 256
img_width = 256
n_channels = 3

# <----- Hyperparameters for the model.....Unfortunately ----->
marker_style = 'o'
marker_size = 50
linewidth = 0.75

flag_value = -9

# Flag values for missing data of static feature according to elasticc
missing_data_flags = [-9, -99, -999, -9999, 999]


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

time_independent_feature_list = ['MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y']
time_dependent_feature_list = ['MJD', 'FLUXCAL', 'FLUXCALERR', 'BAND', 'PHOTFLAG']
book_keeping_feature_list = ['SNID', 'ELASTICC_class']

n_static_features = len(time_independent_feature_list)
n_ts_features = len(time_dependent_feature_list)
n_book_keeping_features = len(book_keeping_feature_list)

def truncate_light_curve_fractionally(x_ts, f=None):

    if f == None:
        # Get a random fraction between 0.1 and 1
        f = np.random.uniform(0.1, 1.0)
    
    original_obs_count = x_ts.shape[0]

    # Find the new length of the light curve
    new_obs_count = int(original_obs_count * f)
    if new_obs_count < 1:
        new_obs_count = 1

    # Truncate the light curve
    x_ts = x_ts[:new_obs_count, :]

    return x_ts


class ELAsTiCC_LC_Dataset(torch.utils.data.Dataset):

    def __init__(self, parquet_file_path, include_lc_plots=False, transform=None):
        super(ELAsTiCC_LC_Dataset, self).__init__()

        # Columns to be read from the parquet file
        self.columns = time_dependent_feature_list + time_independent_feature_list + book_keeping_feature_list

        self.parquet_file_path = parquet_file_path
        self.transform = transform
        self.include_lc_plots = include_lc_plots

        print(f'Loading dataset from {self.parquet_file_path}\n')
        self.parquet_df = pl.read_parquet(self.parquet_file_path, columns=self.columns)
        self.columns_dtypes = self.parquet_df.schema

        self.clean_up_dataset()
               
    def __len__(self):

        return len(self.parquet_df)

    def __getitem__(self, index):

        row = self.parquet_df.row(index, named=True) 

        snid = row['SNID']
        ELAsTiCC_class_name = row['ELASTICC_class']
        astrophysical_class = ELAsTiCC_to_Astrophysical_mappings[ELAsTiCC_class_name]

        lc_length = len(row['MJD_clean'])

        time_series_data = np.zeros((lc_length, n_ts_features), dtype=np.float32)
        for i, feature in enumerate(time_dependent_feature_list):
            time_series_data[:,i] = np.array(row[f"{feature}_clean"], dtype=np.float32)
        time_series_data = torch.from_numpy(time_series_data)

        static_data = torch.zeros(n_static_features)
        for i, feature in enumerate(time_independent_feature_list):
            static_data[i] = row[feature]

        if self.transform != None:
            time_series_data = self.transform(time_series_data)

        dictionary = {
            'ts': time_series_data,
            'static': static_data,
            'label': astrophysical_class,
            'SNID': snid,
        }

        if self.include_lc_plots:
            light_curve_plot = self.get_lc_plots(row)
            dictionary['lc_plot'] = light_curve_plot
        
        return dictionary
    
    def clean_up_dataset(self):

        def remove_saturations_from_series(phot_flag_arr, feature_arr):
            
            saturation_mask =  (np.array(phot_flag_arr) & 1024) == 0 
            feature_arr = np.array(feature_arr)[saturation_mask].tolist()

            return feature_arr
        
        def replace_missing_flags(x):

            if x in missing_data_flags:
                return float(flag_value)
            else:
                return x
            
        print("Starting Dataset Transformation:")

        print("Replacing band labels with mean wavelengths...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("BAND").map_elements(lambda x: [LSST_passband_to_wavelengths[band] for band in x], return_dtype=pl.List(pl.Float64)).alias("BAND")
        )

        # Remove the saturations form the time series data. PHOTFLAG is handled later
        ts_feature_list = [x for x in time_dependent_feature_list if x != "PHOTFLAG"]
        for feature in ts_feature_list:
            print(f"Dropping saturations from {feature} series...")
            self.parquet_df = self.parquet_df.with_columns(
                pl.struct(["PHOTFLAG", feature]).map_elements(lambda x: remove_saturations_from_series(x['PHOTFLAG'], x[feature]), return_dtype=pl.List(pl.Float64)).alias(f"{feature}_clean")
            )

        print(f"Removing saturations from PHOTFLAG series...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("PHOTFLAG").map_elements(lambda x: remove_saturations_from_series(x, x), return_dtype=pl.List(pl.Int64)).alias("PHOTFLAG_clean")
        )

        print("Subtracting time of first observation...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("MJD_clean").map_elements(lambda x: (np.array(x) - min(x)).tolist(), return_dtype=pl.List(pl.Float64)).alias("MJD_clean")
        )

        for feature in time_independent_feature_list:
            print(f"Replacing missing values in {feature} series...")
            self.parquet_df = self.parquet_df.with_columns(
                pl.col(feature).map_elements(lambda x: replace_missing_flags(x), return_dtype=self.columns_dtypes[feature]).alias(f"{feature}_clean")
            )
        print('Done!\n')

    def get_lc_plots(self, row):

        # Get the light curve data
        jd = np.array(row['MJD_clean'])
        flux = np.array(row['FLUXCAL_clean'])
        flux_err = np.array(row['FLUXCALERR_clean'])
        filters = np.array(row['BAND_clean'])
        phot_flag = np.array(row['PHOTFLAG_clean']) # NOTE: might want to use a different marker for ND

        # Create a figure and axes
        fig, ax = plt.subplots(1, 1)

        # Set the figure size in inches
        width_in = 2.56  # Desired width in inches (256 pixels / 100 dpi)
        height_in = 2.56  # Desired height in inches (256 pixels / 100 dpi)

        fig.set_size_inches(width_in, height_in)

        # Remove all spines (black frame)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for wavelength in LSST_passband_wavelengths_to_color.keys():
            
            idx = np.where(filters == wavelength)[0]
            ax.errorbar(jd[idx], flux[idx], yerr=flux_err[idx], linewidth=linewidth, fmt=marker_style, color=LSST_passband_wavelengths_to_color[wavelength])
            #ax.scatter(jd[idx], flux[idx], marker=marker_style, s=marker_size, color=LSST_passband_wavelengths_to_color[wavelength])

        # Save the figure as PNG with the desired DPI
        dpi = 100  # Dots per inch (adjust as needed)

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
        img_arr = torch.from_numpy(img_arr)

        # Close the buffer
        buf.close()

        return img_arr

def custom_collate_ELAsTiCC(batch):

    batch_size = len(batch)

    ts_array = []
    label_array = []
    snid_array = np.zeros((batch_size))

    lengths = np.zeros((batch_size), dtype=np.int32)
    static_features_tensor = torch.zeros((batch_size, n_static_features),  dtype=torch.float32)
    lc_plot_tensor = torch.zeros((batch_size, n_channels, img_height, img_width), dtype=torch.float32)

    for i, sample in enumerate(batch):

        ts_array.append(sample['ts'])
        label_array.append(sample['label'])

        snid_array[i] = sample['SNID']
        lengths[i] = sample['ts'].shape[0]
        static_features_tensor[i,:] = sample['static']

        if 'lc_plot' in sample.keys():
            lc_plot_tensor[i,:,:,:] = sample['lc_plot']

    lengths = torch.from_numpy(lengths)
    label_array = np.array(label_array)

    ts_tensor = pad_sequence(ts_array, batch_first=True, padding_value=flag_value)

    d = {
        'ts': ts_tensor,
        'static': static_features_tensor, 
        'length': lengths,
        'label': label_array,
        'SNID': snid_array,
    }

    if 'lc_plot' in sample.keys():
        d['lc_plot'] = lc_plot_tensor

    return d

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

if __name__=='__main__':
    # <--- Example usage of the dataset --->

    dataset = ELAsTiCC_LC_Dataset('data/ELAsTiCC/test.parquet', include_lc_plots=False, transform=truncate_light_curve_fractionally)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_ELAsTiCC)

    for batch in tqdm(dataloader):
        #print(batch['ts'])
        pass
        

