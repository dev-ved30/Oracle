import io
import torch

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from oracle.constants import ztf_filters, ztf_alert_image_order, ztf_alert_image_dimension, ztf_filter_to_fid, BTS_to_Astrophysical_mappings

# Path to this file's directory
here = Path(__file__).resolve().parent

# Go up to the root, then into data/ and then get the parquet file
BTS_train_parquet_path = str(here.parent.parent.parent / "data" / 'BTS' / 'train.parquet')
BTS_test_parquet_path = str(here.parent.parent.parent / "data" / 'BTS' / 'test.parquet')
BTS_val_parquet_path = str(here.parent.parent.parent / "data" / 'BTS' / 'val.parquet')

# <----- constant for the dataset ----->

img_height = 256
img_width = 256
n_channels = 3

# <----- Hyperparameters for the model.....Unfortunately ----->
marker_style = '-o'
marker_size = 50
linewidth = 0.75

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

ZTF_wavelength_to_color = {
    ZTF_fid_to_wavelengths[1]: np.array((255, 0, 0))/255,
    ZTF_fid_to_wavelengths[2]: np.array((0, 255, 0))/255,
    ZTF_fid_to_wavelengths[3]: np.array((0, 0, 255))/255,
}

flag_value = -9

images_list = ['g_reference', 'g_science', 'g_difference', 'r_reference', 'r_science', 'r_difference', 'i_reference', 'i_science', 'i_difference']
time_dependent_feature_list = ['jd', 'magpsf', 'sigmapsf', 'fid']
book_keeping_feature_list = ['ZTFID', 'bts_class']

n_images = len(images_list)
n_ts_features = len(time_dependent_feature_list)
n_book_keeping_features = len(book_keeping_feature_list)

class BTS_LC_Dataset(torch.utils.data.Dataset):

    def __init__(self, parquet_file_path, include_postage_stamps=False, include_lc_plots=False, transform=None):
        super(BTS_LC_Dataset, self).__init__()

        # Columns to be read from the parquet file
        self.parquet_file_path = parquet_file_path
        self.transform = transform
        self.include_lc_plots = include_lc_plots
        self.include_postage_stamps = include_postage_stamps

        print(f'Loading dataset from {self.parquet_file_path}\n')
        self.columns = time_dependent_feature_list + images_list + book_keeping_feature_list
        self.parquet_df = pl.read_parquet(self.parquet_file_path, columns=self.columns)
        self.columns_dtypes = self.parquet_df.schema

        self.clean_up_dataset()
               
    def __len__(self):

        return self.parquet_df.shape[0]

    def __getitem__(self, index):

        row = self.parquet_df.row(index, named=True) 

        ztfid = row['ZTFID']
        BTS_class = row['bts_class']
        astrophysical_class = BTS_to_Astrophysical_mappings[BTS_class]

        lc_length = len(row['jd'])

        time_series_data = np.zeros((lc_length, n_ts_features), dtype=np.float32)
        for i, feature in enumerate(time_dependent_feature_list):
            time_series_data[:,i] = np.array(row[feature], dtype=np.float32)
        time_series_data = torch.from_numpy(time_series_data)

        if self.transform != None:
            time_series_data = self.transform(time_series_data)

        dictionary = {
            'ts': time_series_data,
            'label': astrophysical_class,
            'ZTFID': ztfid,
        }

        # This operation is costly. Only do it if include_postage stamps is true
        if self.include_postage_stamps:

            postage_stamps = {}
            for f in ztf_filters:
                for img_type in ztf_alert_image_order:

                    img_data = row[f"{f}_{img_type}"]

                    if img_data == None:
                        postage_stamps[f"{f}_{img_type}"] = np.zeros(ztf_alert_image_dimension)
                    else:
                        postage_stamps[f"{f}_{img_type}"] = np.reshape(img_data, ztf_alert_image_dimension)
            postage_stamps = self.get_postage_stamp_plot(postage_stamps)
            dictionary['postage_stamp'] = postage_stamps

        # This operation is costly. Only do it if include_lc_plots stamps is true
        if self.include_lc_plots:
            light_curve_plot = self.get_lc_plots(time_series_data)
            dictionary['lc_plot'] = light_curve_plot
        
        return dictionary
    
    def clean_up_dataset(self):

        print("Starting Dataset Transformations:")
            
        # Subtract out time of first obs
        print("Subtracting time of first observation...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("jd").map_elements(lambda x: (np.array(x) - min(x)).tolist(), return_dtype=pl.List(pl.Float64)).alias("jd")
        )

        # Map pass bands to wavelengths
        print("Replacing band labels with mean wavelengths...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("fid").map_elements(lambda x: [ZTF_fid_to_wavelengths[band] for band in x], return_dtype=pl.List(pl.Float64)).alias("fid")
        )

        print('Done!\n')

    def get_lc_plots(self, x_ts):

        # Get the light curve data
        jd = x_ts[:,time_dependent_feature_list.index('jd')]
        flux = x_ts[:,time_dependent_feature_list.index('magpsf')]
        flux_err = x_ts[:,time_dependent_feature_list.index('sigmapsf')]
        filters = x_ts[:,time_dependent_feature_list.index('fid')]

        # Create a figure and axes
        fig, ax = plt.subplots(1, 1)

        # Set the figure size in inches
        width_in = 2.56  # Desired width in inches (256 pixels / 100 dpi)
        height_in = 2.56  # Desired height in inches (256 pixels / 100 dpi)

        fig.set_size_inches(width_in, height_in)

        # Remove all spines (black frame)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for wavelength in ZTF_wavelength_to_color.keys():
            
            idx = np.where(filters == wavelength)[0]
            ax.errorbar(jd[idx], flux[idx], yerr=flux_err[idx], linewidth=linewidth, fmt=marker_style, color=ZTF_wavelength_to_color[wavelength])

        # Save the figure as PNG with the desired DPI
        dpi = 100  # Dots per inch (adjust as needed)

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
        img_arr = torch.from_numpy(img_arr)

        # Close the buffer
        buf.close()

        return img_arr
    
    def get_postage_stamp_plot(self, examples):

        img_length = ztf_alert_image_dimension[0]


        canvas = np.zeros((img_length, len(ztf_filters)*img_length)) 

        # Loop through all of the filters
        for j, f in enumerate(ztf_filters):
            canvas[:, j*img_length:(j+1)*img_length] = examples[f"{f}_reference"]

        # Create a figure and axes
        fig, ax = plt.subplots(1, 1)

        # Remove all spines (black frame)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Save the figure as PNG with the desired DPI
        dpi = 100  # Dots per inch (adjust as needed)

        # Set the figure size in inches
        width_in = 2.56  # Desired width in inches (256 pixels / 100 dpi)
        height_in = 2.56  # Desired height in inches (256 pixels / 100 dpi)

        fig.set_size_inches(width_in, height_in)

        ax.imshow(canvas)

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
        im = np.array(im, dtype=np.float32)
        im = np.permute_dims(im, (2, 0, 1))
        im = torch.from_numpy(im)
        
        # Close the buffer
        buf.close()
        
        return im
    
def truncate_BTS_light_curve_fractionally(x_ts, f=None):

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

def truncate_BTS_light_curve_by_days_since_trigger(x_ts, d):

    # NOTE: For BTS we are making the assumption that the data set does not contain any non detections. This is not the case with ELAsTiCC

    # Get the days data
    jd_index = time_dependent_feature_list.index('jd')
    jd = x_ts[:, jd_index]

    # Get indices of observations within d days of the first detection (trigger)
    idx = np.where(jd < d)[0]

    # Truncate the light curve
    x_ts = x_ts[idx, :]

    return x_ts

def custom_collate_BTS(batch):

    batch_size = len(batch)

    ts_array = []
    label_array = []
    ztfid_array = []

    lengths = np.zeros((batch_size), dtype=np.float32)
    lc_plot_tensor = torch.zeros((batch_size, n_channels, img_height, img_width), dtype=torch.float32)
    postage_stamps_tensor = torch.zeros((batch_size, n_channels, img_height, img_width), dtype=torch.float32)

    for i, sample in enumerate(batch):

        ts_array.append(sample['ts'])
        label_array.append(sample['label'])
        ztfid_array.append(sample['ZTFID'])

        lengths[i] = sample['ts'].shape[0]

        if 'postage_stamp' in sample.keys():
            postage_stamps_tensor[i,:,:,:] = sample['postage_stamp']        

        if 'lc_plot' in sample.keys():
            lc_plot_tensor[i,:,:,:] = sample['lc_plot']

    lengths = torch.from_numpy(lengths)
    label_array = np.array(label_array)
    ztfid_array = np.array(ztfid_array)

    ts_tensor = pad_sequence(ts_array, batch_first=True, padding_value=flag_value)

    d = {
        'ts': ts_tensor,
        'length': lengths,
        'label': label_array,
        'ZTFID': ztfid_array,
    }

    if 'postage_stamp' in sample.keys():
        d['postage_stamp'] = postage_stamps_tensor

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

    dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_postage_stamps=True, include_lc_plots=True, transform=truncate_BTS_light_curve_fractionally)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=custom_collate_BTS)

    for batch in tqdm(dataloader):

        pass

        for k in (batch.keys()):
            print(f"{k}: \t{batch[k].shape}")

        if 'postage_stamp' in batch.keys():
            show_batch(batch['postage_stamp'], batch['label'])
        
        if 'lc_plot' in batch.keys():
            show_batch(batch['lc_plot'], batch['label'])

    # imgs = []
    # lc_d = []
    # days = np.linspace(10,100,16)
    # for d in days:

    #     k = 1
        
    #     transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
    #     dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_postage_stamps=True, include_lc_plots=True, transform=transform)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=custom_collate_BTS)
    #     for batch in tqdm(dataloader):
    #         imgs.append(batch['lc_plot'][k,:,:,:])
    #         lc_d.append(max(batch['ts'][k,:,0]))
    #         break
    
    # plt.scatter(days, lc_d)
    # plt.show()
    # show_batch(imgs, batch['label'])