import io
import torch

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from functools import partial
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from astropy.coordinates import SkyCoord

from oracle.constants import ztf_filters, ztf_alert_image_order, ztf_alert_image_dimension, ztf_filter_to_fid, BTS_to_Astrophysical_mappings

# Path to this file's directory
here = Path(__file__).resolve().parent

# Go up to the root, then into data/ and then get the parquet file
BTS_train_parquet_path = str(here.parent.parent.parent / "data" / 'BTS_new' / 'train.parquet')
BTS_test_parquet_path = str(here.parent.parent.parent / "data" / 'BTS_new' / 'test.parquet')
BTS_val_parquet_path = str(here.parent.parent.parent / "data" / 'BTS_new' / 'val.parquet')

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
time_independent_feature_list = ['l', 'b', 'W1mag', 'W2mag', 'W3mag', 'W4mag','W1_minus_W3','W2_minus_W3']
meta_data_feature_list = ['sky', 'sgscore1', 'sgscore2', 'distpsnr1', 'distpsnr2', 'fwhm', 'diffmaglim', 'ndethist', 'nmtchps', 'drb', 'ncovhist', 'chinr', 'sharpnr', 'scorr', 'sgmag1', 'srmag1','simag1','szmag1', 'sgmag2', 'srmag2','simag2','szmag2']
book_keeping_feature_list = ['ZTFID', 'bts_class']
features_to_transform = ['magpsf', 'sigmapsf']

n_images = len(images_list)
n_ts_features = len(time_dependent_feature_list)
n_meta_features = len(meta_data_feature_list)
n_static_features = len(time_independent_feature_list)
n_book_keeping_features = len(book_keeping_feature_list)

class BTS_LC_Dataset(torch.utils.data.Dataset):

    def __init__(self, 
                 parquet_file_path, 
                 mapper=None,
                 max_n_per_class=None, 
                 include_postage_stamps=False, 
                 include_lc_plots=False, 
                 transform=None, 
                 over_sample=False, 
                 excluded_classes=[]):
        super(BTS_LC_Dataset, self).__init__()

        self.parquet_file_path = parquet_file_path
        self.transform = transform
        self.include_lc_plots = include_lc_plots
        self.include_postage_stamps = include_postage_stamps
        self.max_n_per_class = max_n_per_class
        self.over_sample = over_sample
        self.excluded_classes = excluded_classes

        if mapper == None:
            self.mapper = BTS_to_Astrophysical_mappings
        else:
            self.mapper = mapper

        print(f'Loading dataset from {self.parquet_file_path}\n')
        self.parquet_df = pl.read_parquet(self.parquet_file_path)

        # Columns to be read from the parquet file
        self.columns_dtypes = self.parquet_df.schema

        self.print_dataset_composition()
        self.convert_mags_to_flux()
        self.clean_up_dataset()
        self.exclude_classes()

        if self.max_n_per_class != None:
            self.limit_max_samples_per_class()

        if self.over_sample:
            self.over_sample_minority_classes()

               
    def __len__(self):

        return self.parquet_df.shape[0]

    def __getitem__(self, index):

        row = self.parquet_df.row(index, named=True) 

        ztfid = row['ZTFID']
        astrophysical_class = row['class']
        bts_class = row['bts_class']

        lc_length = len(row['jd'])

        # Adding extra row for photflag to maintain compatibility. This is always one if we are just dealing with detections
        time_series_data = np.ones((lc_length, n_ts_features+1), dtype=np.float32)
        for i, feature in enumerate(time_dependent_feature_list):
            time_series_data[:,i] = np.array(row[feature], dtype=np.float32)
        time_series_data = torch.from_numpy(time_series_data)

        meta_data = np.ones((lc_length, n_meta_features), dtype=np.float32)
        for i, feature in enumerate(meta_data_feature_list):
            d = np.asarray(row[feature], dtype=np.float32)
            meta_data[: ,i] = d
        meta_data = np.where(np.isnan(meta_data), np.float32(flag_value), meta_data)
        meta_data = np.where(meta_data==-999, np.float32(flag_value), meta_data)
        meta_data = torch.from_numpy(meta_data)

        static_features = np.ones(n_static_features, dtype=np.float32)
        for i, feature in enumerate(time_independent_feature_list):
            d = row[feature]
            static_features[i] = d
        static_features = np.where(np.isnan(static_features), np.float32(flag_value), static_features)
        static_features = torch.from_numpy(static_features)

        if self.transform != None:
            time_series_data, meta_data = self.transform(time_series_data, meta_data)

        dictionary = {
            'ts': time_series_data,
            'meta': meta_data,
            'static': static_features,
            'label': astrophysical_class,
            'bts_class': bts_class,
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
            postage_stamps = self.get_postage_stamp(postage_stamps)
            dictionary['postage_stamp'] = postage_stamps

        # This operation is costly. Only do it if include_lc_plots stamps is true
        if self.include_lc_plots:
            light_curve_plot = self.get_lc_plots(time_series_data)
            dictionary['lc_plot'] = light_curve_plot
        
        return dictionary
    
    def exclude_classes(self):

        print(f"Excluding {self.excluded_classes} from the dataset...")

        class_dfs = []
        unique_classes = np.unique(self.parquet_df['class'])

        for c in unique_classes:

            if c not in self.excluded_classes:
                class_df = self.parquet_df.filter(pl.col("class") == c)
                class_dfs.append(class_df)

        self.parquet_df = pl.concat(class_dfs)

    def print_dataset_composition(self):
        
        print("Before transforms and mappings, the dataset contains...")
        classes, count = np.unique(self.parquet_df['bts_class'], return_counts=True)
        d = {
            'Class': classes,
            'Counts': count
        }
        print(pd.DataFrame(d).to_string())

    def convert_mags_to_flux(self):

        F0_uJy = 3631.0 * 1e6
        factor  = 0.4 * np.log(10)

        self.parquet_df = self.parquet_df.with_columns([
            # flux only needs magpsf
            pl.col("magpsf")
            .map_elements(
                lambda mags: (F0_uJy * 10 ** (-0.4 * np.array(mags))).tolist(),
                return_dtype=pl.List(pl.Float64),
            )
            .alias("flux"),

            # flux_err needs both magpsf and sigmapsf
            pl.struct(["magpsf", "sigmapsf"])
            .map_elements(
                lambda row: (
                factor
                * F0_uJy
                * 10 ** (-0.4 * np.array(row["magpsf"]))
                * np.array(row["sigmapsf"])
                ).tolist(),
                return_dtype=pl.List(pl.Float64),
            )
            .alias("flux_err"),
        ])

    
    def get_all_labels(self):

        return self.parquet_df['class'].to_list()
    
    def over_sample_minority_classes(self):
        
        print("Oversampling minority classes...")
        unique_classes, unique_counts = np.unique(self.parquet_df['class'], return_counts=True)
        max_count = max(unique_counts)
        class_dfs = []

        for i, c in enumerate(unique_classes):

            if unique_counts[i] < max_count:
                class_df = self.parquet_df.filter(pl.col("class") == c).sample(n=max_count, with_replacement=True)
                class_dfs.append(class_df)
                print(f"{c}: {class_df.shape[0]}")

        self.parquet_df = pl.concat(class_dfs)

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

        print("Mapping BTS samples explorer classes to astrophysical classes...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("bts_class").replace(self.mapper, return_dtype=pl.String).alias("class")
        )

        print("Adding wise colors...")
        self.parquet_df = self.parquet_df.with_columns([
            (pl.col("W1mag") - pl.col("W3mag")).alias("W1_minus_W3"),
            (pl.col("W2mag") - pl.col("W3mag")).alias("W2_minus_W3"),
        ])

        print("Converting RA and Dec to l and b...")
        self.parquet_df = self.parquet_df.with_columns([

            pl.struct(["ra", "dec"])
            .map_elements(
                lambda row: SkyCoord(ra=row['ra'][0], dec=row['dec'][0], unit=(u.deg, u.deg)).galactic.l.deg,
                return_dtype=pl.Float64,
            )
            .alias("l"),

            pl.struct(["ra", "dec"])
            .map_elements(
                lambda row: SkyCoord(ra=row['ra'][0], dec=row['dec'][0], unit=(u.deg, u.deg)).galactic.b.deg,
                return_dtype=pl.Float64,
            )
            .alias("b"),
        ])

        print('Done!\n')

    def limit_max_samples_per_class(self):

        print(f"Limiting the number of samples to a maximum of {self.max_n_per_class} per class.")

        class_dfs = []
        unique_classes = np.unique(self.parquet_df['class'])

        for c in unique_classes:

            class_df = self.parquet_df.filter(pl.col("class") == c).slice(0, self.max_n_per_class)
            class_dfs.append(class_df)
            print(f"{c}: {class_df.shape[0]}")

        self.parquet_df = pl.concat(class_dfs)

    def get_lc_plots(self, x_ts):

        # Get the light curve data
        jd = x_ts[:,time_dependent_feature_list.index('jd')]
        flux = x_ts[:,time_dependent_feature_list.index('flux')]
        flux_err = x_ts[:,time_dependent_feature_list.index('flux_err')]
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
        #plt.gca().invert_yaxis()

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
    
    def get_postage_stamp(self, examples):

        img_length = ztf_alert_image_dimension[0]
        canvas = np.zeros((len(ztf_filters), img_length, img_length)) 

        # Loop through all of the filters
        for j, f in enumerate(ztf_filters):
            canvas[j,:,:] = examples[f"{f}_reference"]
        canvas = torch.from_numpy(canvas*255)
        return canvas

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
    
def truncate_BTS_light_curve_fractionally(x_ts, x_static, f=None, normalize_flux=False):

    if f == None:
        # Get a random fraction between 0.1 and 1
        f = np.random.uniform(0.01, 1.0)
    
    original_obs_count = x_ts.shape[0]

    # Find the new length of the light curve
    new_obs_count = int(original_obs_count * f)
    if new_obs_count < 1:
        new_obs_count = 1

    # Truncate the light curve
    x_ts = x_ts[:new_obs_count, :]
    x_static = x_static[:new_obs_count, :]

    # Normalize the time series
    if normalize_flux:

        flux_index = time_dependent_feature_list.index('flux')
        mean = torch.mean(x_ts[:, flux_index])
        std = torch.std(x_ts[:, flux_index])
        
        if std > 0: 
            x_ts[:, flux_index] = (x_ts[:, flux_index] - mean) / std
        else:
            x_ts[:, flux_index] = (x_ts[:, flux_index] - mean)

    return x_ts, x_static

def truncate_BTS_light_curve_by_days_since_trigger(x_ts, x_static, d=None, add_jitter=False, normalize_flux=False):

    # NOTE: For BTS we are making the assumption that the data set does not contain any non detections. This is not the case with ELAsTiCC
    if d == None:
        d = 2**np.random.uniform(0, 11)

    # Get the days data
    jd_index = time_dependent_feature_list.index('jd')
    flux_index = time_dependent_feature_list.index('magpsf')
    flux_err_index = time_dependent_feature_list.index('sigmapsf')

    jd = x_ts[:, jd_index]

    if add_jitter:

        x_ts[:, flux_index] = x_ts[:, flux_index] + np.random.normal([0]*x_ts.shape[0],  x_ts[:, flux_err_index])
        x_ts[:, flux_err_index] = x_ts[:, flux_err_index] + np.random.normal([0]*x_ts.shape[0],  0.01)

    # Get indices of observations within d days of the first detection (trigger)
    idx = np.where(jd <= d)[0]

    # Truncate the light curve
    x_ts = x_ts[idx, :]
    x_static = x_static[idx, :]

    # Normalize the time series
    if normalize_flux:

        mean = torch.mean(x_ts[:, flux_index])
        std = torch.std(x_ts[:, flux_index])
        
        if std > 0: 
            x_ts[:, flux_index] = (x_ts[:, flux_index] - mean) / std
        else:
            x_ts[:, flux_index] = (x_ts[:, flux_index] - mean)

    return x_ts, x_static

def custom_collate_BTS(batch):

    batch_size = len(batch)

    ts_array = []
    label_array = []
    ztfid_array = []
    bts_array = []

    lengths = np.zeros((batch_size), dtype=np.float32)
    meta_features_tensor = torch.zeros((batch_size, n_meta_features),  dtype=torch.float32)
    static_features_tensor = torch.zeros((batch_size, n_static_features),  dtype=torch.float32)
    lc_plot_tensor = torch.zeros((batch_size, n_channels, img_height, img_height), dtype=torch.float32)
    postage_stamps_tensor = torch.zeros((batch_size, n_channels, 63, 63), dtype=torch.float32)

    for i, sample in enumerate(batch):

        ts_array.append(sample['ts'])
        label_array.append(sample['label'])
        ztfid_array.append(sample['ZTFID'])
        bts_array.append(sample['bts_class'])
        
        lengths[i] = sample['ts'].shape[0]
        meta_features_tensor[i, :] = sample['meta'][-1, :]
        static_features_tensor[i, :] = sample['static']

        if 'postage_stamp' in sample.keys():
            postage_stamps_tensor[i,:,:,:] = sample['postage_stamp']        

        if 'lc_plot' in sample.keys():
            lc_plot_tensor[i,:,:,:] = sample['lc_plot']

    lengths = torch.from_numpy(lengths)
    label_array = np.array(label_array)
    bts_array = np.array(bts_array)
    ztfid_array = np.array(ztfid_array)

    ts_tensor = pad_sequence(ts_array, batch_first=True, padding_value=flag_value)
    
    # Combine the static and meta features
    static_features_tensor = torch.cat((static_features_tensor, meta_features_tensor), dim=1)

    d = {
        'ts': ts_tensor,
        'static': static_features_tensor,
        'length': lengths,
        'label': label_array,
        'bts_class': bts_array, 
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

    dataset = BTS_LC_Dataset(BTS_train_parquet_path, include_postage_stamps=False, include_lc_plots=False, transform=truncate_BTS_light_curve_fractionally, max_n_per_class=1000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=custom_collate_BTS, shuffle=True)

    for batch in tqdm(dataloader):

        pass

        for k in (batch.keys()):
            print(f"{k}: \t{batch[k].shape}")

        if 'postage_stamp' in batch.keys():
            show_batch(batch['postage_stamp'], batch['label'])
        
        # if 'lc_plot' in batch.keys():
        #     show_batch(batch['lc_plot'], batch['label'])

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