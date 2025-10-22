"""Custom dataset class for the simulated ZTF light curve dataset."""
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

from oracle.constants import ZTF_sims_to_Astrophysical_mappings
from oracle.custom_datasets.BTS import ZTF_passband_to_wavelengths, ZTF_wavelength_to_color

# Path to this file's directory
here = Path(__file__).resolve().parent

# Go up to the root, then into data/ and then get the parquet file
ZTF_sim_train_parquet_path = str(here.parent.parent.parent / "data" / 'ZTF_sims' / 'train.parquet')
ZTF_sim_test_parquet_path = str(here.parent.parent.parent / "data" / 'ZTF_sims' / 'test.parquet')
ZTF_sim_val_parquet_path = str(here.parent.parent.parent / "data" / 'ZTF_sims' / 'val.parquet')

# <----- constant for the dataset ----->

img_height = 256
img_width = 256
n_channels = 3

# <----- Hyperparameters for the model.....Unfortunately ----->
marker_style_detection = 'o'
marker_style_non_detection = '*'
marker_size = 50
linewidth = 0.75

flag_value = -9

# Flag values for missing data of static feature according to SNANA
missing_data_flags = [-9, -99, -999, -9999, 999]

time_independent_feature_list = ['MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR']
time_dependent_feature_list = ['MJD', 'FLUXCAL', 'FLUXCALERR', 'FLT', 'PHOTFLAG']
book_keeping_feature_list = ['SNID', 'ZTF_class']

n_static_features = len(time_independent_feature_list)
n_ts_features = len(time_dependent_feature_list)
n_book_keeping_features = len(book_keeping_feature_list)


class ZTF_SIM_LC_Dataset(torch.utils.data.Dataset):

    def __init__(self, parquet_file_path, max_n_per_class=None, include_lc_plots=False, transform=None):
        """
        Initializes a ZTF_SIM_LC_Dataset instance by loading a parquet file, selecting required features,
        and performing several data preparation steps including cleaning and optional sample limiting.

        Parameters:
            parquet_file_path (str): The file path to the parquet dataset file.
            max_n_per_class (Optional[int], optional): Maximum number of samples to include per class.
                If None, no sample limiting is applied. Defaults to None.
            include_lc_plots (bool, optional): Flag indicating whether to include light curve plots in the dataset.
                Defaults to False.
            transform (Optional[callable], optional): A transformation function to apply to dataset entries.
                Defaults to None.

        Side Effects:
            - Prints a message indicating the dataset being loaded.
            - Invokes methods to print dataset composition and perform cleanup.
            - Limits the dataset to a maximum number of samples per class if max_n_per_class is specified.

        Raises:
            Any exceptions raised during the parquet file read operation using pl.read_parquet.
        """

        super(ZTF_SIM_LC_Dataset, self).__init__()

        # Columns to be read from the parquet file
        self.columns = time_dependent_feature_list + time_independent_feature_list + book_keeping_feature_list

        self.parquet_file_path = parquet_file_path
        self.transform = transform
        self.include_lc_plots = include_lc_plots
        self.max_n_per_class = max_n_per_class

        print(f'Loading dataset from {self.parquet_file_path}\n')
        self.parquet_df = pl.read_parquet(self.parquet_file_path, columns=self.columns)
        self.columns_dtypes = self.parquet_df.schema

        self.print_dataset_composition()

        self.clean_up_dataset()

        if self.max_n_per_class != None:
            self.limit_max_samples_per_class()
               
    def __len__(self):
        """
        Returns the number of rows in the parquet DataFrame.

        Returns:
            int: The total number of samples in the dataset.
        """

        return self.parquet_df.shape[0]

    def __getitem__(self, index):

        row = self.parquet_df.row(index, named=True) 

        snid = row['SNID']
        astrophysical_class = row['class']

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

        # This operation is costly. Only do it if include_lc_plots stamps is true
        if self.include_lc_plots:
            light_curve_plot = self.get_lc_plots(time_series_data)
            dictionary['lc_plot'] = light_curve_plot
        
        return dictionary
    
    def print_dataset_composition(self):
        """
        Prints the composition of the dataset. It formats these values into 
        a pandas DataFrame and prints the resulting table to the console.

        Returns:
            None
        """
        
        print("Before transforms and mappings, the dataset contains...")
        classes, count = np.unique(self.parquet_df['ZTF_class'], return_counts=True)
        d = {
            'Class': classes,
            'Counts': count
        }
        print(pd.DataFrame(d).to_string(index=False))
    
    def clean_up_dataset(self):
        """
        Cleans and transforms the dataset stored in the object's parquet_df attribute.
        This method performs several dataset cleaning operations including:
        - Replacing band labels in the "FLT" column with their corresponding mean wavelengths
            using the mapping defined in ZTF_passband_to_wavelengths.
        - Removing saturated measurements from time series features:
            - For each time-dependent feature, it removes data points where the
              corresponding "PHOTFLAG" bitmask indicates saturation (using bitwise logic with 1024).
        - Replacing the "PHOTFLAG" bitmask values:
            - Converts the cleaned "PHOTFLAG" list to binary flags where any instance of the flag 4096
              is set to 1 (indicating detection) and 0 otherwise.
        - Adjusting the time series:
            - Normalizes the "MJD_clean" column by subtracting the time of the first observation from all entries.
        - Mapping simulation class labels:
            - Replaces ZTF simulation classes with astrophysical class labels using ZTF_sims_to_Astrophysical_mappings.
        - Handling missing data in time-independent features:
            - Replaces any feature value that matches a missing data flag with a specified flag_value.
        The method updates the parquet_df in place and prints progress messages for each step.
        """

        def remove_saturations_from_series(phot_flag_arr, feature_arr):
            
            saturation_mask =  (np.array(phot_flag_arr) & 1024) == 0 
            feature_arr = np.array(feature_arr)[saturation_mask].tolist()

            return feature_arr
        
        def replace_missing_flags(x):

            if x in missing_data_flags:
                return float(flag_value)
            else:
                return x
            
        print("Starting Dataset Transformations:")

        print("Replacing band labels with mean wavelengths...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("FLT").map_elements(lambda x: [ZTF_passband_to_wavelengths[band] for band in x], return_dtype=pl.List(pl.Float64)).alias("FLT")
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

        # Setting flag as 1 for detections and 0 for anything else
        print(f"Replacing PHOTFLAG bitmask with binary values...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("PHOTFLAG_clean").map_elements(lambda x: np.where(np.array(x) & 4096 != 0, 1, 0).tolist(), return_dtype=pl.List(pl.Int64)).alias("PHOTFLAG_clean")
        )

        print("Subtracting time of first observation...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("MJD_clean").map_elements(lambda x: (np.array(x) - min(x)).tolist(), return_dtype=pl.List(pl.Float64)).alias("MJD_clean")
        )

        print("Mapping ZTF sim classes to astrophysical classes...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("ZTF_class").replace(ZTF_sims_to_Astrophysical_mappings, return_dtype=pl.String).alias("class")
        )

        for feature in time_independent_feature_list:
            print(f"Replacing missing values in {feature} series...")
            self.parquet_df = self.parquet_df.with_columns(
                pl.col(feature).map_elements(lambda x: replace_missing_flags(x), return_dtype=self.columns_dtypes[feature]).alias(f"{feature}_clean")
            )
        print('Done!\n')

    def limit_max_samples_per_class(self):
        """
        Limits the number of samples per class in the dataset.

        For every unique class in the 'class' column of self.parquet_df, this method selects
        at most self.max_n_per_class entries (i.e., the first self.max_n_per_class samples) and
        concatenates them into a new dataframe that replaces self.parquet_df.

        Informative messages are printed to indicate:
          - The overall limit being applied per class.
          - The resulting number of samples retained for each class.
        """

        print(f"Limiting the number of samples to a maximum of {self.max_n_per_class} per class.")

        class_dfs = []
        unique_classes = np.unique(self.parquet_df['class'])

        for c in unique_classes:

            class_df = self.parquet_df.filter(pl.col("class") == c).slice(0, self.max_n_per_class)
            class_dfs.append(class_df)
            print(f"{c}: {class_df.shape[0]}")

        self.parquet_df = pl.concat(class_dfs)

    def get_lc_plots(self, x_ts):
        """
        Generates a light curve plot from the provided time series data and returns it as a PyTorch tensor.

        The function performs the following steps:
            - Extracts light curve parameters such as Julian dates, flux measurements, flux errors, filter identifiers, and photometric flags from the input array.
            - For each wavelength (as defined in 'ZTF_wavelength_to_color'), it plots:
            - Detected data points using the 'marker_style_detection'.
            - Non-detected data points using the 'marker_style_non_detection'. Both with error bars corresponding to flux uncertainties.
            - Overlays a line plot connecting all points for each wavelength.
            - Configures the matplotlib figure by setting a fixed size, removing axis ticks and all spines.
            - Saves the plot into an in-memory PNG buffer at a specified DPI, then loads it via PIL.
            - Converts the image to a NumPy array, permutes the dimensions, and finally converts it into a PyTorch tensor.

        Parameters:
            x_ts (np.ndarray): A 2D NumPy array representing the time series data.

        Returns:
            torch.Tensor: A tensor representation of the generated light curve plot image.
        """


        # Get the light curve data
        jd = x_ts[:,time_dependent_feature_list.index('MJD')] 
        flux = x_ts[:,time_dependent_feature_list.index('FLUXCAL')]
        flux_err =  x_ts[:,time_dependent_feature_list.index('FLUXCALERR')]
        filters =  x_ts[:,time_dependent_feature_list.index('FLT')]
        phot_flag = x_ts[:,time_dependent_feature_list.index('PHOTFLAG')] # NOTE: might want to use a different marker for ND

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
            detection_idx = np.where((filters == wavelength) & (phot_flag==1))[0]
            non_detection_idx = np.where((filters == wavelength) & (phot_flag==0))[0]

            ax.errorbar(jd[detection_idx], flux[detection_idx], yerr=flux_err[detection_idx], fmt=marker_style_detection, color=ZTF_wavelength_to_color[wavelength])
            ax.errorbar(jd[non_detection_idx], flux[non_detection_idx], yerr=flux_err[non_detection_idx], fmt=marker_style_non_detection, color=ZTF_wavelength_to_color[wavelength])
            ax.plot(jd[idx], flux[idx], linewidth=linewidth, color=ZTF_wavelength_to_color[wavelength])

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

    def get_all_labels(self):
        """
        Retrieve all labels from the dataset.

        This method extracts the 'class' column from the parquet dataframe attribute and 
        returns it as a list.

        Returns:
            list: A list containing all labels present in the 'class' column.
        """

        return self.parquet_df['class'].to_list()
    
def truncate_ZTF_SIM_light_curve_by_days_since_trigger(x_ts, d):
    """
    Truncates a ZTF SIM light curve by retaining only the observations within a specified number of days since the trigger (first detection).

    Parameters:
        x_ts (numpy.ndarray): A 2D array representing the time series data of the light curve. It is expected to have columns corresponding
                              to various features, including 'PHOTFLAG' and 'MJD', as specified in the global list 'time_dependent_feature_list'.
        d (float): The time window (in days) from the first detection. Observations beyond this period are removed.

    Returns:
        numpy.ndarray: The truncated light curve array containing only the observations within the specified days since the trigger.
    """

    # Get the first detection index
    photflags = x_ts[:,time_dependent_feature_list.index('PHOTFLAG')]
    first_detection_idx = np.where(photflags==1)[0][0]

    # Get the days data
    mjd_index = time_dependent_feature_list.index('MJD')
    jd = x_ts[:,mjd_index]

    # Get the days since first detection
    days_since_first_detection = jd - jd[first_detection_idx]

    # Get indices of observations within d days of the first detection (trigger)
    idx = np.where(days_since_first_detection < d)[0]

    # Truncate the light curve
    x_ts = x_ts[idx, :]

    return x_ts

def truncate_ZTF_SIM_light_curve_fractionally(x_ts, f=None):
    """
    Truncate a ZTF simulation light curve by retaining only a fraction of its observations.

    Parameters:
        x_ts (numpy.ndarray): A 2D array representing the light curve where each row corresponds to an observation.
        f (float, optional): Fraction of the total observations to retain. If not provided (None),
            a random fraction between 0.1 and 1.0 will be used.
    Returns:
        numpy.ndarray: A truncated version of the input light curve containing a fraction (at least one) of the original observations.
    """

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

def custom_collate_ZTF_SIM(batch):
    """
    Collates a batch of ZTF simulation samples into a single dictionary of tensors suitable for model input.

    Each sample in the input batch is expected to be a dictionary with the following keys:
        - 'ts': A tensor representing the time series data. The tensor should have shape (num_time_points, ...).
        - 'label': The label associated with the sample (e.g., class index or regression target).
        - 'SNID': A unique identifier for the sample.
        - 'static': A tensor of static features with a predefined number of features (n_static_features).
        - 'lc_plot': (Optional) A tensor representing the light curve plot image with shape (n_channels, img_height, img_width).

    Parameters:
        batch (list): A list of sample dictionaries as described above.

    Returns:
        dict: A dictionary containing the collated batch with the same keys as the input samples, where:
    """

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
    """
    Display a grid of images with corresponding labels.
    This function creates a visual representation of the first n images from the provided dataset.
    It arranges the images in a square grid and annotates each image with its corresponding label.

    Parameters:
        images (Tensor or array-like): Collection of images to be displayed. Each image is expected to have the shape
            (C, H, W), where C is the number of channels. For grayscale images, C should be 1.
        labels (Sequence): Sequence of labels corresponding to each image.
        n (int, optional): The number of images to display. The function uses the first n images from the collection.
            Defaults to 16.

    Displays:
        A matplotlib figure containing a grid of images, each annotated with its respective label.
    """

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

    dataset = ZTF_SIM_LC_Dataset(ZTF_sim_train_parquet_path, include_lc_plots=False, transform=truncate_ZTF_SIM_light_curve_fractionally, max_n_per_class=20000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_ZTF_SIM)

    for batch in tqdm(dataloader):

        break

        print(batch['label'])

        for k in (batch.keys()):
            print(f"{k}: \t{batch[k].shape}")
        
        if 'lc_plot' in batch.keys():
            show_batch(batch['lc_plot'], batch['label'])
    
    imgs = []
    lc_d = []
    days = np.linspace(10,100,16)
    for d in days:

        k = 4
        
        transform = partial(truncate_ZTF_SIM_light_curve_by_days_since_trigger, d=d)
        dataset = ZTF_SIM_LC_Dataset(ZTF_sim_test_parquet_path, include_lc_plots=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=custom_collate_ZTF_SIM)
        for batch in tqdm(dataloader):
            imgs.append(batch['lc_plot'][k,:,:,:])
            lc_d.append(max(batch['ts'][k,:,0]))
            break
    
    plt.scatter(days, lc_d)
    plt.show()
    show_batch(imgs, batch['label'])