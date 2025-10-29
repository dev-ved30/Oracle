"""Custom dataset class for the ELAsTiCC light curve dataset for LSST."""
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

from oracle.constants import ELAsTiCC_to_Astrophysical_mappings

# Path to this file's directory
here = Path(__file__).resolve().parent

# Go up to the root, then into data/ and then get the parquet file
ELAsTiCC_train_parquet_path = str(here.parent.parent.parent / "data" / 'ELAsTiCC' / 'train.parquet')
ELAsTiCC_test_parquet_path = str(here.parent.parent.parent / "data" / 'ELAsTiCC' / 'test.parquet')
ELAsTiCC_val_parquet_path = str(here.parent.parent.parent / "data" / 'ELAsTiCC' / 'val.parquet')

# <----- constant for the dataset ----->

img_height = 256
img_width = 256
n_channels = 3

# <----- Hyperparameters for the model.....Unfortunately ----->
marker_style_detection = 'o'
marker_style_non_detection = '*'
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

flag_value = -9

# Flag values for missing data of static feature according to elasticc
missing_data_flags = [-9, -99, -999, -9999, 999]

time_independent_feature_list = ['MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y']
time_dependent_feature_list = ['MJD', 'FLUXCAL', 'FLUXCALERR', 'BAND', 'PHOTFLAG']
book_keeping_feature_list = ['SNID', 'ELASTICC_class']

n_static_features = len(time_independent_feature_list)
n_ts_features = len(time_dependent_feature_list)
n_book_keeping_features = len(book_keeping_feature_list)


class ELAsTiCC_LC_Dataset(torch.utils.data.Dataset):

    def __init__(self, 
                 parquet_file_path, 
                 max_n_per_class=None, 
                 include_lc_plots=False, 
                 transform=None,
                 excluded_classes=[]):
        """
        Initialize an instance of the ELAsTiCC_LC_Dataset.
        This initializer loads a parquet dataset from the specified file path, sets up the columns to be read, and performs
        various preprocessing steps including mapping models to classes, limiting the number of samples per class (if specified),
        cleaning up the dataset, and excluding specific classes.

        Parameters:
            parquet_file_path (str): Path to the parquet file containing the dataset.
            max_n_per_class (int, optional): Maximum number of samples allowed per class. If None, no limit is enforced.
            include_lc_plots (bool, optional): Flag indicating whether to include light curve plots as part of the dataset.
            transform (callable, optional): An optional transform to be applied to the data samples.
            excluded_classes (list, optional): List of classes that should be excluded from the dataset.
            
        Note:
            - The initializer prints a message indicating the path from which the dataset is being loaded.
            - Subsequent methods called within this initializer handle additional dataset modifications such as mapping models
                to classes, limiting samples, cleaning up data, and excluding unwanted classes.
        """
        super(ELAsTiCC_LC_Dataset, self).__init__()

        # Columns to be read from the parquet file
        self.columns = time_dependent_feature_list + time_independent_feature_list + book_keeping_feature_list

        self.parquet_file_path = parquet_file_path
        self.transform = transform
        self.include_lc_plots = include_lc_plots
        self.max_n_per_class = max_n_per_class
        self.excluded_classes = excluded_classes

        print(f'Loading dataset from {self.parquet_file_path}\n')
        self.parquet_df = pl.read_parquet(self.parquet_file_path, columns=self.columns)
        self.columns_dtypes = self.parquet_df.schema
        self.map_models_to_classes()

        if self.max_n_per_class != None:
            self.limit_max_samples_per_class()

        self.clean_up_dataset()
        self.exclude_classes()
               
    def __len__(self):
        """
        Return the number of entries in the dataset.
        This method returns the length of the underlying DataFrame by utilizing its shape attribute. It specifically retrieves the number of rows, which represents the total number of entries in the dataset.
        Returns:
            int: The total number of rows in the parquet DataFrame.
        """

        return self.parquet_df.shape[0]

    def __getitem__(self, index):
        """
        Retrieves a single data sample from the dataset with the specified index.
        Parameters:
            index (int): The index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the following keys:
                - 'ts' (torch.Tensor): Tensor representing the time series data with shape (lc_length, n_ts_features).
                - 'static' (torch.Tensor): Tensor of static features with shape (n_static_features,).
                - 'label': The astrophysical class label extracted from the row.
                - 'ELASTICC_class': The ELASTICC-specific class label from the row.
                - 'SNID': The SNID identifier from the row.
                - 'lc_plot' (optional): The light curve plot if `include_lc_plots` is True; otherwise, this key is omitted.
        """

        row = self.parquet_df.row(index, named=True) 
        
        ELASTICC_class = row['ELASTICC_class']
        snid = row['SNID']
        astrophysical_class = row['class']

        lc_length = len(row['MJD_clean'])

        time_series_data = np.zeros((lc_length, n_ts_features), dtype=np.float32)
        for i, feature in enumerate(time_dependent_feature_list):
            time_series_data[:,i] = np.array(row[f"{feature}_clean"], dtype=np.float32)
        time_series_data = torch.from_numpy(time_series_data)

        # create on CPU via numpy then convert to torch on CPU
        static_np = np.zeros((n_static_features,), dtype=np.float32)
        for i, feature in enumerate(time_independent_feature_list):
            static_np[i] = float(row[feature]) if row[feature] is not None else float(flag_value)
        static_data = torch.from_numpy(static_np)   # CPU tensor

        if self.transform != None:
            time_series_data = self.transform(time_series_data)

        dictionary = {
            'ts': time_series_data,
            'static': static_data,
            'label': astrophysical_class,
            'ELASTICC_class': ELASTICC_class,
            'SNID': snid,
        }

        # This operation is costly. Only do it if include_lc_plots stamps is true
        if self.include_lc_plots:
            light_curve_plot = self.get_lc_plots(time_series_data)
            dictionary['lc_plot'] = light_curve_plot
        
        return dictionary
    
    def map_models_to_classes(self):
        """
        Maps ELAsTiCC classes to astrophysical classes by replacing the values in the 
        'ELASTICC_class' column of the DataFrame with the corresponding astrophysical 
        class names. The transformation is done in-place on self.parquet_df by creating 
        a new column named 'class'.

        Returns:
            None

        Side Effects:
            - Modifies self.parquet_df by adding/updating the 'class' column.
        """


        print("Mapping ELAsTiCC classes to astrophysical classes...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("ELASTICC_class").replace(ELAsTiCC_to_Astrophysical_mappings, return_dtype=pl.String).alias("class")
        )

    def exclude_classes(self):
        """
        Exclude classes specified in self.excluded_classes from the dataset.
        This method filters the records in self.parquet_df, removing any rows where the "class"
        column matches an entry in self.excluded_classes. It concatenates the remaining dataframes
        per class and updates self.parquet_df with the result.

        Returns:
            None

        Side Effects:
            - Modifies self.parquet_df by excluding rows of unwanted classes.
        """

        print(f"Excluding {self.excluded_classes} from the dataset...")

        class_dfs = []
        unique_classes = np.unique(self.parquet_df['class'])

        for c in unique_classes:

            if c not in self.excluded_classes:
                class_df = self.parquet_df.filter(pl.col("class") == c)
                class_dfs.append(class_df)

        self.parquet_df = pl.concat(class_dfs)
    
    def clean_up_dataset(self):
        """
        Cleans and transforms the dataset contained in self.parquet_df by performing a series of operations:
            - Replaces band labels with their corresponding mean wavelengths using a predefined mapping.
            - Removes saturation-affected data points from time-dependent feature series (excluding PHOTFLAG) by:
                - Filtering the data using a bitmask to remove values with saturation (determined by the presence of a specific bit in the photometric flag).
            - Processes the PHOTFLAG series by:
                - Removing data points flagged as saturated.
                - Converting the remaining bitmask values into binary flags (1 for detections based on a specific bit and 0 otherwise).
            - Normalizes the MJD (Modified Julian Date) values by subtracting the time of the first observation, effectively realigning the time series.
            - Replaces missing values in time-independent features with a predetermined flag value.
        
        Returns:
            None
        
        Note:
            - This method defines two helper functions locally:
                - remove_saturations_from_series: Filters a given series based on the photometric flag to remove saturations.
                - replace_missing_flags: Substitutes missing data flags with a specified flag value.
        """

        def remove_saturations_from_series(phot_flag_arr, feature_arr):
            """
            Remove saturated entries from a series of features based on photometric flags.

            Parameters:
                phot_flag_arr (list or array-like): An array of photometric flag values for each observation.
                    Saturation is identified by applying a bitwise AND with 1024.
                feature_arr (list or array-like): An array of features corresponding to the photometric flags.
                    The function returns a filtered list containing only the features where the 
                    observations are not saturated.

            Returns:
                list: A filtered array of features with saturated entries removed.
            """
            saturation_mask =  (np.array(phot_flag_arr) & 1024) == 0 
            feature_arr = np.array(feature_arr)[saturation_mask].tolist()

            return feature_arr
        
        def replace_missing_flags(x):
            """
            Replaces missing data flags in the input value.

            Parameters:
                x: Any
                    The value to be checked against the predefined missing data flags.

            Returns:
                float or any:
                    Returns the flag value (as a float) if 'x' is found in the missing data flags; otherwise, returns 'x' unchanged.
            """
            if x in missing_data_flags:
                return float(flag_value)
            else:
                return x
            
        print("Starting Dataset Transformations:")

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

        # Setting flag as 1 for detections and 0 for anything else
        print(f"Replacing PHOTFLAG bitmask with binary values...")
        self.parquet_df = self.parquet_df.with_columns(
            pl.col("PHOTFLAG_clean").map_elements(lambda x: np.where(np.array(x) & 4096 != 0, 1, 0).tolist(), return_dtype=pl.List(pl.Int64)).alias("PHOTFLAG_clean")
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

    def limit_max_samples_per_class(self):
        """
        Limits the number of samples for each class in the DataFrame.
        This method processes the DataFrame contained in the instance attribute `self.parquet_df` by:
            - Determining the unique classes present in the "class" column.
            - For each unique class, selecting only the first `self.max_n_per_class` samples.
            - Concatenating the limited samples from all classes back into `self.parquet_df`.
        It also prints the maximum allowed samples per class and the resulting sample count for each class.

        Note:
            - Assumes self.parquet_df is a Polars DataFrame.
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
        Generates a light curve plot image from time series data and returns it as a Torch tensor.

        Parameters:
            x_ts (numpy.ndarray): 2D array where each row corresponds to a time step and columns represent
                various features including 'jd' (Julian Date), 'flux' (observed flux),
                'flux_err' (flux error), and 'fid' (filter identifier). The feature indices
                are determined using the global variable time_dependent_feature_list.
        Returns:
            torch.Tensor: A tensor representing the RGB image of the generated light curve plot with shape
            (3, H, W), where H and W are the height and width of the image in pixels.

        Note:
            - The function uses matplotlib to create the plot and PIL to handle image conversion.
            - It iterates over wavelengths defined in the global dictionary ZTF_wavelength_to_color, plotting
              error bars for each wavelength filtered by the 'fid' feature.
            - The output image is saved to an in-memory buffer at 100 dpi, then converted from a PIL Image to a
              NumPy array and finally to a Torch tensor.

        Warning:
            - [TODO] This function can be optimized further to avoid using matplotlib and PIL altogether. Its really slow right now...
        """

        # Get the light curve data
        jd = x_ts[:,time_dependent_feature_list.index('MJD')] 
        flux = x_ts[:,time_dependent_feature_list.index('FLUXCAL')]
        flux_err =  x_ts[:,time_dependent_feature_list.index('FLUXCALERR')]
        filters =  x_ts[:,time_dependent_feature_list.index('BAND')]
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

        for wavelength in LSST_passband_wavelengths_to_color.keys():
            
            idx = np.where(filters == wavelength)[0]
            detection_idx = np.where((filters == wavelength) & (phot_flag==1))[0]
            non_detection_idx = np.where((filters == wavelength) & (phot_flag==0))[0]

            ax.errorbar(jd[detection_idx], flux[detection_idx], yerr=flux_err[detection_idx], fmt=marker_style_detection, color=LSST_passband_wavelengths_to_color[wavelength])
            ax.errorbar(jd[non_detection_idx], flux[non_detection_idx], yerr=flux_err[non_detection_idx], fmt=marker_style_non_detection, color=LSST_passband_wavelengths_to_color[wavelength])
            ax.plot(jd[idx], flux[idx], linewidth=linewidth, color=LSST_passband_wavelengths_to_color[wavelength])

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
        Retrieves all labels from the parquet dataframe's 'class' column.

        Returns:
            list: A list of labels extracted from the 'class' column.
        """

        return self.parquet_df['class'].to_list()
    
def truncate_ELAsTiCC_light_curve_by_days_since_trigger(x_ts, d=None):
    """
    Truncates the light curve data to only include observations within a specified number of days since the first detection.

    Parameters:
        x_ts (np.ndarray): A 2D array representing the time series light curve data, where each row is an observation and columns correspond to different features.
        d (float, optional): The number of days after the first detection to use as the cutoff for truncation. If None, a random value is generated using a uniform distribution over the exponent of 2 in the range [0, 11].

    Returns:
        np.ndarray: The truncated light curve array containing only the observations within 'd' days of the first detection.

    Note:
        - Assumes that the column corresponding to 'PHOTFLAG' (indicating detection status) and 'MJD' (the modified Julian date) exist in x_ts.
        - The indices for 'PHOTFLAG' and 'MJD' are obtained using the global list 'time_dependent_feature_list'.
        - Raises an IndexError if no detection (i.e., a value of 1 in the 'PHOTFLAG' column) is found.
    """

    if d == None:
        d = 2**np.random.uniform(0, 11)

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

def truncate_ELAsTiCC_light_curve_fractionally(x_ts, f=None):
    """
    Truncate an ELAstiCC light curve by a fractional amount.
    This function reduces the number of observations in the light curve array based on 
    a specified fraction. If no fraction is provided, a random fraction between 0.1 and 1.0 
    is chosen. The truncation ensures that at least one observation remains.

    Parameters:
        x_ts (numpy.ndarray): A 2D array representing the light curve data, where each row 
            corresponds to an observation and the columns represent different features.
        f (float, optional): A fraction between 0.0 and 1.0 to determine the portion of the 
            light curve to retain. If None, a random fraction in the range [0.1, 1.0] is used.

    Returns:
        numpy.ndarray: The truncated light curve, containing only the first portion of the 
        observations as determined by the fraction.
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

def custom_collate_ELAsTiCC(batch):
    """
    Custom collation function for processing a batch of ELAsTiCC dataset samples.

    Parameters:
        batch (list): A list of dictionaries, each representing a sample.

    Returns:
        dict: A dictionary containing the collated batch with the following keys:
            - 'ts': A padded tensor of time series data with shape (batch_size, max_length, ...), where padding is applied using the predefined flag_value.
            - 'static': A tensor of static features with shape (batch_size, n_static_features).
            - 'length': A tensor containing the lengths of each time series in the batch.
            - 'label': A numpy array of labels for the batch (array-like).
            - 'raw_label': A numpy array of raw ELAsTiCC class labels (array-like).
            - 'id': A numpy array of SNIDs corresponding to each sample.
            - 'lc_plot' (if present in the input samples): A tensor of light curve plots with shape (batch_size, n_channels, img_height, img_width).
    """

    batch_size = len(batch)

    ts_array = []
    label_array = []
    ELASTICC_class_array = []
    snid_array = np.zeros((batch_size))

    lengths = np.zeros((batch_size), dtype=np.int32)
    static_features_tensor = torch.zeros((batch_size, n_static_features),  dtype=torch.float32, device='cpu')
    lc_plot_tensor = torch.zeros((batch_size, n_channels, img_height, img_width), dtype=torch.float32, device='cpu')

    for i, sample in enumerate(batch):

        ts_array.append(sample['ts'])
        label_array.append(sample['label'])
        ELASTICC_class_array.append(sample['ELASTICC_class'])

        snid_array[i] = sample['SNID']
        lengths[i] = sample['ts'].shape[0]
        static_features_tensor[i,:] = sample['static']

        if 'lc_plot' in sample.keys():
            lc_plot_tensor[i,:,:,:] = sample['lc_plot']

    lengths = torch.from_numpy(lengths)
    label_array = np.array(label_array)
    ELASTICC_class_array = np.array(ELASTICC_class_array)

    ts_tensor = pad_sequence(ts_array, batch_first=True, padding_value=flag_value)

    d = {
        'ts': ts_tensor,
        'static': static_features_tensor, 
        'length': lengths,
        'label': label_array,
        'raw_label': ELASTICC_class_array,
        'id': snid_array,
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

    dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_test_parquet_path, include_lc_plots=False, transform=truncate_ELAsTiCC_light_curve_fractionally, max_n_per_class=20000)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_ELAsTiCC)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_ELAsTiCC, num_workers=4, pin_memory=True, prefetch_factor=2)

    for k in range(10):
        for batch in tqdm(dataloader):
            
            pass

        # for k in (batch.keys()):
        #     print(f"{k}: \t{batch[k].shape}")
        # print(batch['label'])

        # for k in (batch.keys()):
        #     print(f"{k}: \t{batch[k].shape}")
        
        # if 'lc_plot' in batch.keys():
        #     show_batch(batch['lc_plot'], batch['label'])
    
    # imgs = []
    # lc_d = []
    # days = np.linspace(10,100,16)
    # for d in days:

    #     k = 4
        
    #     transform = partial(truncate_ELAsTiCC_light_curve_by_days_since_trigger, d=d)
    #     dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_train_parquet_path, include_lc_plots=True, transform=transform)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=custom_collate_ELAsTiCC)
    #     for batch in tqdm(dataloader):
    #         imgs.append(batch['lc_plot'][k,:,:,:])
    #         lc_d.append(max(batch['ts'][k,:,0]))
    #         break
    
    # plt.scatter(days, lc_d)
    # plt.show()
    # show_batch(imgs, batch['label'])