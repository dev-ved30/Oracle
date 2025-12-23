import io
import torch

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

from oracle.custom_datasets.ELAsTiCC import truncate_ELAsTiCC_light_curve_by_days_since_trigger, truncate_ELAsTiCC_light_curve_fractionally

# Path to this file's directory
here = Path(__file__).resolve().parent

MALLORN_train_parquet_path = str(here.parent.parent.parent / "data" / 'MALLORN' / 'mallorn_train.parquet')
MALLORN_val_parquet_path = str(here.parent.parent.parent / "data" / 'MALLORN' / 'mallorn_val.parquet')
MALLORN_test_parquet_path = str(here.parent.parent.parent / "data" / 'MALLORN' / 'mallorn_test.parquet')

LSST_passband_to_wavelengths = {
    'u': (320 + 400) / (2 * 1000),
    'g': (400 + 552) / (2 * 1000),
    'r': (552 + 691) / (2 * 1000),
    'i': (691 + 818) / (2 * 1000),
    'z': (818 + 922) / (2 * 1000),
    'y': (950 + 1080) / (2 * 1000),
}

# Mean wavelength to colors for plotting
LSST_passband_wavelengths_to_color = {
    LSST_passband_to_wavelengths['u']: np.array((0, 127, 255))/255,
    LSST_passband_to_wavelengths['g']: np.array((127, 0, 255))/255,
    LSST_passband_to_wavelengths['r']: np.array((0, 255, 127))/255,
    LSST_passband_to_wavelengths['i']: np.array((127, 255, 0))/255,
    LSST_passband_to_wavelengths['z']: np.array((255, 127, 0))/255,
    LSST_passband_to_wavelengths['y']: np.array((255, 0, 127))/255,
}

# Marker styles for plotting
marker_style_detection = 'o'
marker_style_non_detection = 'v'
marker_size = 6
linewidth = 0.1

class_mappings = {
    'SN Ia': 'Not TDE',
    'SN Ia-91T-like': 'Not TDE',
    'SN Ia-91bg-like': 'Not TDE',
    'SN Iax[02cx-like]': 'Not TDE',
    'SN Ia-pec': 'Not TDE',
    'SN Ib': 'Not TDE',
    'SN Ib/c': 'Not TDE',
    'SN Ic': 'Not TDE',
    'SN Ic-BL': 'Not TDE',
    'SN II': 'Not TDE',
    'SN IIb': 'Not TDE',
    'SN IIP': 'Not TDE',
    'SN IIn': 'Not TDE',
    'SLSN-I': 'Not TDE',
    'SLSN-II': 'Not TDE',
    'TDE': 'TDE',
    'AGN': 'Not TDE',
}

time_independent_feature_list = ['Z', 'EBV']
time_dependent_feature_list = ['MJD', 'FLUXCAL', 'FLUXCALERR', 'BAND', 'PHOTFLAG']
book_keeping_feature_list = ['object_id', 'SpecType']

n_static_features = len(time_independent_feature_list)
n_ts_features = len(time_dependent_feature_list)
n_book_keeping_features = len(book_keeping_feature_list)

flag_value = -9


class MALLORN_Dataset(torch.utils.data.Dataset):

    def __init__(self, 
                 parquet_file_path,
                 transform=None):
        
        self.transform = transform
        self.parquet_file_path = parquet_file_path

        self.df = pl.read_parquet(self.parquet_file_path)
        self.clean_up_dataset()
        self.map_models_to_classes()

    def map_models_to_classes(self):

        print("Mapping ELAsTiCC classes to astrophysical classes...")
        self.df = self.df.with_columns(
            pl.col("SpecType").replace(class_mappings, return_dtype=pl.String).alias("class")
        )

    def clean_up_dataset(self):

        print("Replacing band labels with mean wavelengths...")
        self.df = self.df.with_columns(
            pl.col("BAND").map_elements(lambda x: [LSST_passband_to_wavelengths[band] for band in x], return_dtype=pl.List(pl.Float64)).alias("BAND")
        )    

        print("Subtracting time of first observation...")
        self.df = self.df.with_columns(
            pl.col("MJD").map_elements(lambda x: (np.array(x) - min(x)).tolist(), return_dtype=pl.List(pl.Float64)).alias("MJD")
        )

        print("Compute the SNR and adding photflag column...")
        self.df = self.df.with_columns(
            pl.struct(["FLUXCAL", "FLUXCALERR"])
            .apply(lambda row: [1 if np.abs(flux / flux_err) >= 3 else 0 for flux, flux_err in zip(row["FLUXCAL"], row["FLUXCALERR"])])
            .alias("PHOTFLAG")
        )

        print("Subtracting time of first observation...")
        self.df = self.df.with_columns(
            pl.col("MJD").map_elements(lambda x: (np.array(x) - min(x)).tolist(), return_dtype=pl.List(pl.Float64)).alias("MJD")
        )

    def get_all_labels(self):
        """
        Retrieves all labels from the parquet dataframe's 'class' column.

        Returns:
            list: A list of labels extracted from the 'class' column.
        """

        return self.df['class'].to_list()

    def __len__(self):

        return self.df.height

    def __getitem__(self, idx):

        row = self.df.row(idx, named=True)

        mallorn_class = row['SpecType']
        id = row['object_id']
        astrophysical_class = row['class']

        lc_length = len(row['MJD'])

        static_np = np.zeros((n_static_features,), dtype=np.float32)
        for i, feature in enumerate(time_independent_feature_list):
            static_np[i] = float(row[feature]) if row[feature] is not None else float(flag_value)
        static_data = torch.from_numpy(static_np)   # CPU tensor

        time_series_data = np.zeros((lc_length, n_ts_features), dtype=np.float32)
        for i, feature in enumerate(time_dependent_feature_list):
            time_series_data[:,i] = np.array(row[f"{feature}"], dtype=np.float32)
        time_series_data = torch.from_numpy(time_series_data)

        if self.transform != None:
            time_series_data = self.transform(time_series_data)
    
        dictionary = {
            'ts': time_series_data,
            'static': static_data,
            'label': astrophysical_class,
            'MALLORN_class': mallorn_class,
            'object_id': id,
        }

        return dictionary

def custom_collate_MALLORN(batch):
    """
    Custom collation function for processing a batch of MALLORN dataset samples.

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
    MALLORN_class_array = []
    id_array = []

    lengths = np.zeros((batch_size), dtype=np.int32)
    static_features_tensor = torch.zeros((batch_size, n_static_features),  dtype=torch.float32, device='cpu')

    for i, sample in enumerate(batch):

        ts_array.append(sample['ts'])
        label_array.append(sample['label'])
        MALLORN_class_array.append(sample['MALLORN_class'])

        id_array.append(sample['object_id'])
        lengths[i] = sample['ts'].shape[0]
        static_features_tensor[i,:] = sample['static']

    lengths = torch.from_numpy(lengths)
    label_array = np.array(label_array)
    MALLORN_class_array = np.array(MALLORN_class_array)
    ts_tensor = pad_sequence(ts_array, batch_first=True, padding_value=flag_value)

    d = {
        'ts': ts_tensor,
        'static': static_features_tensor, 
        'length': lengths,
        'label': label_array,
        'raw_label': MALLORN_class_array,
        'id': id_array,
    }

    return d    


def visualize_batch_light_curves(batch, n_samples=None, ncols=4, figsize_per_plot=(4, 3), save_path=None):
    """
    Visualize light curves from a batch of MALLORN dataset samples.
    
    Parameters:
        batch (dict): A batch dictionary returned by custom_collate_MALLORN containing:
            - 'ts': Padded tensor of time series data (batch_size, max_length, n_features)
            - 'length': Tensor of actual lengths for each light curve
            - 'label': Array of astrophysical class labels
            - 'raw_label': Array of MALLORN class labels
            - 'id': Array of object IDs
        n_samples (int, optional): Number of samples to visualize. If None, visualizes all samples in batch.
        ncols (int, optional): Number of columns in the subplot grid. Default is 4.
        figsize_per_plot (tuple, optional): Size of each individual plot (width, height). Default is (4, 3).
        save_path (str, optional): If provided, saves the figure to this path instead of displaying it.
    
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    
    Note:
        - Detections (PHOTFLAG=1) are shown with circles (o)
        - Non-detections (PHOTFLAG=0) are shown with triangles (v)
        - Each filter is plotted with a different color according to LSST_passband_wavelengths_to_color
        - The title shows the object ID, astrophysical class, and MALLORN class
    """
    
    # Extract data from batch
    ts_tensor = batch['ts']  # (batch_size, max_length, n_features)
    lengths = batch['length']  # (batch_size,)
    labels = batch['label']  # (batch_size,)
    raw_labels = batch['raw_label']  # (batch_size,)
    object_ids = batch['id']  # (batch_size,)
    
    batch_size = ts_tensor.shape[0]
    
    # Determine number of samples to plot
    if n_samples is None:
        n_samples = batch_size
    else:
        n_samples = min(n_samples, batch_size)
    
    # Calculate grid dimensions
    nrows = int(np.ceil(n_samples / ncols))
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
    
    # Ensure axes is always a 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Feature indices
    mjd_idx = time_dependent_feature_list.index('MJD')
    flux_idx = time_dependent_feature_list.index('FLUXCAL')
    flux_err_idx = time_dependent_feature_list.index('FLUXCALERR')
    band_idx = time_dependent_feature_list.index('BAND')
    photflag_idx = time_dependent_feature_list.index('PHOTFLAG')
    
    # Plot each sample
    for i in range(n_samples):
        ax = axes_flat[i]
        
        # Get the actual length of this light curve
        actual_length = lengths[i].item()
        
        # Extract time series data for this sample (only up to actual length)
        ts_data = ts_tensor[i, :actual_length, :].cpu().numpy()
        
        # Extract features
        mjd = ts_data[:, mjd_idx]
        flux = ts_data[:, flux_idx]
        flux_err = ts_data[:, flux_err_idx]
        bands = ts_data[:, band_idx]
        phot_flags = ts_data[:, photflag_idx]
        
        # Plot each filter
        for wavelength in LSST_passband_wavelengths_to_color.keys():
            # Find indices for this wavelength
            band_mask = np.isclose(bands, wavelength, atol=1e-6)
            detection_mask = band_mask & (phot_flags == 1)
            non_detection_mask = band_mask & (phot_flags == 0)
            
            color = LSST_passband_wavelengths_to_color[wavelength]
            
            # Plot detections
            if np.any(detection_mask):
                ax.errorbar(mjd[detection_mask], flux[detection_mask], 
                           yerr=flux_err[detection_mask], 
                           fmt=marker_style_detection, 
                           color=color,
                           markersize=marker_size,
                           linewidth=linewidth,
                           capsize=2)
            
            # Plot non-detections
            if np.any(non_detection_mask):
                ax.errorbar(mjd[non_detection_mask], flux[non_detection_mask], 
                           yerr=flux_err[non_detection_mask], 
                           fmt=marker_style_non_detection, 
                           color=color,
                           markersize=marker_size,
                           linewidth=linewidth,
                           capsize=2)
            
            # Connect points with lines
            if np.any(band_mask):
                ax.plot(mjd[band_mask], flux[band_mask], 
                       color=color, 
                       linewidth=linewidth,
                       alpha=0.5)
        
        # Set title with object info
        title = f"ID: {object_ids[i]}\n{labels[i]} ({raw_labels[i]})"
        ax.set_title(title, fontsize=9)
        
        # Labels
        ax.set_xlabel('MJD (days)', fontsize=8)
        ax.set_ylabel('Flux', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_samples, len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Create legend for filters
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=LSST_passband_wavelengths_to_color[LSST_passband_to_wavelengths['u']], 
                   markersize=8, label='u'),
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=LSST_passband_wavelengths_to_color[LSST_passband_to_wavelengths['g']], 
                   markersize=8, label='g'),
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=LSST_passband_wavelengths_to_color[LSST_passband_to_wavelengths['r']], 
                   markersize=8, label='r'),
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=LSST_passband_wavelengths_to_color[LSST_passband_to_wavelengths['i']], 
                   markersize=8, label='i'),
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=LSST_passband_wavelengths_to_color[LSST_passband_to_wavelengths['z']], 
                   markersize=8, label='z'),
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=LSST_passband_wavelengths_to_color[LSST_passband_to_wavelengths['y']], 
                   markersize=8, label='y'),
        plt.Line2D([0], [0], marker='o', color='k', markersize=8, label='Detection'),
        plt.Line2D([0], [0], marker='v', color='k', markersize=8, label='Non-detection'),
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=8, fontsize=9, frameon=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space for legend at bottom
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


if __name__ == "__main__":

    dataset = MALLORN_Dataset(parquet_file_path=MALLORN_train_parquet_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_MALLORN, num_workers=4, pin_memory=True, prefetch_factor=2)

    # Example: Visualize the first batch
    batch = next(iter(dataloader))
    visualize_batch_light_curves(batch, n_samples=16, ncols=4)
    
    # Optionally test the full dataloader
    # for k in range(10):
    #     for batch in tqdm(dataloader):
    #         pass