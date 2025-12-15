import io
import torch

import polars as pl
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

from oracle.custom_datasets.ELAsTiCC import truncate_ELAsTiCC_light_curve_by_days_since_trigger, truncate_ELAsTiCC_light_curve_fractionally

# Path to this file's directory
here = Path(__file__).resolve().parent

MALLORN_train_parquet = str(here.parent.parent.parent / "data" / 'MALLORN' / 'mallorn_train.parquet')
MALLORN_val_parquet = str(here.parent.parent.parent / "data" / 'MALLORN' / 'mallorn_val.parquet')
MALLORN_test_parquet = str(here.parent.parent.parent / "data" / 'MALLORN' / 'mallorn_test.parquet')

LSST_passband_to_wavelengths = {
    'u': (320 + 400) / (2 * 1000),
    'g': (400 + 552) / (2 * 1000),
    'r': (552 + 691) / (2 * 1000),
    'i': (691 + 818) / (2 * 1000),
    'z': (818 + 922) / (2 * 1000),
    'y': (950 + 1080) / (2 * 1000),
}

class_mappings = {
    'SN Ia': 'Not TDE',
    'SN Ia-91T-like': 'Not TDE',
    'SN Ia-91bg-like': 'Not TDE',
    'SN Ia02cx-like': 'Not TDE',
    'SN Ia-pec': 'Not TDE',
    'SN Ib': 'Not TDE',
    'SN Ib/c': 'Not TDE',
    'SN Ic': 'Not TDE',
    'SN Ic-BL': 'Not TDE',
    'SN II': 'Not TDE',
    'SN IIb': 'Not TDE',
    'SN IIn': 'Not TDE',
    'SLSN-I': 'Not TDE',
    'SLSN-II': 'Not TDE',
    'TDEs': 'TDE',
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
            .apply(lambda row: [4096 if np.abs(flux / flux_err) > 0 else 0 for flux, flux_err in zip(row["FLUXCAL"], row["FLUXCALERR"])])
            .alias("PHOTFLAG")
        )

        print(f"Replacing PHOTFLAG bitmask with binary values...")
        self.df = self.df.with_columns(
            pl.col("PHOTFLAG").map_elements(lambda x: np.where(np.array(x) & 4096 != 0, 1, 0).tolist(), return_dtype=pl.List(pl.Int64)).alias("PHOTFLAG")
        )

        print("Subtracting time of first observation...")
        self.df = self.df.with_columns(
            pl.col("MJD").map_elements(lambda x: (np.array(x) - min(x)).tolist(), return_dtype=pl.List(pl.Float64)).alias("MJD")
        )


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

if __name__ == "__main__":

    dataset = MALLORN_Dataset(parquet_file_path=MALLORN_train_parquet)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_MALLORN, num_workers=4, pin_memory=True, prefetch_factor=2)

    for k in range(10):
        for batch in tqdm(dataloader):
            
            pass