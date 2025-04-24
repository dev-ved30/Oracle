import torch

import numpy as np
import pandas as pd

from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from oracle.models import ORACLE_1
from oracle.taxonomies import ORACLE_Taxonomy
from oracle.constants import ELAsTiCC_to_Astrophysical_mappings as class_mapper

time_independent_feature_list = ['MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y']
time_dependent_feature_list = ['MJD', 'PHOTFLAG', 'FLUXCAL', 'FLUXCALERR', 'BAND']
book_keeping_feature_list = ['SNID', 'ELASTICC_class']

# Path to this file's directory
here = Path(__file__).resolve().parent

# Go up to the root, then into data/
data_path = here.parent.parent.parent / "data" 

data_files={
    'train': str(data_path / 'ELasTiCC' / 'train.parquet'), 
    'test': str(data_path / 'ELasTiCC' / 'test.parquet')
}

# Mean wavelengths for the LSST pass bands in micrometers
LSST_passband_to_wavelengths = {
    'u': (320 + 400) / (2 * 1000),
    'g': (400 + 552) / (2 * 1000),
    'r': (552 + 691) / (2 * 1000),
    'i': (691 + 818) / (2 * 1000),
    'z': (818 + 922) / (2 * 1000),
    'Y': (950 + 1080) / (2 * 1000),
}

max__elasticc_ts_len = 500

flag_value = -9

# Flag values for missing data of static feature according to elasticc
missing_data_flags = [-9, -99, -999, -9999, 999]

def truncate_light_curves_fractionally(examples, fraction=None):

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

        # Mask off the saturations
        PHOTFLAG = examples['PHOTFLAG'][i]
        saturation_mask =  (np.array(PHOTFLAG) & 1024) == 0 

        # Remove the saturations
        MJD = np.array(examples['MJD'][i])[saturation_mask]
        FLUXCAL =  np.array(examples['FLUXCAL'][i])[saturation_mask] 
        FLUXCALERR =  np.array(examples['FLUXCALERR'][i])[saturation_mask]
        BAND =  np.array(examples['BAND'][i])[saturation_mask]
        PHOTFLAG =  np.array(examples['PHOTFLAG'][i])[saturation_mask]

        # Apply the fraction limit on the light curve
        original_obs_count = len(MJD)
        new_obs_count = int(original_obs_count * fractions_array[i])
        if new_obs_count < 1:
            new_obs_count = 1
        
        # Other transformations
        examples['MJD'][i] = MJD[:new_obs_count] - MJD[0]
        examples['FLUXCAL'][i] = FLUXCAL[:new_obs_count]
        examples['FLUXCALERR'][i] = FLUXCALERR[:new_obs_count]
        examples['BAND'][i] = [LSST_passband_to_wavelengths[b] for b in BAND[:new_obs_count]]
        examples['PHOTFLAG'][i] = np.where((PHOTFLAG[:new_obs_count] & 4096 != 0), 1, 0)
    
    return examples

def ELAsTiCC_collate_fn(batch):

    ts_data = []
    static_data = []
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
            if b[feature] in missing_data_flags:
                s.append(flag_value)
            else:
                s.append(b[feature])

        static_data.append(np.array(s, dtype=np.float32))
        lengths.append(length)
        labels.append(class_mapper[b['ELASTICC_class']])
        snids.append(b['SNID'])
        
    ts_data = [torch.from_numpy(np.squeeze(x)) for x in ts_data]
    ts_data = pad_sequence(ts_data, batch_first=True, padding_value=flag_value)

    static_data = torch.from_numpy(np.squeeze(static_data))
    
    d = {
        'ts_data':ts_data,
        'static_data':static_data,
        'lengths':lengths,
        'labels':labels,
        'SNID':snids
    }

    return d

def get_ELAsTiCC_dataset(split):

    dataset = load_dataset("parquet", data_files=data_files, split=split)
    dataset = dataset.select_columns(time_independent_feature_list+time_dependent_feature_list+book_keeping_feature_list)
    dataset.set_format(type="torch")
    return dataset

def main():

    dataset = get_ELAsTiCC_dataset('train')
    dataset = dataset.with_transform(truncate_light_curves_fractionally)
    dataloader = DataLoader(dataset, batch_size=256, collate_fn=ELAsTiCC_collate_fn, shuffle=True)

    taxonomy = ORACLE_Taxonomy()
    model = ORACLE_1(taxonomy)

    for batch in dataloader:

        logits = model(batch)
        print(logits.shape)

        model.eval()
        print(model.get_class_probabilities_df(batch))

        print(batch['ts_data'].shape)
        print(batch['static_data'].shape)
        print(batch['labels'])
        print('----------')
        break

if __name__ == '__main__':

    main()