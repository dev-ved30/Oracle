import os
import torch

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from oracle.constants import ELAsTiCC_to_Astrophysical_mappings

# <----- constant for the dataset ----->

batch_size = 512

img_height = 256
img_width = 256
n_channels = 3


# Creating a custom dataset class 
class BTS_LC_Image_Dataset(torch.utils.data.Dataset):

    def __init__(self, dir, transform=None): 
        super(BTS_LC_Image_Dataset, self).__init__()

        self.labels_df = pd.read_csv(f'{dir}/../bts_labels.txt')

        print(np.unique(self.labels_df['type'], return_counts=True))
        self.data_dir = dir
        self.images = os.listdir(dir) 
        self.transform = transform 
  
    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.images) 
  
    # Defining the method to get an item from the dataset 
    def __getitem__(self, index): 

        image_path = os.path.join(self.data_dir, self.images[index]) 
        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32) 
        image = np.permute_dims(image, (2, 0, 1))
  
        # Applying the transform 
        if self.transform: 
            image = self.transform(image) 
        
        ztf_object_id = self.images[index].split('.')[0]
        class_name = self.labels_df[self.labels_df['ZTFID'] == ztf_object_id]['type'].to_numpy()

        if len(class_name) == 0:
            class_name = ''
        elif len(class_name) == 1:
            class_name = class_name[0]
        else:
            raise ValueError(f'Found multiple class names for {ztf_object_id}')
          
        return image, class_name

class ELAsTiCC_LC_Dataset(torch.utils.data.Dataset):

    def __init__(self, parquet_file_path, transform=None):
        super(ELAsTiCC_LC_Dataset, self).__init__()

        # Columns to be read from the parquet file
        self.columns = ['SNID', 'MJD', 'BAND','PHOTFLAG', 'PHOTPROB', 'FLUXCAL', 'FLUXCALERR', 'ELASTICC_class']

        self.parquet_file_path = parquet_file_path
        self.parquet_df = pd.read_parquet(self.parquet_file_path, columns=self.columns, engine='pyarrow')
        self.transform = transform

    def __len__(self):

        return len(self.parquet_df)

    def __getitem__(self, index):

        row = self.parquet_df.iloc[index]

        SNID = row['SNID']
        class_name = row['ELASTICC_class']

        time_series = np.array([row['MJD'], row['PHOTFLAG'], row['PHOTPROB'], row['FLUXCAL'], row['FLUXCALERR']]) 
        time_series = torch.from_numpy(time_series)

        print(SNID, class_name, time_series)
        
        return time_series, class_name, SNID

if __name__=='__main__':
    # <--- Example usage of the dataset --->
    dataset = BTS_LC_Image_Dataset('data/BTS/bts_lc_images')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, item in enumerate(dataloader):
        data, labels = item
        print(data.shape)
        print(labels)
        break

    dataset = ELAsTiCC_LC_Dataset('data/ELAsTiCC/train_parquet.parquet')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, item in enumerate(dataloader):
        data, labels, snid = item
        print(data.shape)
        print(labels)
        break
