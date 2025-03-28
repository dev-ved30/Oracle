import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# <----- constant for the dataset ----->

batch_size = 512

img_height = 256
img_width = 256
n_channels = 3


# Creating a custom dataset class 
class BTS_LC_Dataset(torch.utils.data.Dataset): 
    def __init__(self, dir, transform=None): 

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
          
        return image, np.ones(26)

if __name__=='__main__':
    # <--- Example usage of the dataset --->
    dataset = BTS_LC_Dataset('data/BTS/LC')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, data in enumerate(dataloader):
        print(data.shape)
        break