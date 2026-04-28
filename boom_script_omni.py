#load json
import base64
import json
import torch
import astropy
import numpy as np
import gzip
import io
from astropy.io import fits

from oracle.custom_datasets.BTS import show_batch

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from PIL import Image

from oracle.custom_datasets.BTS import ZTF_passband_to_wavelengths
from oracle.custom_datasets.BTS import time_dependent_feature_list, time_independent_feature_list, meta_data_feature_list, flag_value

from oracle.presets import get_model
from astropy.visualization import ImageNormalize, ZScaleInterval, LinearStretch

def load_cutout(field):
    """Decode one {'$binary': {'base64': ...}} field into a 2D numpy array."""
    b64 = field["$binary"]["base64"]
    raw = gzip.decompress(base64.b64decode(b64))
    with fits.open(io.BytesIO(raw)) as hdul:

        image_template = hdul[0].data.astype(float)
        norm = np.linalg.norm(image_template)
        if norm != 0:
            image_template /= norm

        return torch.from_numpy(image_template)

path = Path("alert_aux.json")
prv_cand = None

image_path = Path("test.json")

# Loading the model
Oracle2_omni = get_model("BTSv2-pro")
missing, unexpected = Oracle2_omni.load_state_dict(torch.load("models/BTSv2-pro/morning-feather-572/best_model_f1.pth", map_location='cpu'), strict=True)
Oracle2_omni.eval()

with open(path, "r") as f:
    data = json.load(f)
    prv_cand = data['prv_candidates']
    print(data['_id'])

    prv_cand = pd.DataFrame(prv_cand)
    prv_cand.sort_values('jd', inplace=True)

    final_alert_band = prv_cand['band'].values[-1]

    # convert the jd to time since first detection 
    prv_cand['jd'] = prv_cand['jd'] - prv_cand['jd'].min()

    # convert the filter ids to mean wavelengths
    prv_cand['band'] = prv_cand['band'].map(ZTF_passband_to_wavelengths)

    # we use galactic coordinates as static features, so convert the ra and dec to l and b
    coords = SkyCoord(ra=prv_cand['ra'].to_numpy()*u.deg, dec=prv_cand['dec'].to_numpy()*u.deg, frame='icrs')
    prv_cand['l'] = coords.galactic.l
    prv_cand['b'] = coords.galactic.b


    with open("test.json", "r") as f:
        cutouts = json.load(f)

    image_tensor = torch.zeros((1, 3, 63, 63))  # Dummy image tensor (1, C, H, W)
    if final_alert_band == 'g':
        image_tensor[0,0,:,:] = load_cutout(cutouts['cutoutTemplate'])# put the image here
    elif final_alert_band == 'r':
        image_tensor[0,1,:,:] = load_cutout(cutouts['cutoutTemplate'])# put the image here
    elif final_alert_band == 'i':
        image_tensor[0,2,:,:] = load_cutout(cutouts['cutoutTemplate'])# put the image here


    # Add the wise colors for nearest source within 2.75" 
    # prv_cand['W1mag'] = data['cross_matches']['AllWISE'][0]['w1mpro']  # Default value   
    # prv_cand['W2mag'] = data['cross_matches']['AllWISE'][0]['w2mpro']  # Default value
    # prv_cand['W3mag'] = data['cross_matches']['AllWISE'][0]['w3mpro']  # Default value
    # prv_cand['W4mag'] = data['cross_matches']['AllWISE'][0]['w4mpro']  # Default value
    # prv_cand['W1_minus_W3'] = prv_cand['W1mag'] - prv_cand['W3mag']  # Default value
    # prv_cand['W2_minus_W3'] = prv_cand['W2mag'] - prv_cand['W3mag']  # Default value

    
    # 1 is the batch size
    ts_tensor = torch.zeros((1, len(prv_cand), len(time_dependent_feature_list) + 1))
    for i, col in enumerate(time_dependent_feature_list):
        ts_tensor[0, :, i] = torch.tensor(prv_cand[col].values)
    
    static_tensor = torch.zeros((1, len(time_independent_feature_list)))
    for i, col in enumerate(time_independent_feature_list):
        try:
            static_tensor[0, i] = torch.tensor(prv_cand[col].values[-1])  # Use the last value for static features
        except KeyError:
            print(f"Column {col} not found in data. Setting to default flag value.")
            static_tensor[0, i] = flag_value  # Default value if no data is available

    meta_data_tensor = torch.zeros((1, len(meta_data_feature_list)))
    for i, col in enumerate(meta_data_feature_list):
        try:
            meta_data_tensor[0, i] = torch.tensor(prv_cand[col].values[-1])  # Use the last value for meta data features
        except KeyError:
            print(f"Column {col} not found in data. Setting to default flag value.")
            meta_data_tensor[0, i] = flag_value  # Default value if no data is available
    
    length = torch.tensor([len(prv_cand)])
    static_tensor = torch.cat((static_tensor, meta_data_tensor), dim=1)

    batch = {
        'ts': ts_tensor,
        'static': static_tensor,
        'length': length,
        'postage_stamp': image_tensor
    }

    with torch.no_grad():

        class_scores =  Oracle2_omni.predict_class_probabilities(batch)[0]
        class_scores_df = Oracle2_omni.predict_class_probabilities_df(batch)
        Oracle2_omni.taxonomy.plot_colored_taxonomy(class_scores)

        print("Class Probabilities:")
        print(class_scores_df)
        print("Conditional Probabilities:")
        print(Oracle2_omni.predict_conditional_probabilities_df(batch))






