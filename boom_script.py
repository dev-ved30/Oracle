#load json
import json
import torch
import astropy


import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord

from oracle.custom_datasets.BTS import ZTF_passband_to_wavelengths
from oracle.custom_datasets.BTS import time_dependent_feature_list, time_independent_feature_list, meta_data_feature_list, flag_value

from oracle.presets import get_model

path = Path("alert_aux.json")
prv_cand = None

# Loading the model
Oracle2 = get_model("BTSv2")
Oracle2.load_state_dict(torch.load("models/BTSv2/peachy-sweep-4/best_model_f1.pth", map_location='cpu'), strict=False)
Oracle2.eval()

with open(path, "r") as f:
    data = json.load(f)
    prv_cand = data['prv_candidates']

    prv_cand = pd.DataFrame(prv_cand)
    prv_cand.sort_values('jd', inplace=True)

    # convert the jd to time since first detection 
    prv_cand['jd'] = prv_cand['jd'] - prv_cand['jd'].min()

    # convert the filter ids to mean wavelengths
    prv_cand['band'] = prv_cand['band'].map(ZTF_passband_to_wavelengths)

    # we use galactic coordinates as static features, so convert the ra and dec to l and b
    coords = SkyCoord(ra=prv_cand['ra'].to_numpy()*u.deg, dec=prv_cand['dec'].to_numpy()*u.deg, frame='icrs')
    prv_cand['l'] = coords.galactic.l
    prv_cand['b'] = coords.galactic.b

    # Add the wise colors for nearest source within 2.75" 
    # prv_cand['W1mag'] = flag_value  # Default value   
    # prv_cand['W2mag'] = flag_value  # Default value
    # prv_cand['W3mag'] = flag_value  # Default value
    # prv_cand['W4mag'] = flag_value  # Default value
    # prv_cand['W1_minus_W3'] = flag_value  # Default value
    # prv_cand['W2_minus_W3'] = flag_value  # Default value

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
    }

    with torch.no_grad():

        class_scores = Oracle2.predict_class_probabilities(batch)[0]
        class_scores_df = Oracle2.predict_conditional_probabilities_df(batch)
        Oracle2.taxonomy.plot_colored_taxonomy(class_scores)

        print(class_scores_df)






