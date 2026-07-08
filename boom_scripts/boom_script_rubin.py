#load json
import json
import torch
import astropy


import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord

from oracle.custom_datasets.ELAsTiCC import LSST_passband_to_wavelengths
from oracle.custom_datasets.ELAsTiCC import time_dependent_feature_list

from oracle.presets import get_model

detection_snr_threshold = 5.0

path = Path("alert_lsst_aux.json")
prv_cand = None

Oracle2_lite = get_model("ELAsTiCCv2-lite")
Oracle2_lite.load_state_dict(torch.load("models/ELAsTiCCv2-lite/bright-wood-176/best_model_f1.pth", map_location='cpu'), strict=False)
Oracle2_lite.eval()

with open(path, "r") as f:
    data = json.load(f)
    prv_cand = data['prv_candidates']

    prv_cand = pd.DataFrame(prv_cand)
    prv_cand.sort_values('jd', inplace=True)

    # convert the jd to time since first detection 
    prv_cand['MJD'] = prv_cand['jd'] - prv_cand['jd'].min()

    # convert the filter ids to mean wavelengths
    prv_cand['BAND'] = prv_cand['band'].map(LSST_passband_to_wavelengths)

    # NOTE: exact fields may need to be changed based on the actual data format. This is just an example of how to handle the alert.
    prv_cand['FLUXCAL'] = prv_cand['scienceFlux']  
    prv_cand['FLUXCALERR'] = prv_cand['scienceFluxErr']
    prv_cand['PHOTFLAG'] = [1 if x >= 5 else 0 for x in prv_cand['snr']]  # Handle None values

    # 1 is the batch size
    ts_tensor = torch.zeros((1, len(prv_cand), len(time_dependent_feature_list)))
    for i, col in enumerate(time_dependent_feature_list):
        ts_tensor[0, :, i] = torch.tensor(prv_cand[col].values)

    length = torch.tensor([len(prv_cand)], dtype=torch.long)

    batch = {
        'ts': ts_tensor,
        'length': length,
    }


    with torch.no_grad():

        class_scores =  Oracle2_lite.predict_class_probabilities(batch)[0]
        Oracle2_lite.taxonomy.plot_colored_taxonomy(class_scores)

        print(class_scores)

