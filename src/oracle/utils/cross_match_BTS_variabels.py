
import pandas as pd
import time

from astropy.coordinates import SkyCoord
from astropy.io import ascii
from tqdm import tqdm
from astroquery.vizier import Vizier

import astropy.units as u
import requests

headers = {"Authorization": "token 260a3d22-a832-4213-b6aa-29052908e5da"}

# # Load dataframe that contains a column: 'ztf_id'
# df = pd.read_csv("data/BTS_new/rcfdeep_objids.txt", names=["ztf_id"])

# ras = []
# decs = []
# ra_errs = []
# dec_errs = []



# for obj_id in tqdm(df['ztf_id'].values):

#     url = "https://fritz.science/api/sources/__OBJ_ID__"
#     response = requests.get(url.replace("__OBJ_ID__", str(obj_id)), headers=headers)
#     # Make sure request was successful else sleep and try again

#     while response.status_code != 200:

#         time.sleep(1)
#         response = requests.get(url.replace("__OBJ_ID__", str(obj_id)), headers=headers)

#     json = response.json()['data'] 
#     ra, dec, ra_err, dec_err = json['ra'], json['dec'], json['ra_err'], json['dec_err']
#     ras.append(ra)
#     decs.append(dec)
#     ra_errs.append(ra_err)
#     dec_errs.append(dec_err)

    
# df['ra'] = ras
# df['dec'] = decs
# df['ra_err'] = ra_errs
# df['dec_err'] = dec_errs

# df.to_csv("data/BTS_new/rcfdeep_objids_with_coords.txt", index=False)


df = pd.read_csv("data/BTS_new/rcfdeep_objids_with_coords.txt")


# Load the Vizier catalog

catalog_path = "data/BTS_new/apjsab9caet2_mrt.txt"
catalog = ascii.read(catalog_path, format='cds')


# Cross match the df with the catalog using astropy
df_coords = SkyCoord(ra=df['ra'].values*u.degree, dec=df['dec'].values*u.degree)
catalog_coords = SkyCoord(ra=catalog['RAdeg'], dec=catalog['DEdeg'], unit=(u.deg, u.deg))

idx, d2d, _ = df_coords.match_to_catalog_sky(catalog_coords)

# Only keep matches within 2 arcseconds
max_sep = 2 * u.arcsec
matched = d2d < max_sep

matched_df = df[matched]
matched_df['class'] = catalog[idx[matched]]['Type'] 
print(matched_df)
matched_df.to_csv("data/BTS_new/var_star_list.txt", index=False)