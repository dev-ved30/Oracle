import argparse

import numpy as np
import pandas as pd
import astropy.units as u

from astropy.table import Table
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from tqdm import tqdm

missing_flag_value = -9

def parse_args():

    parse = argparse.ArgumentParser(description="Fetch WISE magnitudes for given parquet_file.")
    parse.add_argument("parquet_file", type=str, help="Path to the input parquet file containing RA and Dec columns.")
    return parse.parse_args()

def get_wise_magnitudes(ra, dec, radius_arcsec=2.75, catalog='II/328/allwise'):
    """
    Queries Vizier for WISE magnitudes (W1, W2, W3, W4) for a given RA and Dec.
    
    Parameters:
    ra (float): Right Ascension in degrees.
    dec (float): Declination in degrees.
    radius_arcsec (float): Search radius in arcseconds.
    catalog (str): Vizier catalog ID. Default is AllWISE (II/328/allwise).
    
    Returns:
    astropy.table.Table: A table containing the matched WISE sources and their magnitudes.
    """
    
    # Initialize Vizier with the columns we want
    v = Vizier(columns=['W1mag', 'W2mag', 'W3mag', 'W4mag'],
               catalog=[catalog])
    
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    
    try:
        result = v.query_region(coord, radius=radius_arcsec * u.arcsec)
    except Exception as e:
        print(f"Error querying Vizier: {e}")
        return None

    if not result:
        # print(f"No sources found in {catalog} within {radius_arcsec} arcseconds of RA={ra}, Dec={dec}")

        # Use flag values to indicate no data
        table = Table(names=['W1mag', 'W2mag', 'W3mag', 'W4mag'],
                            dtype=[float, float, float, float])
        table.add_row([missing_flag_value, missing_flag_value, missing_flag_value, missing_flag_value])
    
    else:
        table = result[0]

    # result is a TableList, we want the first table
    table = table[:1].to_pandas()
    return table

def main():

    args = parse_args()
    parquet_file = args.parquet_file

    df = pd.read_parquet(parquet_file)
    print(df)

    if 'ra' not in df.columns or 'dec' not in df.columns:
        raise ValueError("Input parquet file must contain 'ra' and 'dec' columns.")

    wise_magnitudes_list = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Fetching WISE magnitudes"):
        ra = row['ra']
        dec = row['dec']

        wise_table = get_wise_magnitudes(np.mean(ra), np.mean(dec))
        wise_table['ZTFID'] = row['ZTFID']
        wise_magnitudes_list.append(wise_table)

    # Combine all WISE tables into a single DataFrame
    wise_magnitudes_df = pd.concat([pd.DataFrame(t) for t in wise_magnitudes_list], ignore_index=True)

    # Combine with original DataFrame if needed
    combined_df = pd.merge(df, wise_magnitudes_df, on='ZTFID', how='left')

    # Save to a new parquet file
    output_file = parquet_file.replace('.parquet', '_wise_magnitudes.parquet')
    print(combined_df)
    combined_df.to_parquet(output_file)
    print(f"Wrote WISE magnitudes to {output_file}")

if __name__ == "__main__":

    main()