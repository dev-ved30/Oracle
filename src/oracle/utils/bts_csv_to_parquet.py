import numpy as np
import pandas as pd
import polars as pl

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from oracle.constants import ztf_filter_to_fid, ztf_filters, ztf_alert_image_order

def parse_args(argv=None):
    parser = ArgumentParser(
        prog='Csv2parquet',
        description='Convert CSV of all BTS alerts to parquet',
    )
    parser.add_argument('--data_csv_path', type=Path, 
                        help='Path to csv file with alerts')
    parser.add_argument('--labels_csv_path', type=Path, 
                        help='Path to csv file with labels')
    parser.add_argument('--images_np_path', type=Path, 
                        help='Path to np arrays containing alert images')
    parser.add_argument('--output_path', type=Path,
                        help='Output parquet file path, typically has .parquet extension')

    return parser.parse_args(argv)

def main(argv=None):

    args = parse_args(argv)
    data_csv_path = args.data_csv_path
    labels_csv_path = args.labels_csv_path
    images_np_path = args.images_np_path
    output_path = args.output_path

    data_df = pd.read_csv(data_csv_path)
    labels_df = pd.read_csv(labels_csv_path)
    images = np.load(images_np_path, mmap_mode='r')

    unique_data_id = np.unique(data_df['objectId'])
    id_with_labels = labels_df['ZTFID'].unique()

    print(data_df)

    # Array to store all the rows before finally converting it to a parquet file
    all_rows = []

    for id in tqdm(unique_data_id):

        # Check if the id has a label
        if id in id_with_labels:

            # Get the label from the other data frame
            bts_class = labels_df[labels_df['ZTFID']==id]['type'].to_numpy()[0]

            if bts_class not in ['-','bogus','duplicate']:

                # Get the time series data
                source_df = data_df[data_df['objectId'] == id].sort_values(by='jd').drop('objectId', axis=1)

                # Next, we find the images corresponding to this source. The number of images should be equal to the length of the time series * 3
                # Since we only care about the reference images, we pick the first one from each of the filters
                img_dictionary = {}

                for f in ztf_filters:
                    
                    # Find all observations in this pass band
                    source_df_in_filter = source_df[source_df['fid']==ztf_filter_to_fid[f]]

                    # Make sure there is at least one alert in the pass band
                    if source_df_in_filter.to_numpy().shape[0] > 0:

                        img_index = source_df_in_filter.index[0]

                        # Grab the set of ref, science, and diff images from the first alert
                        for j, img_type in enumerate(ztf_alert_image_order):

                            # Grab the image data and then flatten it
                            img = images[img_index, :, :, j] # shape: (N, 63, 63, 3)
                            flattened_img = img.flatten()

                            img_dictionary[f"{f}_{img_type}"] = flattened_img

                    else: 

                        # Grab the set of ref, science, and diff images from the first alert
                        for j, img_type in enumerate(ztf_alert_image_order):
                            img_dictionary[f"{f}_{img_type}"] = None

                    
                # Start assembling new row in parquet file
                new_row ={}
                new_row['ZTFID'] = id
                new_row['bts_class'] = bts_class
                
                # Add all the images
                for k in img_dictionary.keys():
                    new_row[k] = [img_dictionary[k]]
                
                # Add all the time series data
                for c in source_df.columns:
                    new_row[c] = [source_df[c].to_numpy()]

                new_row = pl.DataFrame(new_row)
                all_rows.append(new_row)

    final_parquet = pl.concat(all_rows, how='diagonal')
    final_parquet = pl.from_dataframe(final_parquet)
    final_parquet.write_parquet(output_path)

    print([c for c in zip(np.unique(final_parquet['bts_class'], return_counts=True))])
    print(f"Labels found for {len(final_parquet)} out of {len(unique_data_id)} sources.")
    #print(np.unique([x[0] for x in final_parquet['source_set']],return_counts=True))
    print(final_parquet)

if __name__ == '__main__':
    main()