from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sncosmo import read_snana_fits

seed = 42


def parse_args(argv=None):
    parser = ArgumentParser(
        prog='Combine parquet',
        description='Combine all parquet files into one big parquet file',
    )
    parser.add_argument('input', type=Path, 
                        help='Path to folder containing individual parquet files')
    parser.add_argument('output', type=Path,
                        help='Output parquet file path')
    return parser.parse_args(argv)

class_mapper = {
    'ZTF_MSIP_MODEL01': 'SNIa-normal',  
    'ZTF_MSIP_MODEL02':  'SNCC-II',
    'ZTF_MSIP_MODEL03': 'SNCC-Ibc',  
    'ZTF_MSIP_MODEL12': 'SNCC-II',  
    'ZTF_MSIP_MODEL13': 'SNCC-Ibc',  
    'ZTF_MSIP_MODEL14': 'SNCC-II',
    'ZTF_MSIP_MODEL41': 'SNIa-91bg',  
    'ZTF_MSIP_MODEL43': 'SNIa-x ',
    'ZTF_MSIP_MODEL51': 'KN',  
    'ZTF_MSIP_MODEL60': 'SLSN-I',  
    'ZTF_MSIP_MODEL61': 'PISN',  
    'ZTF_MSIP_MODEL62': 'ILOT',  
    'ZTF_MSIP_MODEL63': 'CART',  
    'ZTF_MSIP_MODEL64': 'TDE',  
    'ZTF_MSIP_MODEL70': 'AGN',  
    'ZTF_MSIP_MODEL80': 'RRlyrae',  
    'ZTF_MSIP_MODEL81': 'Mdwarf',  
    'ZTF_MSIP_MODEL83': 'EBE',  
    'ZTF_MSIP_MODEL84': 'MIRA',  
    'ZTF_MSIP_MODEL90': 'uLens-Binary',  
    'ZTF_MSIP_MODEL91': 'uLens-Point',  
    'ZTF_MSIP_MODEL92': 'uLens-STRING',  
    'ZTF_MSIP_MODEL93': 'uLens-Point',  
}

def main(argv=None):
    args = parse_args(argv)

    file_paths = sorted(args.input.glob('*.parquet'))

    # Read each parquet file into a Polars DataFrame.
    frames = []

    for file_path in file_paths:

        object_class = str(file_path).split('/')[-1].split('.')[0]
        object_class = class_mapper[object_class]
        print(f'Opening {object_class}', flush=True)

        # Load file, add label, and shuffle
        dataframe = pl.read_parquet(file_path)
        dataframe = dataframe.with_columns(pl.lit(object_class).alias('ZTF_class'))
        dataframe = dataframe.sample(fraction=1, shuffle=True, seed=seed)

        frames.append(dataframe)

    # Combine the DataFrames into a single DataFrame.
    combined_file = pl.concat(frames, how='diagonal')

    print(f"=========\Combined size {combined_file.shape[0]}", flush=True)

    # Write the combined DataFrame to a new parquet file.
    combined_file.write_parquet(args.output)
    print("Saved!")

if __name__ == '__main__':
    main()