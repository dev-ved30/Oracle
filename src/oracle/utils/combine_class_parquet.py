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


def main(argv=None):
    args = parse_args(argv)

    file_paths = sorted(args.input.glob('*.parquet'))

    # Read each parquet file into a Polars DataFrame.
    frames = []

    for file_path in file_paths:

        object_class = str(file_path).split('/')[-1].split('.')[0]
        print(f'Opening {object_class}', flush=True)

        # Load file, add label, and shuffle
        dataframe = pl.read_parquet(file_path)
        dataframe = dataframe.with_columns(pl.lit(object_class).alias('ELASTICC_class'))
        dataframe = dataframe.sample(fraction=1, shuffle=True, seed=seed)

        frames.append(dataframe)

    # Combine the DataFrames into a single DataFrame.
    combined_file = pl.concat(frames, how='diagonal')

    print(f"=========\Combined size {combined_file.shape[0]}", flush=True)

    # Write the combined DataFrame to a new parquet file.
    combined_file.write_parquet(args.train_output)
    print("Saved!")

if __name__ == '__main__':
    main()