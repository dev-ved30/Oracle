import polars as pl
import argparse
import random

def parse_args():
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser(description="Split a Parquet file into train and validation sets.")
    parser.add_argument("input_path", help="Path to the input Parquet file.")
    parser.add_argument("output_train_path", help="Path to save the train Parquet file.")
    parser.add_argument("output_val_path", help="Path to save the validation Parquet file.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data. Default is 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")

    args = parser.parse_args()
    return args

def split_parquet(input_path, output_train_path, output_val_path, val_ratio=0.1,seed=42):

    # Load the full dataset
    df = pl.read_parquet(input_path)
    
    random.seed(seed)
    df = df.sample(fraction=1.0, with_replacement=False, shuffle=True, seed=seed)

    total_rows = df.shape[0]
    val_size = int(total_rows * val_ratio)

    # Split the DataFrame
    df_val = df[:val_size]
    df_train = df[val_size:]

    # Write the splits to new Parquet files
    df_train.write_parquet(output_train_path)
    df_val.write_parquet(output_val_path)

    print(f"Split {input_path} ->")
    print(f"Train: {output_train_path} ({df_train.shape[0]} rows)")
    print(f"Validation: {output_val_path} ({df_val.shape[0]} rows)")

if __name__ == "__main__":

    args = parse_args()

    split_parquet(
        input_path=args.input_path,
        output_train_path=args.output_train_path,
        output_val_path=args.output_val_path,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
