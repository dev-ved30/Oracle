import pandas as pd
import argparse
import random

from oracle.constants import BTS_to_Astrophysical_mappings

def parse_args():
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser(description="Split a Parquet file into train and validation sets.")
    parser.add_argument("input_path", help="Path to the input Parquet file.")
    parser.add_argument("output_train_path", help="Path to save the train Parquet file.")
    parser.add_argument("output_val_path", help="Path to save the validation Parquet file.")
    parser.add_argument("output_test_path", help="Path to save the test Parquet file.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data. Default is 0.1")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test data. Default is 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")

    args = parser.parse_args()
    return args

def split_parquet(input_path, output_train_path, output_val_path, output_test_path, val_ratio=0.1, test_ratio=0.1, seed=42):

    train_ratio = 1 - val_ratio - test_ratio

    # Load the full dataset
    df = pd.read_parquet(input_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(df)

    df_train = []
    df_val = []
    df_test = []

    for c in df['bts_class'].unique():

        df_class = df[df['bts_class'] == c]

        if BTS_to_Astrophysical_mappings[c] == 'Anomaly':

            # Only include this in the test set
            df_test.append(df_class)

        else:

            # distribute between train, val, and test

            total_rows = df_class.shape[0]

            val_size = int(total_rows * val_ratio)
            test_size = int(total_rows * test_ratio)
            train_size = total_rows - val_size - test_size

            if train_size > 0:
                df_train.append(df_class[:train_size])
            if val_size > 0:
                df_val.append(df_class[train_size:train_size + val_size])
            if test_size > 0:
                df_test.append(df_class[train_size + val_size:])


    df_train = pd.concat(df_train, ignore_index=True)
    df_val = pd.concat(df_val, ignore_index=True)
    df_test = pd.concat(df_test, ignore_index=True)

    # Write the splits to new Parquet files
    df_train.to_parquet(output_train_path)
    df_val.to_parquet(output_val_path)
    df_test.to_parquet(output_test_path)

    # assert that they have no common ZTFIDs
    assert len(set(df_train['ZTFID']).intersection(set(df_val['ZTFID']))) == 0
    assert len(set(df_train['ZTFID']).intersection(set(df_test['ZTFID']))) == 0
    assert len(set(df_val['ZTFID']).intersection(set(df_test['ZTFID']))) == 0

    print(f"Split {input_path} ->")
    print(f"Train: {output_train_path} ({df_train.shape[0]} rows)")
    print(f"Validation: {output_val_path} ({df_val.shape[0]} rows)")
    print(f"Test: {output_test_path} ({df_test.shape[0]} rows)")

    # Print the class distributions
    print("\nClass distributions:")
    for split_name, split_df in zip(['Train', 'Validation', 'Test'], [df_train, df_val, df_test]):
        class_counts = split_df['bts_class'].value_counts()
        print(f"{split_name} set:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
        print()

if __name__ == "__main__":

    args = parse_args()

    split_parquet(
        input_path=args.input_path,
        output_train_path=args.output_train_path,
        output_val_path=args.output_val_path,
        output_test_path=args.output_test_path,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
