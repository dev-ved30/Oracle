import polars as pl
from pathlib import Path

# Path to this file's directory
here = Path(__file__).resolve().parent

BTS_train_parquet_path = str(here.parent.parent.parent / "data" / 'BTSv3' / 'train_PS.parquet')
BTS_test_parquet_path = str(here.parent.parent.parent / "data" / 'BTSv3' / 'test_PS.parquet')
BTS_val_parquet_path = str(here.parent.parent.parent / "data" / 'BTSv3' / 'val_PS.parquet')

ztf_img_data = str(here.parent.parent.parent / "ZTF_references_april26.parquet")

df_ztf_img = pl.read_parquet(ztf_img_data)
df_ztf_img = df_ztf_img.select(['g_reference', 'r_reference', 'i_reference', 'ZTFID'])

for split in [BTS_test_parquet_path, BTS_train_parquet_path, BTS_val_parquet_path]:

    df_split = pl.read_parquet(split)

    # fail if there are any ZTFIDs in df_split that are not in df_ztf_img
    ztf_ids_split = set(df_split['ZTFID'].unique().to_list())
    ztf_ids_img = set(df_ztf_img['ZTFID'].unique().to_list())
    missing_ids = ztf_ids_split - ztf_ids_img
    if missing_ids:
        print(f"Error: The following ZTFIDs are in {split} but not in {ztf_img_data}: {missing_ids}")
        print(f"Number of missing ZTFIDs: {len(missing_ids)/len(ztf_ids_split)*100:.2f}% of the split")
    df_merged = df_split.join(df_ztf_img, on='ZTFID', how='left')

    df_merged.write_parquet(split.replace('.parquet', '_ZTF.parquet'))