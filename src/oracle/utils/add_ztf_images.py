import polars as pl

main_path = "../../../data/BTSv3/"
ztf_img_data = '../../../ZTF_references_april26.parquet'

df_ztf_img = pl.read_parquet(ztf_img_data)
df_ztf_img = df_ztf_img.select(['g_reference', 'r_reference', 'i_reference', 'ZTFID'])

for split in ['train', 'validation', 'test']:

    print(f"Processing {split} split...")

    df_split = pl.read_parquet(f'{main_path}{split}_PS.parquet')

    # fail if there are any ZTFIDs in df_split that are not in df_ztf_img
    ztf_ids_split = set(df_split['ZTFID'].unique().to_list())
    ztf_ids_img = set(df_ztf_img['ZTFID'].unique().to_list())
    missing_ids = ztf_ids_split - ztf_ids_img
    if missing_ids:
        raise ValueError(f"Error: The following ZTFIDs are in {split}_PS.parquet but not in the ZTF image data: {missing_ids}")
    df_merged = df_split.join(df_ztf_img, on='ZTFID', how='left')

    df_merged.write_parquet(f'{main_path}{split}_PS_ZTF.parquet')