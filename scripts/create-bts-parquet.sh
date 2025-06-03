#!/bin/bash
#SBATCH --account=b1094
#SBATCH --partition=ciera-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=52
#SBATCH --mem=90G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=build-bts-parquet.out
#SBATCH --error=build-bts-parquet.err
#SBATCH --mail-user=vedshah2029@u.northwestern.edu


python src/oracle/utils/bts_csv_to_parquet.py --labels_csv_path data/BTS/bts_labels.txt --data_csv_path /projects/b1094/rehemtulla/BTSbot/data/train_cand_v11.csv  --images_np_path /projects/b1094/rehemtulla/BTSbot/data/train_triplets_v11.npy --output_path data/BTS/train.parquet
python src/oracle/utils/bts_csv_to_parquet.py --labels_csv_path data/BTS/bts_labels.txt --data_csv_path /projects/b1094/rehemtulla/BTSbot/data/test_cand_v11.csv --images_np_path /projects/b1094/rehemtulla/BTSbot/data/test_triplets_v11.npy --output_path data/BTS/test.parquet
python src/oracle/utils/bts_csv_to_parquet.py --labels_csv_path data/BTS/bts_labels.txt --data_csv_path /projects/b1094/rehemtulla/BTSbot/data/val_cand_v11.csv --images_np_path /projects/b1094/rehemtulla/BTSbot/data/val_triplets_v11.npy --output_path data/BTS/val.parquet
