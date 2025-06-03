#!/bin/bash
#SBATCH --account=b1094
#SBATCH --partition=ciera-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=52
#SBATCH --mem=90G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vedshah2029@u.northwestern.edu

cd /projects/b1094/ved/code/Hierarchical-VT/
module purge all
conda deactivate
source activate oracle2

pip install -e .

oracle-train --model ORACLE1-lite_BTS --lr 1e-3 --alpha 0.5 --batch_size 256 --num_epochs 1000 --max_n_per_class 1000