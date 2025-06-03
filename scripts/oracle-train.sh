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

oracle-train --model ORACLE2-lite_swin_LSST