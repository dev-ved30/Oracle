#!/bin/bash
#SBATCH --account=p32795
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=90G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vedshah2029@u.northwestern.edu
#SBATCH --output=oracle-train-gen.out
#SBATCH --error=oracle-train-gen.out


cd /projects/b1094/ved/code/Hierarchical-VT/
module purge all
conda init bash
conda deactivate
source activate oracle2

#pip install -e .

oracle-train --model ORACLE1 --num_epochs 2000 --alpha 0.5 --lr 0.0002