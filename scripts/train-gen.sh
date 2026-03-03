#!/bin/bash
#SBATCH --account=p32795
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=25:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=90G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vedshah2029@u.northwestern.edu
#SBATCH --output=train-gen.out
#SBATCH --error=train-gen.out


cd /projects/b1094/ved/code/Hierarchical-VT/
source /hpc/software/mamba/23.1.0/etc/profile.d/conda.sh
conda activate oracle2

# pip install -e .

# Final runs after HPO sweeps:

# oracle-train BTSv2 --lr 0.00045076 --batch_size 64 --num_epochs 2000 --alpha 0.311
# oracle-train BTSv2-lite --lr 0.00025514 --batch_size 64 --num_epochs 2000 --alpha 0.3419
# oracle-train BTSv2-pro --lr 0.0001363 --batch_size 32 --num_epochs 2000 --alpha 0.069757 --warmup_epochs 100
# oracle-train ELAsTiCCv2 --max_n_per_class 20000 --lr 2e-4 --batch_size 1024 --num_epochs 2000 --alpha 0.5
# oracle-train ELAsTiCCv2-lite --max_n_per_class 20000 --lr 2e-4 --batch_size 1024 --num_epochs 2000 --alpha 0.5