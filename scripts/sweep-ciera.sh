#!/bin/bash
#SBATCH --account=b1094
#SBATCH --partition=ciera-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=52
#SBATCH --mem=90G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vedshah2029@u.northwestern.edu
#SBATCH --output=sweep-ciera.out
#SBATCH --error=sweep-ciera.out

cd /projects/b1094/ved/code/Hierarchical-VT/
module purge all
conda init bash
conda deactivate
source activate oracle2

# oracle-sweep BTSv2_PSonly --count 10
# oracle-sweep BTSv2 --count 10
# oracle-sweep BTSv2-lite --count 10
oracle-sweep BTSv2-pro-free-gamma --count 10
# oracle-sweep ELAsTiCCv2-lite --count 10
# oracle-sweep ELAsTiCCv2 --count 10