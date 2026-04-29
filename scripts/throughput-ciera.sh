#!/bin/bash
#SBATCH --account=b1094
#SBATCH --partition=ciera-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=25:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=52
#SBATCH --mem=90G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vedshah2029@u.northwestern.edu
#SBATCH --output=throughput-ciera.out
#SBATCH --error=throughput-ciera.out

cd /projects/b1094/ved/code/Hierarchical-VT/
module purge all
conda init bash
conda deactivate
source activate oracle2

# Final runs after HPO sweeps:

lscpu

python paper_figures/figure_throughput/throughput.py bts_oracle2
python paper_figures/figure_throughput/throughput.py bts_oracle2_lite
python paper_figures/figure_throughput/throughput.py bts_oracle2_omni
python paper_figures/figure_throughput/throughput.py image_backbone

python paper_figures/figure_throughput/throughput.py elasticc_oracle2
python paper_figures/figure_throughput/throughput.py elasticc_oracle2_lite









