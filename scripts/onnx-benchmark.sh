#!/bin/bash
#SBATCH --account=b1094
#SBATCH --partition=ciera-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=25:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=90G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vedshah2029@u.northwestern.edu
#SBATCH --output=onnx-benchmark-%j.out
#SBATCH --error=onnx-benchmark-%j.out

set -e

cd /projects/b1094/ved/code/Hierarchical-VT/
module purge all

echo "Running ONNX benchmark with $HOME/.conda/envs/oracle2/bin/python"
exec "$HOME/.conda/envs/oracle2/bin/python" -u boom_scripts/ONNX_benchmark.py "$@"
