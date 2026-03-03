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
#SBATCH --output=test-ciera.out
#SBATCH --error=test-ciera.out

cd /projects/b1094/ved/code/Hierarchical-VT/
module purge all
conda init bash
conda deactivate
source activate oracle2

# Final inference runs after HPO sweeps:

# oracle-test models/BTSv2/peachy-sweep-4
# oracle-test models/BTSv2/rare-night-119
# oracle-test models/BTSv2/winter-meadow-120
# oracle-test models/BTSv2/rural-tree-121
# oracle-test models/BTSv2/resilient-feather-125

# oracle-test models/BTSv2-lite/gentle-sweep-1
# oracle-test models/BTSv2-lite/comfy-smoke-128
# oracle-test models/BTSv2-lite/deft-monkey-129
# oracle-test models/BTSv2-lite/flowing-feather-130
# oracle-test models/BTSv2-lite/stilted-dawn-131

# oracle-test models/BTSv2-pro/woven-sweep-9 --batch_size 64
# oracle-test models/BTSv2-pro/smooth-haze-141 --batch_size 64
# oracle-test models/BTSv2-pro/splendid-sun-142 --batch_size 64
# oracle-test models/BTSv2-pro/sweet-glade-143 --batch_size 64
# oracle-test models/BTSv2-pro/dazzling-thunder-144 --batch_size 64

# oracle-test models/ELAsTiCCv2/icy-violet-168 --max_n_per_class 1000
# oracle-test models/ELAsTiCCv2/dazzling-river-171 --max_n_per_class 1000
# oracle-test models/ELAsTiCCv2/rich-serenity-172 --max_n_per_class 1000
# oracle-test models/ELAsTiCCv2/fine-violet-173 --max_n_per_class 1000
# oracle-test models/ELAsTiCCv2/wandering-snowball-174 --max_n_per_class 1000

# oracle-test models/ELAsTiCCv2-lite/smart-river-169 --max_n_per_class 1000
# oracle-test models/ELAsTiCCv2-lite/playful-waterfall-170 --max_n_per_class 1000
# oracle-test models/ELAsTiCCv2-lite/happy-glade-175 --max_n_per_class 1000
# oracle-test models/ELAsTiCCv2-lite/bright-wood-176 --max_n_per_class 1000
# oracle-test models/ELAsTiCCv2-lite/whole-sun-177 --max_n_per_class 1000