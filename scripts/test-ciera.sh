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

# oracle-test models/BTSv2/polar-sky-593
# oracle-test models/BTSv2/legendary-frost-594
# oracle-test models/BTSv2/lucky-thunder-595
# oracle-test models/BTSv2/vague-thunder-596
# oracle-test models/BTSv2/fancy-sponge-597

# oracle-test models/BTSv2-lite/fast-aardvark-588
# oracle-test models/BTSv2-lite/ethereal-water-589
# oracle-test models/BTSv2-lite/dstellar-morning-590
# oracle-test models/BTSv2-lite/blooming-rain-591
# oracle-test models/BTSv2-lite/robust-resonance-592

# oracle-test models/BTSv2-pro/morning-feather-610 --batch_size 64
# oracle-test models/BTSv2-pro/summer-feather-611 --batch_size 64
# oracle-test models/BTSv2-pro/vivid-monkey-612 --batch_size 64
# oracle-test models/BTSv2-pro/desert-cherry-613 --batch_size 64
# oracle-test models/BTSv2-pro/celestial-jazz-614 --batch_size 64

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