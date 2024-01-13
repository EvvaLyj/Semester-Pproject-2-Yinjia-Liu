#!/bin/bash
#SBATCH --mem=40G
#SBATCH --output=slurm/%j.out
#SBATCH --gres=gpu:1
#SBATCH --account=student
#SBATCH --mail-type=fail
#SBATCH --mail-user=yinjliu@ethz.ch

# source activate simclr
export LD_LIBRARY_PATH=/scratch_net/biwidl206/yinjliu/miniconda3/lib:$LD_LIBRARY_PATH
python run_eval.py --eval-epochs 500 \
                    --eval-download True \
                    --projectname simclr_eval \
                    --folder folder \


