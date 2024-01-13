#!/bin/bash
#SBATCH --mem=40G
#SBATCH --output=slurm/%j.out
#SBATCH --gres=gpu:1
#SBATCH --account=student
#SBATCH --mail-type=fail
#SBATCH --mail-user=yinjliu@ethz.ch

# An example shell script to perform the encoder training with method A

export LD_LIBRARY_PATH=/scratch_net/biwidl206/yinjliu/miniconda3/lib:$LD_LIBRARY_PATH
python run.py \
    -data ./datasets \
    -dataset-name cifar10 \
    --epochs 1000  \
    --warmup-epoch 10 \
    --aug-type amp_gm_v1_phs_TI_square_control \
    --use-simclr 1 \
    --use-fftclr 1 \
    --prob-transform 0.5 \
    --max-coeff-amp 0.5 \
    --max-coeff-phs 0.5 \
    --seed 229 \
    --projectname simclrA \