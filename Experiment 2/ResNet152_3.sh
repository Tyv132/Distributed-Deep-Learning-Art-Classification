#!/usr/bin/env bash
#SBATCH --account=ds--6013
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --nodes=1
#SBATCH --time=25:00:00

#SBATCH --job-name="ResNet152_3"
#SBATCH --output=output/ResNet152_3.out
#SBATCH --error=err/ResNet152_3.err
#SBATCH --mem=32GB

module purge
module load singularity/3.7.1 pytorch/1.12.0

# Train the model and get the validation accuracy after each epoch
singularity run --nv $CONTAINERDIR/pytorch-1.12.0.sif ./ResNet152.py --master_port 62003 --nodes 1 --gpus 3 --batch_size 36 --learning_rate 0.005 --train_size 76814 --val_size 9601 --test_size 9601 --epochs 6