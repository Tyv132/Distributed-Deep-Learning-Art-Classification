#!/usr/bin/env bash
#SBATCH --account=ds--6013
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --time=7:00:00

#SBATCH --job-name="8"
#SBATCH --output=output/8.out
#SBATCH --error=err/8.err
#SBATCH --mem=16GB

module purge
module load singularity/3.7.1 pytorch/1.12.0


# Train the model and get the validation accuracy after each epoch
singularity run --nv $CONTAINERDIR/pytorch-1.12.0.sif ./DDP.py --master_port 65208 --gpus 1 --batch_size 64 --learning_rate 0.01 --train_size 9601 --val_size 9601 --test_size 76814 --epochs 7