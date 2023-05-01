#!/usr/bin/env bash
#SBATCH --account=ds--6013
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --time=7:00:00

#SBATCH --job-name="1"
#SBATCH --output=output/1.out
#SBATCH --error=err/1.err
#SBATCH --mem=16GB

module purge
module load singularity/3.7.1 pytorch/1.12.0


# Train the model and get the validation accuracy after each epoch
singularity run --nv $CONTAINERDIR/pytorch-1.12.0.sif ./DDP.py --master_port 65201 --gpus 1 --batch_size 32 --learning_rate 0.005 --train_size 76814 --val_size 9601 --test_size 9601 --epochs 7