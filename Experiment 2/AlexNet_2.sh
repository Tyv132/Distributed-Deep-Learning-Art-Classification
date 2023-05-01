#!/usr/bin/env bash
#SBATCH --account=ds--6013
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --time=25:00:00

#SBATCH --job-name="AlexNet2"
#SBATCH --output=output/AlexNet2.out
#SBATCH --error=err/AlexNet2.err
#SBATCH --mem=32GB

module purge
module load singularity/3.7.1 pytorch/1.12.0

# Train the model and get the validation accuracy after each epoch
singularity run --nv $CONTAINERDIR/pytorch-1.12.0.sif ./AlexNet.py --master_port 64002 --nodes 1 --gpus 2 --batch_size 54 --learning_rate 0.005 --train_size 76814 --val_size 9601 --test_size 9601 --epochs 6