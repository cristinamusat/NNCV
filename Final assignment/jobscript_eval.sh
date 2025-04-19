#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:10:00

srun apptainer exec --nv --env-file .env container.sif python eval.py