#!/bin/bash -l

################################################################################
#Script Name: 
#Description: 
#Args       : None 
#Author     : Mukundan Thanigaivelan
#Email      : thanigam@union.edu
################################################################################

#SBATCH --job-name=train_uadage_unreg_models
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=thanigam@union.edu
#SBATCH --array=1-50%10
#SBATCH -o /scratch/user/u.mt372094/logs/%x-%A_%a.out
#SBATCH -e /scratch/user/u.mt372094/logs/%x-%A_%a.err

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate Conda environment
conda activate /scratch/user/u.mt372094/conda/envs/uadage-conda

# Environment configuration
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

python train_unreg_models.py ${SLURM_ARRAY_TASK_ID}