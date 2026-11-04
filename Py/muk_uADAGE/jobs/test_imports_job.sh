#!/bin/bash -l

################################################################################
#Script Name: 
#Description: 
#Args       : None 
#Author     : Mukundan Thanigaivelan
#Email      : thanigam@union.edu
################################################################################

#SBATCH --job-name=uadage_test_imports
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=thanigam@union.edu
#SBATCH -o /scratch/user/u.mt372094/logs/%x-%j.out
#SBATCH -e /scratch/user/u.mt372094/logs/%x-%j.err

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate the EXACT environment used interactively
conda activate /scratch/user/u.mt372094/conda/envs/uadage-conda

# Environment configuration
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd $SLURM_SUBMIT_DIR

python test_imports.py