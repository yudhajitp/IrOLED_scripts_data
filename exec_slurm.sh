#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

export OMP_NUM_THREADS=1

source ~/.bashrc
#conda activate chemml_env
conda activate trial_env
#module load python/3.7

python mlp_model.py
#python lig_MEGNet_model.py
#python ligMEG_load_predict.py
#python mlp_load_predict.py
