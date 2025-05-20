#!/bin/bash
#SBATCH -c 4 # Number of Cores per Task
#SBATCH --mem=8000  # Requested Memory
#SBATCH -p ceewater_kandread-cpu # Partition
#SBATCH -t 30:00:00  # Job time limit
#SBATCH -o /work/pi_kandread_umass_edu/lake_temp_bias/joboutputs/lstm_cloud_sim_log/sim_lst-%A_%a.out

source /home/xinchenhe_umass_edu/.bashrc
module load cuda/11.8
conda activate pytorch
python3 /work/pi_kandread_umass_edu/lake_temp_bias/satbias_model/satlswt/notebooks/C-lstm/D-simulate_lstm_cloud.py