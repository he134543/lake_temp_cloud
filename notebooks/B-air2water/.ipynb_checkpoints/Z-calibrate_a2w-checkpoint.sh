#!/bin/bash
#SBATCH -c 10 # Number of Cores per Task
#SBATCH --mem=12000  # Requested Memory
#SBATCH -p cpu-preempt # Partition
#SBATCH -t 30:00:00  # Job time limit
#SBATCH -o /work/pi_kandread_umass_edu/lake_temp_bias/joboutputs/a2w_cal-%A_%a.out

source /home/xinchenhe_umass_edu/.bashrc
conda activate laketemp
python3 /work/pi_kandread_umass_edu/lake_temp_bias/satbias_model/satlswt/notebooks/C-air2water/C-calibrate_air2water.py