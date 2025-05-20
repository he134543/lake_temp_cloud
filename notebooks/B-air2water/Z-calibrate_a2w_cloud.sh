#!/bin/bash
#SBATCH -c 12 # Number of Cores per Task
#SBATCH --mem=12000  # Requested Memory
#SBATCH -p cpu-preempt # Partition
#SBATCH -t 30:00:00  # Job time limit
#SBATCH -o /work/pi_kandread_umass_edu/lake_temp_bias/joboutputs/a2w_cloud/a2w_cal-%A_%a.out

source /home/xinchenhe_umass_edu/.bashrc
conda activate /work/xinchenhe_umass_edu/.conda/envs/laketemp
python3 /work/pi_kandread_umass_edu/lake_temp_bias/satbias_model/satlswt/notebooks/C-air2water/C-calibrate_air2water_cloud.py