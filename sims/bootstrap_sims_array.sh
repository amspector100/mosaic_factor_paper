#!/usr/bin/bash
#SBATCH --ntasks 1
#SBATCH --array=1-2000:50
#SBATCH --job-name=bootstrap_sim
#SBATCH --output=../log/bootstrap/sim_%A_%a.out
#SBATCH --error=../log/bootstrap/sim_%A_%a.err
#SBATCH --time=23:59:00
#SBATCH -p candes 
#SBATCH -c 1
#SBATCH --mem=4GB

NREPS=50 # make sure this is the same as the job array step size
NPROC=1

MAIN_ARGS="
        --industry [FIN]
        --n [300]
        --center_date [covid,2021,2022,2023]
        --reps $NREPS
        --num_processes $NPROC
        --seed_start $SLURM_ARRAY_TASK_ID
        --job_id $SLURM_ARRAY_JOB_ID
"


source /home/users/aspector/mosaic/setup_env.sh
python bootstrap_sims.py $MAIN_ARGS
