#!/usr/bin/bash
#SBATCH --ntasks 1
#SBATCH --array=1-2000:10
#SBATCH --job-name=mpt_sim
#SBATCH --output=../log/bootstrap/sim%A_%a.out
#SBATCH --error=../log/bootstrap/sim_%A_%a.err
#SBATCH --time=23:59:00
#SBATCH -p candes 
#SBATCH -c 1
#SBATCH --mem=4GB

NREPS=10 # make sure this is the same as the job array step size
NPROC=1

MAIN_ARGS="
        --industry [FIN]
        --n [300]
        --reps $NREPS
        --num_processes $NPROC
        --seed_start $SLURM_ARRAY_TASK_ID
        --job_id $SLURM_ARRAY_JOB_ID
"


module load python/3.9
module load py-numpy/1.20.3_py39
python3.9 bootstrap_sims.py $MAIN_ARGS
