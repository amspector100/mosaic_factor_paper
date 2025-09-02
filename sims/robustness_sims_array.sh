#!/usr/bin/bash
#SBATCH --ntasks 1
#SBATCH --array=1-4000:20
#SBATCH --job-name=mpt_sim
#SBATCH --output=../log/robustness/sim_%A_%a.out
#SBATCH --error=../log/robustness/sim_%A_%a.err
#SBATCH --time=23:59:00
#SBATCH -p candes,stat,hns,normal
#SBATCH -c 1
#SBATCH --mem=4GB

NREPS=20 # make sure this the same as the job array step size
NPROC=1

MAIN_ARGS="
        --sampling_method [ar1,garch,garch_ar1,mvn_arch]
        --industry [EGY,FIN,HLC]
        --n [300]
        --reps $NREPS
        --num_processes $NPROC
        --seed_start $SLURM_ARRAY_TASK_ID
        --job_id $SLURM_ARRAY_JOB_ID
"


source /home/users/aspector/mosaic/setup_env.sh
python robustness_sims.py $MAIN_ARGS
