#!/usr/bin/bash
#SBATCH --ntasks 1
#SBATCH --array=1-500:2
#SBATCH --job-name=mpt_sim
#SBATCH --output=../log/mpt_sim_%A_%a.out
#SBATCH --error=../log/mpt_sim_%A_%a.err
#SBATCH --time=23:59:00
#SBATCH -p candes 
#SBATCH -c 1
#SBATCH --mem=4GB

NREPS=2 # make sure this is the same as the job array step size
NPROC=1


# todo:
# add back test_stat = [maxcorr,mcpi,invtrace,fisher,sparse_fisher]
MAIN_ARGS="
        --n [100,150]
        --sparsity [0.05,0.15,0.5]
        --rho [0,1,2,3,4,5,6,7] 
        --reps $NREPS
        --num_processes $NPROC
        --seed_start $SLURM_ARRAY_TASK_ID
        --job_id $SLURM_ARRAY_JOB_ID
"

module load python/3.9
module load py-numpy/1.20.3_py39
python3.9 mpt_sim.py $MAIN_ARGS