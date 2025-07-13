#!/bin/bash

#SBATCH --job-name=ss-llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.thamma@student.vu.nl
#SBATCH -t 6:45:00

# srun  --nodes=1 --partition=gpu --gpus-per-node=4 -t 0:5:00 --pty /bin/bash
# sbatch --export=config_file_name=wikipedia_bpe/train_wikipedia_gpt_exp2.py test_run_job.sh

echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

module load 2021
module load Python/3.9.5-GCCcore-10.3.0

source $HOME/repo/mt1_p39/bin/activate

cd $HOME/repo/ss-llm/nanoGPT

torchrun --standalone --nproc_per_node=4 train.py config/$config_file_name

