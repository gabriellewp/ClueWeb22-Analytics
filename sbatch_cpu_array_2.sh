#!/bin/sh

#SBATCH --job-name=searchid       # The job name.
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --time=12-00:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/ivi/ilps/personal/gpoerwa/ClueWeb22-Analytics/output/output-%x-%J.log
#SBATCH -p cpu
#SBATCH --array=0-40             # Process 41 tasks

source ~/.bashrc
conda activate test

export HF_HOME="/ivi/ilps/personal/gpoerwa/.cache/huggingface" \
export HF_DATASETS_CACHE="/ivi/ilps/personal/gpoerwa/.cache/huggingface/datasets" \
export TRANSFORMERS_CACHE="/ivi/ilps/personal/gpoerwa/.cache/huggingface/models" \
export HF_KEY="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" 

# Assign task_id from SLURM_ARRAY_TASK_ID
task_id=${SLURM_ARRAY_TASK_ID}

NUM_ROWS=53509992
NUM_PROCESS=41

ROWS_PER_PROCESS=$((NUM_ROWS / NUM_PROCESS))
EXTRA_ROWS=$((NUM_ROWS % NUM_PROCESS))
echo $ROWS_PER_PROCESS
echo $EXTRA_ROWS

task_id=${SLURM_ARRAY_TASK_ID}

# Calculate the start and end for this task
START=$((task_id * ROWS_PER_PROCESS + (task_id < EXTRA_ROWS ? task_id : EXTRA_ROWS)))
END=$((START + ROWS_PER_PROCESS + (task_id < EXTRA_ROWS ? 1 : 0)))

echo "Task ID: $task_id"
echo "Processing rows $START to $((END))"

# Execute the Python script for this chunk
python3 searchid.py $START $END

