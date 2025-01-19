#!/bin/sh

#SBATCH --job-name=cw_extract       # The job name.
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --time=00-20:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/ivi/ilps/personal/gpoerwa/ClueWeb22-Analytics/output/output-%x-%J.log
#SBATCH --error=/ivi/ilps/personal/gpoerwa/ClueWeb22-Analytics/output/error-%x-%J.log
#SBATCH -p cpu

source ~/.bashrc
conda activate test

export HF_HOME="/ivi/ilps/personal/gpoerwa/.cache/huggingface" \
export HF_DATASETS_CACHE="/ivi/ilps/personal/gpoerwa/.cache/huggingface/datasets" \
export TRANSFORMERS_CACHE="/ivi/ilps/personal/gpoerwa/.cache/huggingface/models" \
export HF_KEY="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" 
# export PYTHONPATH=$(pwd)
echo "/ivi/ilps/datasets/clueweb09/ClueWeb09_English_1/en0000/"
python searchid.py /ivi/ilps/datasets/clueweb09/ClueWeb09_English_1/en0000/