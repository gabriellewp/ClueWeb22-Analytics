#!/bin/sh

#SBATCH --job-name=searchid       # The job name.
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --time=02-00:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/ivi/ilps/personal/gpoerwa/ClueWeb22-Analytics/output/output-%x-%J.log
#SBATCH -p cpu
#SBATCH --array=0-4             # Process 14 tasks, 8 at a time

source ~/.bashrc
conda activate test

export HF_HOME="/ivi/ilps/personal/gpoerwa/.cache/huggingface" \
export HF_DATASETS_CACHE="/ivi/ilps/personal/gpoerwa/.cache/huggingface/datasets" \
export TRANSFORMERS_CACHE="/ivi/ilps/personal/gpoerwa/.cache/huggingface/models" \
export HF_KEY="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" 

# Assign task_id from SLURM_ARRAY_TASK_ID
task_id=${SLURM_ARRAY_TASK_ID}

# # Define an array of directory names
DIR_NAMES=("en00" "en01" "en02" "en03" "en04" "en05" "en06" "en07" "en08" "en09" "en10" "en11" "en12" "en13" "en14" "en15" "en16" "en17" "en18" "en19" "en20" "en21" "en22" "en23" "en24" "en25" "en26" "en27" "en28" "en29" "en30" "en31" "en32" "en33" "en34" "en35" "en36" "en37" "en38" "en39" "en40" "en41" "en42" "en43" "en44" "en45" "en46" "en47" "en48" "en49") 
# DIR_NAMES=("en0000")
DIR_NAME=${DIR_NAMES[task_id]}

dir_path="/ivi/ilps/datasets/clueweb09/ClueWeb09_English_5/$DIR_NAME/"
echo "Processing directory: $dir_path"
python3 data_processing.py $dir_path


# FILES_GROUPS=("g1" "g2" "g3" "g4" "g5")
# FILE_GROUPS=${FILES_GROUPS[task_id]}
# echo "Processing file group: $FILE_GROUPS"
# python3 searchid.py $FILE_GROUPS

# python3 searchid.py "abc"