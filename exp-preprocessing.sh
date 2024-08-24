#!/bin/bash

PARTITION="funky"
NODE="edwards"

DATASET_NAME=$1
DRY_RUN=$2
VIEW_PROMPT=$3

OUTPUT_DIR=./slurm_output
JOB_NAME="preprocessing"-$DATASET_NAME

if [ "$VIEW_PROMPT" == "true" ]; then
  python3 main.py --dataset_name $DATASET_NAME --stage preprocessing --view_prompt
else
  sbatch --nodes=1 --partition=$PARTITION --nodelist=$NODE --gres=gpu:1 --time=6-00:00:00 \
        --job-name $JOB_NAME --output $OUTPUT_DIR/$JOB_NAME.out --error $OUTPUT_DIR/$JOB_NAME.err \
        run_python.sh --dataset_name $DATASET_NAME --stage preprocessing \
                    --turn_id 1 --dry_run $DRY_RUN --gpu_partition ${PARTITION} --gpu_node $NODE
fi
