#!/bin/bash

PARTITION="funky"
NODELIST=("edwards" "rodgers" "bernard" "pascal")
# NODELIST=("bernard")
NUM_NODES=${#NODELIST[@]}

DATASET_NAME=$1
STAGE=$2
TURN_ID=$3
DRY_RUN=$4

OUTPUT_DIR=./slurm_output

USER_SIMULATION_MODES=("select" "respond" "select+respond")
# USER_SIMULATION_MODES=("select")
PROMPT_TYPES=("few-shot" "AT-few-shot" "CoT-few-shot" "AT-CoT-few-shot")

if [ "$STAGE" == "reformulation" ]; then
  USER_SIMULATION_MODES=("${USER_SIMULATION_MODES[@]:1}")
fi

job_counter=0

for usm in "${USER_SIMULATION_MODES[@]}"; do
  NODE=${NODELIST[$job_counter]}
  for pt in "${PROMPT_TYPES[@]}"; do
    JOB_NAME=${usm}-${pt}
    sbatch --nodes=1 --partition=$PARTITION --nodelist=$NODE --gres=gpu:1 --time=6-00:00:00 \
           --job-name $JOB_NAME --output $OUTPUT_DIR/$JOB_NAME.out --error $OUTPUT_DIR/$JOB_NAME.err \
           run_python.sh --dataset_name $DATASET_NAME --stage $STAGE --user_simulation_mode ${usm} --prompt_type ${pt} \
                       --turn_id $TURN_ID --dry_run $DRY_RUN --gpu_partition ${PARTITION} --gpu_node $NODE
  done
  job_counter=$((job_counter+1))
done

