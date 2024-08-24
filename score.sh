#!/bin/bash

PARTITION="funky"
NODELIST=("edwards" "rodgers" "bernard" "pascal")
NUM_NODES=${#NODELIST[@]}

DATASET_NAME=$1
SCORE_TYPE=$2
SCORE_STAGE=$3
TURN_ID=$4
DRY_RUN=$5

OUTPUT_DIR=./slurm_output

PROMPT_TYPES=("few-shot" "AT-few-shot" "CoT-few-shot" "AT-CoT-few-shot")

if [ "$SCORE_TYPE" == "cq" ]; then
  STAGE=generation
  USER_SIMULATION_MODES=("select+respond")
else
  USER_SIMULATION_MODES=("select" "respond" "select+respond")
fi

job_counter=0

for usm in "${USER_SIMULATION_MODES[@]}"; do
  NODE=${NODELIST[$job_counter]} 
  for pt in "${PROMPT_TYPES[@]}"; do
    JOB_NAME=score-${DATASET_NAME}-${pt}
    sbatch --nodes=1 --partition=$PARTITION --nodelist=$NODE --gres=gpu:1 --time=6-00:00:00 \
            --job-name $JOB_NAME --output $OUTPUT_DIR/$JOB_NAME.out --error $OUTPUT_DIR/$JOB_NAME.err \
            run_score.sh --dataset_name $DATASET_NAME --turn_id $TURN_ID  --stage score --user_simulation_mode ${usm} --prompt_type ${pt} \
                         --score_type $SCORE_TYPE --score_stage $SCORE_STAGE --dry_run $DRY_RUN --gpu_partition $PARTITION --gpu_node $NODE
  done
  job_counter=$((job_counter+1))
done

