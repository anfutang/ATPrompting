#!bin/sh
DATASET_NAME=$1
STAGE=$2
TURN_ID=$3

USER_SIMULATION_MODES=("select" "respond" "select+respond")
PROMPT_TYPES=("few-shot" "AT-few-shot" "CoT-few-shot" "AT-CoT-few-shot")

if [ "$STAGE" == "reformulation" ]; then
  USER_SIMULATION_MODES=("${USER_SIMULATION_MODES[@]:1}")
fi

for usm in "${USER_SIMULATION_MODES[@]}"; do
  for pt in "${PROMPT_TYPES[@]}"; do
    python3 main.py --dataset_name $DATASET_NAME --stage $STAGE --user_simulation_mode ${usm} \
                    --prompt_type ${pt} --turn_id $TURN_ID --view_prompt
  done
done
