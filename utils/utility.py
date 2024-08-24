import os
import datetime
import numpy as np
import random
import torch

valid_prompt_types = ["zero-shot","few-shot","AT-zero-shot","AT-few-shot","CoT-zero-shot","CoT-few-shot","AT-CoT-zero-shot","AT-CoT-few-shot"]

def validate_arguments(args):
    assert os.path.exists(args.data_dir), "input data does not exist."
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    if not os.path.exists(args.logging_dir):
        os.makedirs(args.logging_dir,exist_ok=True)
    if not os.path.exists(args.prompt_dir):
        os.makedirs(args.prompt_dir,exist_ok=True)
    assert args.turn_id >= 1, "turn id must be superior or equal to 1."
    assert args.stage in ["preprocessing","generation","response","reformulation"], "INVALID stage."
    assert args.user_simulation_mode in ["select","respond","select+respond",""], "INVALID user simulation mode."
    assert args.prompt_type in valid_prompt_types, "INVALID prompt type."
    if args.stage == "preprocessing":
        assert args.turn_id == 1, "no multi-turn dialog for preprocessing."
    if args.stage == "reformulation":
        assert args.user_simulation_mode != "select", "UNMATCH: user simulation mode 'SELECT' does not need reformulation."

def build_dst_folder(args):
    if args.stage == "preprocessing":
        comb = [args.dataset_name,f"turn_{args.turn_id}",args.stage]
    else:
        comb = [args.dataset_name,f"turn_{args.turn_id}",args.stage,args.user_simulation_mode,args.prompt_type]

    if args.view_prompt:
        curr_logging_dir, curr_output_dir = args.prompt_dir, args.prompt_dir
    else:
        curr_logging_dir, curr_output_dir = args.logging_dir, args.output_dir

    for c in comb:
        curr_logging_dir = os.path.join(curr_logging_dir,c)
        if not os.path.exists(curr_logging_dir):
            os.makedirs(curr_logging_dir,exist_ok=True)
        curr_output_dir = os.path.join(curr_output_dir,c)
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir,exist_ok=True)
    
    return curr_logging_dir, curr_output_dir

def show_job_infos(args):
    infos = []
    infos.append('\n'+'='*5 + "Job information" + '='*5)
    infos.append("*Start time:"+str(datetime.datetime.now()))
    infos.append(f"*Gpu: {args.gpu_partition}-{args.gpu_node}")
    infos.append(f"*Dataset: {args.dataset_name}")
    infos.append(f"*Turn: {args.turn_id}")
    infos.append(f"*Stage: {args.stage}")
    infos.append(f"*Maximum retry tims: {args.maximum_retry_times}")
    if args.stage != "preprocessing":
        infos.append(f"*User simulation mode: {args.user_simulation_mode}")
        infos.append(f"*Prompt type: {args.prompt_type}")
    infos.append('='*10+'\n')
    return '\n'.join(infos)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
