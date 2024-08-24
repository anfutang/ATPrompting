import os
import logging
import time
import numpy as np
import pandas as pd
import json
from opt import get_args
from loader.model import LLMLoader
from loader.prompt import PromptLoader
from loader.data import DataLoader
from utils.utility import *

if __name__ == "__main__":
    args = get_args()
    validate_arguments(args)

    logging_dir, output_dir = build_dst_folder(args)
    logging_filename = os.path.join(logging_dir,"log.log")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename=logging_filename,filemode='w')

    logging.info(show_job_infos(args))

    if args.dry_run:
        logging.info(f"DRY RUN mode: test on first {args.dry_run_number_of_examples} examples.")
    
    if args.view_prompt:
        logging.info(f"VIEW PROMPT mode: will only print out formatted prompts.")

    start_time = time.time()

    set_seed(args.seed)
    llm = LLMLoader(args)
    logging.info("LLM loaded.")

    prompt_template = PromptLoader(args,llm.tokenizer)
    logging.info("Prompt template loaded.")

    data = DataLoader(args).data
    logging.info("Data Loaded.")

    prompts = prompt_template.format(data)
    logging.info("Prompt formatted.")

    if not args.view_prompt:
        if args.dry_run:
            prompts = prompts[:args.dry_run_number_of_examples]
        else:
            logging.info(f"\t# prompts: {len(prompts)}.")

        outputs = llm.inference(prompts,prompt_template.parser)
        json.dump(outputs,open(os.path.join(output_dir,"output.json"),'w'))
        logging.info(f"Output saved under {output_dir}.")

        end_time = time.time()
        logging.info(f"Finished: {end_time-start_time} s.")