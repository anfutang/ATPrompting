import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import product
import pandas as pd
from langchain.output_parsers import PydanticOutputParser

from parser.parser_templates import *
from utils.process_example_funcs import *
# from ..opt import get_args

class Arg:
    def __init__(self,turn,stage,user_simulation_mode=None,prompt_type=None):
        self.turn = turn
        self.stage = stage
        if user_simulation_mode is not None:    
            self.user_simulation_mode = user_simulation_mode
        if prompt_type is not None:
            self.prompt_type = prompt_type

def get_parser(args):
    if args.stage == "preprocessing":
        pydantic_obj = IsAmbiguousCoT
    elif args.stage == "generation":
        if args.user_simulation_mode == "select":
            if "CoT" in args.prompt_type:
                pydantic_obj = RQCoTMultiple
            else:
                pydantic_obj = RQMultiple
        elif args.user_simulation_mode == "respond":
            if "CoT" in args.prompt_type:
                pydantic_obj = CQCoTSingle
            else:
                pydantic_obj = CQSingle
        else:
            if "CoT" in args.prompt_type:
                pydantic_obj = CQCoTMultiple
            else:
                pydantic_obj = CQMultiple
    elif args.stage == "response":
        if args.user_simulation_mode == "select":
            pydantic_obj = Select
        elif args.user_simulation_mode == "respond":
            pydantic_obj = Respond
        elif args.user_simulation_mode == "select+respond":
            pydantic_obj = SelectRespond
    elif args.stage == "reformulation":
        pydantic_obj = Reformulate
    return PydanticOutputParser(pydantic_object=pydantic_obj)

def turn_examples_to_pydantic_string(args):
    if args.turn == "multi_turn":
        if args.stage == "generation":
            if args.user_simulation_mode == "select":
                example_string = single_turn_generation_select(args)
            elif args.user_simulation_mode == "respond":
                example_string = multi_turn_generation_respond(args)
            elif args.user_simulation_mode == "select+respond":
                example_string = multi_turn_generation_select_respond(args)
        elif args.stage == "response":
            if args.user_simulation_mode == "select":
                example_string = single_turn_response_select()
            elif args.user_simulation_mode == "respond":
                example_string = multi_turn_response_respond()
            elif args.user_simulation_mode == "select+respond":
                example_string = multi_turn_response_select_respond()
        elif args.stage == "reformulation":
            if args.user_simulation_mode == "respond":
                example_string = multi_turn_reformulation_respond()
            elif args.user_simulation_mode == "select+respond":
                example_string = multi_turn_reformulation_select_respond()
    else:
        if args.stage == "preprocessing":
            example_string = single_turn_preprocessing()
        if args.stage == "generation":
            if args.user_simulation_mode == "select":
                example_string = single_turn_generation_select(args)
            elif args.user_simulation_mode == "respond":
                example_string = single_turn_generation_respond(args)
            elif args.user_simulation_mode == "select+respond":
                example_string = single_turn_generation_select_respond(args)
        if args.stage == "response":
            if args.user_simulation_mode == "select":
                example_string = single_turn_response_select()
            elif args.user_simulation_mode == "respond":
                example_string = single_turn_response_respond()
            elif args.user_simulation_mode == "select+respond":
                example_string = single_turn_response_select_respond()
        if args.stage == "reformulation":
            if args.user_simulation_mode == "respond":
                example_string = single_turn_reformulation_respond()
            elif args.user_simulation_mode == "select+respond":
                example_string = single_turn_reformulation_select_respond()
    return example_string

if __name__ == "__main__":
    turns = ["single_turn","multiple_turn"]
    user_simulation_modes = ["select","respond","select+respond"]
    prompt_types = ["few-shot","AT-few-shot","CoT-few-shot","AT-CoT-few-shot"]

    # argument_combs = ["single","preprocessing"] + list(product(turns,["generation"],user_simulation_modes,prompt_types)) + \
    #         list(product(turns,["response"],user_simulation_modes)) + list(product(turns,["reformulation"],user_simulation_modes))
    argument_combs = [('single_turn', 'preprocessing')] + \
                     list(product(["single_turn"],["generation"],user_simulation_modes,prompt_types)) + \
                     list(product(["single_turn"],["response"],user_simulation_modes)) + \
                     list(product(["single_turn"],["reformulation"],user_simulation_modes[1:])) + \
                     list(product(["multi_turn"],["generation"],user_simulation_modes,prompt_types)) + \
                     list(product(["multi_turn"],["response"],user_simulation_modes)) + \
                     list(product(["multi_turn"],["reformulation"],user_simulation_modes[1:]))
    
    fs_examples = {}
    for arg in argument_combs:
        tmp = fs_examples
        for a in arg[:-1]:
            if a not in tmp:
                tmp[a] = {}
            tmp = tmp[a]

    for comb in argument_combs:
        args = Arg(*comb)
        parser = get_parser(args)
        examples = turn_examples_to_pydantic_string(args)
        tmp = fs_examples
        for arg in comb[:-1]:
            tmp = tmp[arg]
        tmp[comb[-1]] = ''
        
        for input, output in examples:
            try:
                parser.parse(output)
                tmp[comb[-1]] += f"{input}\n{output}\n\n"
            except:
                raise ValueError(f"{comb}: unable to parse few-shot examples.")
        else:
            print(f"{comb}:finished.")
            
    dst_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',"few_shot_examples"))
    json.dump(fs_examples,open(os.path.join(dst_dir,"few_shot_examples.json"),'w'))
    
    print("succeded: building few-shot examples.")
    print(f"formatted few-shot examples saved to {dst_dir}.")


    