import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from parser.parser_templates import *
from .prompt_templates import *
from .system_instructions import SystemInstruction

logger = logging.getLogger(__name__)

split_str = '\n'*3 + '='*10 + '\n'*3

class PromptLoader:
    def __init__(self,args,tokenizer):
        self.view_prompt = args.view_prompt
        self.prompt_dir = args.prompt_dir
        self.dataset_name = args.dataset_name
        self.stage = args.stage
        self.noise_type = args.noise_type
        self.prompt_type = args.prompt_type
        self.user_simulation_mode = args.user_simulation_mode
        self.turn_id = args.turn_id
        self.tokenizer = tokenizer
        self.load_few_shot_examples()
        self.get_prompt_template() 
        self.get_parser()
        self.get_system_instruction(args)
        self.get_few_shot_examples()
        self.final_prompt_template = PromptTemplate(template=self.prompt_template,
                                                    input_variables=self.input_variables,
                                                    partial_variables={"system_instruction":self.system_instruction,
                                                                       "format_instruction": self.parser.get_format_instructions(),
                                                                       "few_shot_examples":self.few_shot_examples})
    
    def get_prompt_template(self):
        if self.stage in ["preprocessing","generation"]:    
            if self.turn_id > 1:
                if self.user_simulation_mode == "select":
                    self.prompt_template = generation_template
                    self.input_variables = ["query"]
                else:
                    self.prompt_template = multi_turn_generation_template
                    self.input_variables = ["chat history"]
            else:
                self.prompt_template = generation_template
                self.input_variables = ["query"]
        elif self.stage == "response":
            self.prompt_template = response_template
            self.input_variables = ["chat_history","user_intention"]
        elif self.stage == "reformulation":
            self.prompt_template = reformulation_template
            self.input_variables = ["chat history"]
        self.prompt_template = self.tokenizer.apply_chat_template(self.prompt_template,add_generation_prompt=True,tokenize=False)

    def get_parser(self):
        if self.stage == "preprocessing":
            pydantic_obj = IsAmbiguousCoT
        elif self.stage == "generation":
            if self.user_simulation_mode == "select":
                if "CoT" in self.prompt_type:
                    pydantic_obj = RQCoTMultiple
                else:
                    pydantic_obj = RQMultiple
            elif self.user_simulation_mode == "respond":
                if "CoT" in self.prompt_type:
                    pydantic_obj = CQCoTSingle
                else:
                    pydantic_obj = CQSingle
            else:
                if "CoT" in self.prompt_type:
                    pydantic_obj = CQCoTMultiple
                else:
                    pydantic_obj = CQMultiple
        elif self.stage == "response":
            if self.user_simulation_mode == "select":
                pydantic_obj = Select
            elif self.user_simulation_mode == "respond":
                pydantic_obj = Respond
            elif self.user_simulation_mode == "select+respond":
                pydantic_obj = SelectRespond
        elif self.stage == "reformulation":
            pydantic_obj = Reformulate
        
        self.parser = PydanticOutputParser(pydantic_object=pydantic_obj)

    def get_system_instruction(self,args):
        self.system_instruction = SystemInstruction(args).instruction

    def get_few_shot_examples(self):
        tmp = self.all_few_shot_examples
        if self.turn_id > 1:
            tmp = tmp["multi_turn"]
        else:
            tmp = tmp["single_turn"]
        tmp = tmp[self.stage]
        if self.stage != "preprocessing":
            tmp = tmp[self.user_simulation_mode]
        if self.stage == "generation":
            tmp = tmp[self.prompt_type]
        self.few_shot_examples = tmp
        
    def load_few_shot_examples(self):
        all_few_shot_example_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',"few_shot_examples"))
        self.all_few_shot_examples = json.load(open(os.path.join(all_few_shot_example_dir,"few_shot_examples.json")))

    def format(self,data):
        # in case of response, iterate over chat history x user intention
        prompts = []
        if self.stage in ["preprocessing","generation"]:    
            if self.turn_id > 1:
                if self.user_simulation_mode == "select":
                    for q in data["query"]:
                        prompts.append(self.final_prompt_template.format_prompt(query=q).text)
                else:
                    for ch in data["chat_history"]:
                        prompts.append(self.final_prompt_template.format_prompt(chat_history=ch).text)
            else:
                for q in data["query"]:
                    prompts.append(self.final_prompt_template.format_prompt(query=q).text)
        elif self.stage == "response":
            if self.turn_id > 1:
                for ch, ui in zip(data["chat_history"],data["user_intention"]):
                    prompts.append(self.final_prompt_template.format_prompt(chat_history=ch,user_intention=ui).text)
            else:
                if self.user_simulation_mode == "select":
                    for q, uis, rqs in zip(data["query"],data["user_intention"],data["reformulated_query"]):
                        for ui in uis:
                            formatted_rqs = '\n'.join([f"({i+1}) {rqs[i]}" for i in range(len(rqs))])
                            chat_history = f"Query: {q}"+'\n'+f"List of reformulated queries: {formatted_rqs}"
                            prompts.append(self.final_prompt_template.format_prompt(chat_history=chat_history,user_intention=ui).text)
                elif self.user_simulation_mode == "respond":
                    for q, uis, cq in zip(data["query"],data["user_intention"],data["clarification_question"]):
                        for ui in uis:
                            chat_history = f"Query: {q}"+'\n'+f"Clarification question: {cq}"
                            prompts.append(self.final_prompt_template.format_prompt(chat_history=chat_history,user_intention=ui).text)
                elif self.user_simulation_mode == "select+respond":
                    for q, uis, cqs in zip(data["query"],data["user_intention"],data["clarification_question"]):
                        for ui in uis:
                            formatted_cqs = '\n'.join([f"({i+1}) {cqs[i]}" for i in range(len(cqs))])
                            chat_history = f"Query: {q}"+'\n'+f"List of clarification questions: {formatted_cqs}"
                            prompts.append(self.final_prompt_template.format_prompt(chat_history=chat_history,user_intention=ui).text)
        elif self.stage == "reformulation":
            for ch in data["chat_history"]:
                prompts.append(self.final_prompt_template.format_prompt(chat_history=ch).text)

        if self.view_prompt:
            if self.stage == "preprocessing":
                prompt_dir = os.path.join(self.prompt_dir,self.dataset_name,f"noise_type_{self.noise_type}",f"turn_{self.turn_id}",self.stage)
            else:
                prompt_dir = os.path.join(self.prompt_dir,self.dataset_name,f"noise_type_{self.noise_type}",f"turn_{self.turn_id}",self.stage,self.user_simulation_mode,self.prompt_type)
            with open(os.path.join(prompt_dir,f"prompts.txt"),'w') as f:
                f.write(split_str.join(prompts))
            logger.info(f"VIEW PROMPT mode: prompt texts are printed out and saved under {prompt_dir}.")
        return prompts