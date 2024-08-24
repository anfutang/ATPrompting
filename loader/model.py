import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from parser.parser import parse

model_name_to_huggingface_ids = {"llama3-8b":"meta-llama/Meta-Llama-3-8B-Instruct",
                                 "mistral-7b-v0.2":"mistralai/Mistral-7B-Instruct-v0.2"}

logger = logging.getLogger(__name__)

class LLMLoader:
    def __init__(self,args):
        self._get_llm_kwargs(args)
        self.model_name = args.model_name
        self.view_prompt = args.view_prompt
        self.maximum_retry_times = args.maximum_retry_times
        self.load(args.model_name)

    def _get_llm_kwargs(self,args):
        kwargs = {"max_new_tokens":args.max_new_tokens}
        if args.no_sampling:
            kwargs["do_sample"] = False
            self.llm_kwargs = kwargs
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = args.temperature
            if args.sampling_strategy == "top_k":
                kwargs["top_k"] = args.top_k
            elif args.sampling_strategy == "top_p":
                kwargs["top_p"] = args.top_p
            else:
                raise ValueError("Invalid sampling strategy: should be 'top_k' or 'top_p'.")  
            self.llm_kwargs = kwargs

    def load(self,model_name):
        model_id = model_name_to_huggingface_ids[model_name]

        tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        if not self.view_prompt:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            terminators = [tokenizer.eos_token_id]
            if "llama3" in model_name:
                terminators += [tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                eos_token_id=terminators,
                **self.llm_kwargs
            )

            # llm = HuggingFacePipeline(pipeline=pipeline)
            self.llm = pipeline

    def get_retry_info(self,retry_id):
        if retry_id == self.maximum_retry_times:
            return f"Last retry"
        if retry_id == 1:
            return '*'*5+"First retry"
        if retry_id == 2:
            return '*'*5+"Second retry"
        if retry_id == 3:
            return '*'*5+"Third retry"
        return '*'*5+f"{retry_id}-th retry"
        
    def inference(self,prompts,parser,retry_id=0,last_outputs=None):
        if retry_id > self.maximum_retry_times:
            logging.info(f"Maximum retry attempts reached: stop; {len(prompts)} remained parsing errors.")
            return last_outputs
        
        if retry_id > 0:
            retry_info = self.get_retry_info(retry_id)
            logging.info(retry_info+f": search to handle {len(prompts)} parsing errors.")

            last_error_ixs = last_outputs["error_indexes"]

            if len(last_error_ixs) < 30:
                logging.info('*'*5+f"Remaining error indexes: {last_error_ixs}")
            else:
                logging.info('*'*5+f"Remaining error indexes: {last_error_ixs[:30]}...")

        if retry_id == self.maximum_retry_times:
            logging.info(f"\tLast retry: set do_sample=False")
            docs = self.llm(prompts,do_sample=False)
        else:
            docs = self.llm(prompts)
        logger.info("LLM inference finished.")
        outputs = parse(docs,parser,self.model_name)
        error_ixs = outputs["error_indexes"]

        # if retry, add results of succesfully resolved examples that caused parsing errors
        if retry_id > 0 and len(error_ixs) != len(prompts):
            for ix in range(len(prompts)):
                if ix not in error_ixs:
                    last_outputs["output"][last_error_ixs[ix]] = outputs["output"][ix]

        if len(error_ixs) == 0:
            if retry_id == 0:
                return outputs
            else:
                last_outputs["error_indexes"] = []
                return last_outputs
        else:
            prompts = [prompts[ix] for ix in error_ixs]
            if retry_id == 0:
                return self.inference(prompts,parser,retry_id=1,last_outputs=outputs)    
            else:
                last_outputs["error_indexes"] = [last_error_ixs[ix] for ix in error_ixs]
                return self.inference(prompts,parser,retry_id=retry_id+1,last_outputs=last_outputs)
