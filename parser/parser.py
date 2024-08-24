import os
import logging

logger = logging.getLogger(__name__)

eoi_token = {"llama3-8b":"assistant<|end_header_id|>",
             "mistral-7b-v0.2":"[/INST]",
             }

def strip_to_wrapped_by_brackets(s):
    if s.count('{') == 1 and s.count('}') == 1:
        start_ix = s.index('{')
        end_ix = s.index('}')
        return s[start_ix:end_ix+1]
    elif '{' in s and '}' in s:
        start_ix_found, end_ix_found = False, False
        for ix in range(len(s)-1,-1,-1):
            if s[ix] == '{' and not start_ix_found:
                start_ix = ix
                start_ix_found = True
            if s[ix] == '}' and not end_ix_found:
                end_ix = ix
                end_ix_found = True
            if start_ix_found and end_ix_found:
                break
        return s[start_ix:end_ix+1]
    else:
        return s
    
def parse(docs,pydantic_parser,model_name,retry=False):
    outputs = []
    error_ixs = []
    for ix, doc in enumerate(docs):
        s = doc[0]["generated_text"].split(eoi_token[model_name])[-1].strip("\n").strip()
        try:
            output = pydantic_parser.parse(strip_to_wrapped_by_brackets(s)).dict()
        except:
            error_ixs.append(ix)
            output = {}
        # sometimes LLM follows the json format but gives an empty output.
        # in this case, treat it as a parsing error and retry.
        for ky in output:
            if not output[ky]:
                logger.info(f">>>>> empty value detected. key: {ky}; position: {ix}")
                logger.info(output)
                error_ixs.append(ix)
                output = {}
                break
        outputs.append({"processed":output,"raw":s})
    n_failures = len(error_ixs)
    if retry:
        logger.info(f"Retry parsing finished:\n\t# success: {len(docs)-n_failures}\n\t# failure: {n_failures}.")
    else:
        logger.info(f"Parsing finished:\n\t# success: {len(docs)-n_failures}\n\t# failure: {n_failures}.")
    return {"output":outputs,"error_indexes":error_ixs}