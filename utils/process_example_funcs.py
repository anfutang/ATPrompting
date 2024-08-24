import os
import sys
import pandas as pd

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',"few_shot_examples","original_files"))

# python str() by default use single quotes which will cause parsing error using Langchain Pydantic parser. Therefore, can't simply use str() to turn a list to a string.
def list_to_string(l):
    s = "["
    for i, ss in enumerate(l):
        if i < len(l) - 1:
            s += '"' + ss + '", '
        else:
            s += '"' + ss + '"'
    return s + "]"

def single_turn_generation_select(args):
    df = pd.read_csv(os.path.join(data_dir,"single_turn","generation","select.csv"))
    examples = []
    if args.prompt_type in ["few-shot","AT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            rqs = [row[f"rq{i}"] for i in range(1,6)]
            examples.append((f"Query: {query}","{\"reformulated_queries\":"+list_to_string(rqs)+"}"))
    elif args.prompt_type in ["CoT-few-shot","AT-CoT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            rqs = [row[f"rq{i}"] for i in range(1,6)]
            if args.prompt_type == "CoT-few-shot":
                reasoning = '"'+row["reasoning"].replace('"',"'")+'"'
            else:
                reasoning = '"'+row["AT-reasoning"].replace('"',"'")+'"'
            examples.append((f"Query: {query}","{\"reasoning\":"+reasoning+",\"reformulated_queries\":"+list_to_string(rqs)+"}"))
    return examples
    
def single_turn_generation_respond(args):
    df = pd.read_csv(os.path.join(data_dir,"single_turn","generation","respond.csv"))
    examples = []
    if args.prompt_type in ["few-shot","AT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            cq = '"'+row["cq"]+'"'
            examples.append((f"Query: {query}","{\"clarification_question\":"+cq+"}"))
    elif args.prompt_type in ["CoT-few-shot","AT-CoT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            cq = '"'+row["cq"]+'"'
            if args.prompt_type == "CoT-few-shot":
                reasoning = '"'+row["reasoning"].replace('"',"'")+'"'
            else:
                reasoning = '"'+row["AT-reasoning"].replace('"',"'")+'"'
            examples.append((f"Query: {query}","{\"reasoning\":"+reasoning+",\"clarification_question\":"+cq+"}"))
    return examples

def single_turn_generation_select_respond(args):
    df = pd.read_csv(os.path.join(data_dir,"single_turn","generation","select+respond.csv"))
    examples = []
    if args.prompt_type in ["few-shot","AT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            cqs = [row[f"cq{i}"] for i in range(1,6)]
            examples.append((f"Query: {query}","{\"clarification_questions\":"+list_to_string(cqs)+"}"))
    elif args.prompt_type in ["CoT-few-shot","AT-CoT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            cqs = [row[f"cq{i}"] for i in range(1,6)]
            if args.prompt_type == "CoT-few-shot":
                reasoning = '"'+row["reasoning"].replace('"',"'")+'"'
            else:
                reasoning = '"'+row["AT-reasoning"].replace('"',"'")+'"'
            examples.append((f"Query: {query}","{\"reasoning\":"+reasoning+",\"clarification_questions\":"+list_to_string(cqs)+"}"))
    return examples

def single_turn_response_select():
    df = pd.read_csv(os.path.join(data_dir,"single_turn","response","select.csv"))
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        rqs = row["rqs"]
        user_intention = row["user_intention"]
        selected_rq = '"'+row["selected_rq"].replace('"',"'")+'"'
        input = f"Query: {query}"+'\n'+f"List of reformulated queries: {rqs}"+'\n'+f"User intention: {user_intention}"+'\n'
        examples.append((input,"{\"best_reformulated_query\":"+selected_rq+"}"))
    return examples

def single_turn_response_respond():
    df = pd.read_csv(os.path.join(data_dir,"single_turn","response","respond.csv"))
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        cq = row["cq"]
        user_intention = row["user_intention"]
        response = '"'+row["response"].replace('"',"'")+'"'
        input = f"Query: {query}"+'\n'+f"Clarification question: {cq}"+'\n'+f"User intention: {user_intention}"+'\n'
        examples.append((input,"{\"response\":"+response+"}"))
    return examples

def single_turn_response_select_respond():
    df = pd.read_csv(os.path.join(data_dir,"single_turn","response","select+respond.csv"))
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        cqs = row["cqs"]
        user_intention = row["user_intention"]
        selected_cq = '"'+row["selected_cq"].replace('"',"'")+'"'
        response = '"'+row["response"].replace('"',"'")+'"'
        input = f"Query: {query}"+'\n'+f"List of clarification questions: {cqs}"+'\n'+f"User intention: {user_intention}"+'\n'
        examples.append((input,"{\"best_clarification_question\":"+selected_cq+",\"response\":"+response+"}"))
    return examples

def single_turn_reformulation_respond():
    df = pd.read_csv(os.path.join(data_dir,"single_turn","reformulation","respond.csv"))
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        cq = row["cq"]
        response = row["response"]
        rq = '"'+row["rq"].replace('"',"'")+'"'
        input = f"Query: {query}"+'\n'+f"Clarification question: {cq}"+'\n'+f"Response: {response}"+'\n'
        examples.append((input,"{\"reformulated_query\":"+rq+'}'))
    return examples 

def single_turn_reformulation_select_respond():
    df = pd.read_csv(os.path.join(data_dir,"single_turn","reformulation","select+respond.csv"))
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        cq = row["cq"]
        response = row["response"]
        rq = '"'+row["rq"].replace('"',"'")+'"'
        input = f"Query: {query}"+'\n'+f"Selected clarification question: {cq}"+'\n'+f"Response: {response}"+'\n'
        examples.append((input,"{\"reformulated_query\":"+rq+'}'))
    return examples 

def single_turn_preprocessing():
    df = pd.read_csv(os.path.join(data_dir,"single_turn","preprocessing","preprocessing.csv"))
    str2bool = {'y':'true','n':'false'}
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        reasoning = '"'+row["reasoning"].replace('"',"'")+'"'
        ambiguous = str2bool[row["ambiguous"]]
        input = f"Query: {query}"+'\n'
        examples.append((input,"{\"reasoning\":"+reasoning+f",\"ambiguous\":{ambiguous}"+'}'))
    return examples 

def multi_turn_generation_respond(args):
    df = pd.read_csv(os.path.join(data_dir,"multi_turn","generation","respond.csv"))
    examples = []
    if args.prompt_type in ["few-shot","AT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            prev_cq = row["previous_cq"]
            prev_response = row["previous_response"]
            input = f"Query: {query}" + '\n' + f"Clarification question: {prev_cq}" + '\n' + f"Response: {prev_response}"
            cq = '"'+row["cq"]+'"'
            examples.append((input,"{\"clarification_question\":"+cq+"}"))
    elif args.prompt_type in ["CoT-few-shot","AT-CoT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            prev_cq = row["previous_cq"]
            prev_response = row["previous_response"]
            input = f"Query: {query}" + '\n' + f"Clarification question: {prev_cq}" + '\n' + f"Response: {prev_response}"
            cq = '"'+row["cq"]+'"'
            if args.prompt_type == "CoT-few-shot":
                reasoning = '"'+row["reasoning"].replace('"',"'")+'"'
            else:
                reasoning = '"'+row["AT-reasoning"].replace('"',"'")+'"'
            examples.append((input,"{\"reasoning\":"+reasoning+",\"clarification_question\":"+cq+"}"))
    return examples

def multi_turn_generation_select_respond(args):
    df = pd.read_csv(os.path.join(data_dir,"multi_turn","generation","select+respond.csv"))
    examples = []
    if args.prompt_type in ["few-shot","AT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            prev_cq = row["previous_selected_cq"]
            prev_response = row["previous_response"]
            input = f"Query: {query}" + '\n' + f"Selected clarification question: {prev_cq}" + '\n' + f"Response: {prev_response}"
            cqs = [row[f"cq{i}"] for i in range(1,6)]
            examples.append((input,"{\"clarification_questions\":"+list_to_string(cqs)+"}"))
    elif args.prompt_type in ["CoT-few-shot","AT-CoT-few-shot"]:
        for _, row in df.iterrows():
            query = row["query"]
            prev_cq = row["previous_selected_cq"]
            prev_response = row["previous_response"]
            input = f"Query: {query}" + '\n' + f"Selected clarification question: {prev_cq}" + '\n' + f"Response: {prev_response}"
            cqs = [row[f"cq{i}"] for i in range(1,6)]
            if args.prompt_type == "CoT-few-shot":
                reasoning = '"'+row["reasoning"].replace('"',"'")+'"'
            else:
                reasoning = '"'+row["AT-reasoning"].replace('"',"'")+'"'
            examples.append((input,"{\"reasoning\":"+reasoning+",\"clarification_questions\":"+list_to_string(cqs)+"}"))
    return examples

def multi_turn_response_respond():
    df = pd.read_csv(os.path.join(data_dir,"multi_turn","response","respond.csv"))
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        prev_cq = row["previous_cq"]
        prev_response = row["previous_response"]
        cq = row["cq"]
        user_intention = row["user_intention"]
        input = f"Query: {query}" + '\n' + f"Clarification question: {prev_cq}" + '\n' + f"Response: {prev_response}" + '\n' + \
                 f"Clarification question: {cq}" + '\n' + f"User intention: {user_intention}" + '\n'
        response = '"'+row["response"].replace('"',"'")+'"'
        examples.append((input,"{\"response\":"+response+"}"))
    return examples

def multi_turn_response_select_respond():
    df = pd.read_csv(os.path.join(data_dir,"multi_turn","response","select+respond.csv"))
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        prev_cq = row["previous_selected_cq"]
        prev_response = row["previous_response"]
        cqs = row["cqs"]
        user_intention = row["user_intention"]
        input = f"Query: {query}" + '\n' + f"Selected clarification question: {prev_cq}" + '\n' + f"Response: {prev_response}" + '\n' + \
                 f"List of clarification questions: {cqs}" + '\n' + f"User intention: {user_intention}"+'\n'
        selected_cq = '"'+row["selected_cq"].replace('"',"'")+'"'
        response = '"'+row["response"].replace('"',"'")+'"'
        examples.append((input,"{\"best_clarification_question\":"+selected_cq+",\"response\":"+response+"}"))
    return examples

def multi_turn_reformulation_respond():
    df = pd.read_csv(os.path.join(data_dir,"multi_turn","reformulation","respond.csv"))
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        prev_cq, cq = row["previous_cq"], row["cq"]
        prev_response, response = row["previous_response"], row["response"]
        rq = '"'+row["rq"].replace('"',"'")+'"'
        input = f"Query: {query}"+'\n'+f"Clarification question: {prev_cq}"+'\n'+f"Response: {prev_response}"+'\n'+\
                f"Clarification question: {cq}"+'\n'+f"Response: {response}"+'\n'
        examples.append((input,"{\"reformulated_query\":"+rq+'}'))
    return examples 

def multi_turn_reformulation_select_respond():
    df = pd.read_csv(os.path.join(data_dir,"multi_turn","reformulation","select+respond.csv"))
    examples = []
    for _, row in df.iterrows():
        query = row["query"]
        prev_cq, cq = row["previous_selected_cq"], row["selected_cq"]
        prev_response, response = row["previous_response"], row["response"]
        rq = '"'+row["rq"].replace('"',"'")+'"'
        input = f"Query: {query}"+'\n'+f"Selected clarification question: {prev_cq}"+'\n'+f"Response: {prev_response}"+'\n'+\
                f"Selected clarification question: {cq}"+'\n'+f"Response: {response}"+'\n'
        examples.append((input,"{\"reformulated_query\":"+rq+'}'))
    return examples 