import os
import re
import json
import sys
from itertools import product
from opt import get_args

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
usms = ["select", "respond", "select+respond"]
pts = ["few-shot", "AT-few-shot", "CoT-few-shot", "AT-CoT-few-shot"]

def clean_sentence(s):
    return re.sub(r'^\(\d+\)\s*', '', s)

def collect_chat_history(args,qs,ls,usm,pt):
    if args.turn_id > 1:
        prev_turn_chs = json.load(open(os.path.join(args.output_dir,args.dataset_name,f"turn_{args.turn_id-1}","summary.json")))[usm][pt]["chat_history"]
        response_result = json.load(open(os.path.join(args.output_dir,args.dataset_name,f"turn_{args.turn_id}","response",usm,pt,"output.json")))["output"]
        assert len(prev_turn_chs) == len(response_result), f"{usm}/{pt}: length of response result should be the number of previous turn chat histories."
        if usm != "select":
            generation_result = json.load(open(os.path.join(args.output_dir,args.dataset_name,f"turn_{args.turn_id}","generation",usm,pt,"output.json")))["output"]
            reformulation_result = json.load(open(os.path.join(args.output_dir,args.dataset_name,f"turn_{args.turn_id}","reformulation",usm,pt,"output.json")))["output"]
            assert len(prev_turn_chs) == len(reformulation_result), f"{usm}/{pt}: length of reformulation result should be the number of previous turn chat histories."
        
        if usm == "select":
            rqs = [clean_sentence(doc["processed"]["best_reformulated_query"]) for doc in response_result]
            chat_history = [ch+'\n'+f"Selected reformulated query: {rq}" for ch, rq in zip(prev_turn_chs,rqs)]
        if usm == "respond":
            rqs = [doc["processed"]["reformulated_query"] for doc in reformulation_result]
            cqs = [doc["processed"]["clarification_question"] for doc in generation_result]
            rs = [doc["processed"]["response"] for doc in response_result]
            chat_history = ['\n'.join([ch,f"Clarification question: {cq}",f"Response: {r}"]) for ch, cq, r in zip(prev_turn_chs,cqs,rs)]
        if usm == "select+respond":
            rqs = [doc["processed"]["reformulated_query"] for doc in reformulation_result]
            cqs = [clean_sentence(doc["processed"]["best_clarification_question"]) for doc in response_result]
            rs = [doc["processed"]["response"] for doc in response_result]
            chat_history = ['\n'.join([ch,f"Selected clarification question: {cq}",f"Response: {r}"]) for ch, cq, r in zip(prev_turn_chs,cqs,rs)]
    else:
        response_result = json.load(open(os.path.join(args.output_dir,args.dataset_name,f"turn_{args.turn_id}","response",usm,pt,"output.json")))["output"]
        assert len(qs) == len(response_result), f"{usm}/{pt}: length of response result should be the number of user intentions."
        if usm != "select":
            generation_result = json.load(open(os.path.join(args.output_dir,args.dataset_name,f"turn_{args.turn_id}","generation",usm,pt,"output.json")))["output"]
            reformulation_result = json.load(open(os.path.join(args.output_dir,args.dataset_name,f"turn_{args.turn_id}","reformulation",usm,pt,"output.json")))["output"]
            assert len(ls) == len(generation_result), f"{usm}/{pt}: length of generation result should be the number of initial queries."
            assert len(qs) == len(reformulation_result), f"{usm}/{pt}: length of reformulation result should be the number of user intentions."

        if usm == "select":
            rqs = [clean_sentence(doc["processed"]["best_reformulated_query"]) for doc in response_result]
            chat_history = [f"Query:{q}"+'\n'+f"Selected reformulated query: {rq}" for q, rq in zip(qs,rqs)]
        if usm == "respond":
            rqs = [doc["processed"]["reformulated_query"] for doc in reformulation_result]
            single_cqs = [doc["processed"]["clarification_question"] for doc in generation_result]
            cqs = []
            for cq, l in zip(single_cqs,ls):
                cqs += [cq] * l
            rs = [doc["processed"]["response"] for doc in response_result]
            chat_history = ['\n'.join([f"Query: {q}",f"Clarification question: {cq}",f"Response: {r}"]) for q, cq, r in zip(qs,cqs,rs)]
        if usm == "select+respond":
            rqs = [doc["processed"]["reformulated_query"] for doc in reformulation_result]
            cqs = [clean_sentence(doc["processed"]["best_clarification_question"]) for doc in response_result]
            rs = [doc["processed"]["response"] for doc in response_result]
            chat_history = ['\n'.join([f"Query: {q}",f"Selected clarification question: {cq}",f"Response: {r}"]) for q, cq, r in zip(qs,cqs,rs)]
    
    return {"chat_history":chat_history,"reformulated_query":rqs}

if __name__ == "__main__":
    args = get_args()

    source_data = json.load(open(os.path.join(args.data_dir,f"{args.dataset_name}.json")))
    qs, ls = [], []
    for q, uis in zip(source_data["query"],source_data["user_intention"]):
        qs += [q] * len(uis)
        ls.append(len(uis))

    summary = {usm:{} for usm in usms}

    for usm in usms:
        for pt in pts:
            summary[usm][pt] = collect_chat_history(args,qs,ls,usm,pt)
    
    dst_fn = os.path.join(args.output_dir,args.dataset_name,f"turn_{args.turn_id}","summary.json")
    with open(dst_fn,'w') as f:
        json.dump(summary,f)
    
    print(f"success: summary file of {args.dataset_name.upper()}-TURN_{args.turn_id} saved to {dst_fn}.")
    



    