import os
import re
import json

def clean_sentence(s):
    return re.sub(r'^\(\d+\)\s*', '', s)

class DataLoader:
    def __init__(self,args):
        self.dataset_name = args.dataset_name
        self.turn_id = args.turn_id
        self.stage = args.stage
        self.user_simulation_mode = args.user_simulation_mode
        self.prompt_type = args.prompt_type
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','data'))
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','output'))
        self.load()

    def load(self):
        self.data = {}
        if self.turn_id == 1:
            if self.stage in ["preprocessing","generation"]:
                source_data = json.load(open(os.path.join(self.data_dir,f"{self.dataset_name}.json")))
                self.data["query"] = source_data["query"]
            elif self.stage == "response":
                source_data = json.load(open(os.path.join(self.data_dir,f"{self.dataset_name}.json")))
                generation_result = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"turn_{self.turn_id}","generation",self.user_simulation_mode,self.prompt_type,"output.json")))["output"]
                self.data["query"] = source_data["query"]
                self.data["user_intention"] = source_data["user_intention"]
                if self.user_simulation_mode == "select":
                    self.data["reformulated_query"] = [res["processed"]["reformulated_queries"] for res in generation_result]
                elif self.user_simulation_mode == "respond":
                    self.data["clarification_question"] = [res["processed"]["clarification_question"] for res in generation_result]
                    # for q, cq in zip(self.data["query"],self.data["clarification_question"]):
                    #     print(q)
                    #     print('-'*10)
                    #     print(cq)
                    #     print('='*10)
                else:
                    self.data["clarification_question"] = [res["processed"]["clarification_questions"] for res in generation_result]
            elif self.stage == "reformulation":
                source_data = json.load(open(os.path.join(self.data_dir,f"{self.dataset_name}.json")))
                generation_result = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"turn_{self.turn_id}","generation",self.user_simulation_mode,self.prompt_type,"output.json")))["output"]
                response_result = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"turn_{self.turn_id}","response",self.user_simulation_mode,self.prompt_type,"output.json")))["output"]
                qs = self.extend_data_based_on_user_intention(source_data["user_intention"],source_data["query"])
                if self.user_simulation_mode == "respond":
                    cqs = [res["processed"]["clarification_question"] for res in generation_result]
                    cqs = self.extend_data_based_on_user_intention(source_data["user_intention"],cqs)
                    rs = [res["processed"]["response"] for res in response_result]
                    cq_tag = "Clarification question"
                else:
                    cqs = [clean_sentence(res["processed"]["best_clarification_question"]) for res in response_result]
                    rs = [res["processed"]["response"] for res in response_result]
                    cq_tag = "Selected clarification question"
                chs = []
                for q, cq, r in zip(qs,cqs,rs):
                    ch = f"Query: {q}"+'\n'+f"{cq_tag}: {cq}"+'\n'+f"Response: {r}"+'\n'
                    chs.append(ch)
                self.data["chat_history"] = chs
        else:
            if self.stage == "generation":
                prev_turn_data = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"turn_{self.turn_id-1}","summary.json")))[self.user_simulation_mode][self.prompt_type]
                if self.user_simulation_mode == "select":
                    self.data["query"] = prev_turn_data["reformulated_query"]
                else:
                    self.data["chat_history"] = prev_turn_data["chat_history"]
            elif self.stage == "response":
                source_data = json.load(open(os.path.join(self.data_dir,f"{self.dataset_name}.json")))
                prev_turn_data = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"turn_{self.turn_id-1}","summary.json")))[self.user_simulation_mode][self.prompt_type]
                generation_result = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"turn_{self.turn_id}","generation",self.user_simulation_mode,self.prompt_type,"output.json")))["output"]
                self.data["user_intention"] = self.flatten_user_intention(source_data["user_intention"])
                if self.user_simulation_mode == "select":
                    qs = prev_turn_data["reformulated_query"]
                    rqs = [res["processed"]["reformulated_queries"] for res in generation_result]
                    assert len(qs) == len(rqs) == len(self.data["user_intention"]), "List length dismatch: selected reformulated queries (from last turn); generated lists of reformulated queries; "\
                                                                                                           "user intention."
                    chat_history = []
                    for q, rq in zip(qs,rqs):
                        formatted_rqs = '\n'.join([f"({i+1}) {rq[i]}" for i in range(len(rq))])
                        chat_history.append(f"Query: {q}"+'\n'+f"List of reformulated queries: {formatted_rqs}")
                    self.data["chat_history"] = chat_history
                if self.user_simulation_mode == "respond":
                    cqs = [res["processed"]["clarification_question"] for res in generation_result]
                    prev_chs = prev_turn_data["chat_history"]
                    assert len(prev_chs) == len(cqs) == len(self.data["user_intention"]), "List length dismatch: chat history (from last turn), generated clarification questions, "\
                                                                                                           "user intention."
                    self.data["chat_history"] = [ch+'\n'+f"Clarification question: {cq}" for ch, cq in zip(prev_chs,cqs)]
                if self.user_simulation_mode == "select+respond":
                    cqs = [res["processed"]["clarification_questions"] for res in generation_result]
                    prev_chs = prev_turn_data["chat_history"]
                    assert len(prev_chs) == len(cqs) == len(self.data["user_intention"]), "List length dismatch: chat history (from last turn); generated lists of clarification questions; "\
                                                                                                           "user intention."
                    chat_history = []
                    for ch, cq in zip(prev_chs,cqs):
                        formatted_cqs = '\n'.join([f"({i+1}) {cq[i]}" for i in range(len(cq))])
                        chat_history.append(ch+'\n'+f"List of clarification questions: {formatted_cqs}")
                    self.data["chat_history"] = chat_history

            elif self.stage == "reformulation":
                prev_turn_data = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"turn_{self.turn_id-1}","summary.json")))[self.user_simulation_mode][self.prompt_type]
                generation_result = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"turn_{self.turn_id}","generation",self.user_simulation_mode,self.prompt_type,"output.json")))["output"]
                response_result = json.load(open(os.path.join(self.output_dir,self.dataset_name,f"turn_{self.turn_id}","response",self.user_simulation_mode,self.prompt_type,"output.json")))["output"]
                prev_chs = prev_turn_data["chat_history"]
                if self.user_simulation_mode == "respond":
                    cqs = [res["processed"]["clarification_question"] for res in generation_result]
                    rs = [res["processed"]["response"] for res in response_result]
                    cq_tag = "Clarification question"
                elif self.user_simulation_mode == "select+respond":
                    cqs = [clean_sentence(res["processed"]["best_clarification_question"]) for res in response_result]
                    rs = [res["processed"]["response"] for res in response_result]
                    cq_tag = "Selected clarification question"

                assert len(prev_chs) == len(cqs) == len(rs), "List length dismatch: chat history (from last turn); generated (lists of) clarifications; responses."

                chs = []
                for ch, cq, r in zip(prev_chs,cqs,rs):
                    ch = ch+'\n'+f"{cq_tag}: {cq}"+'\n'+f"Response: {r}"+'\n'
                    chs.append(ch)
                self.data["chat_history"] = chs

    def extend_data_based_on_user_intention(self,uis,data):
        extended_data = []
        for d, ui in zip(data,uis):
            extended_data += [d] * len(ui)
        return extended_data
    
    def flatten_user_intention(self,uis):
        flattened_uis = []
        for ui in uis:
            flattened_uis += ui
        return flattened_uis

