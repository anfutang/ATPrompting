few_shot_instruction = "Here are some examples.\n\n"

avoid_copying_instruction = "Do not repeat the examples exactly; instead, you should generalize beyond the given examples. Now, focus on the following "

generation_template = [{"role":"system","content":"{system_instruction}\n{format_instruction}"},
                       {"role":"user","content":few_shot_instruction+"{few_shot_examples}\n"+avoid_copying_instruction+"query.\n\n"+"Query: {query}\n"}]

multi_turn_generation_template = [{"role":"system","content":"{system_instruction}\n{format_instruction}"},
                                    {"role":"user","content":few_shot_instruction+"{few_shot_examples}\n"+avoid_copying_instruction+"chat.\n\n"+"{chat_history}\n"}]

# select_response_template = [{"role":"system","content":"{system_instruction}\n{format_instruction}"},
#                             {"role":"user","content":"{few_shot_examples}\nQuery: {query}\nUser intention: {user_intention}\n"
#                                                      "List of reformulated queries: {reformulated_queries}\n"}]

# respond_response_template = [{"role":"system","content":"{system_instruction}\n{format_instruction}"},
#                              {"role":"user","content":"{few_shot_examples}\nQuery: {query}\nUser intention: {user_intention}\n"
#                                                       "Clarification question: {clarification_question}\n"}]

# select_respond_response_template = [{"role":"system","content":"{system_instruction}\n{format_instruction}"},
#                             {"role":"user","content":"{few_shot_examples}\nQuery: {query}\nUser intention: {user_intention}\n"
#                                                      "List of clarification questions: {clarification_questions}\n"}]

response_template = [{"role":"system","content":"{system_instruction}\n{format_instruction}"},
                     {"role":"user","content":few_shot_instruction+"{few_shot_examples}\n"+avoid_copying_instruction+"chat.\n\n"+"{chat_history}\nUser intention: {user_intention}\n"}]

reformulation_template = [{"role":"system","content":"{system_instruction}\n\n{format_instruction}"},
                          {"role":"user","content":few_shot_instruction+"{few_shot_examples}\n"+avoid_copying_instruction+"chat.\n\n"+"{chat_history}\n"}]