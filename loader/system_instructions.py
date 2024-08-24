ambiguity_type_definitions = """The ambiguity of a query can be multifaceted, and there are multiple possible ambiguity types: 
[1] Semantic: the query is semantically ambiguous for several common reasons: it may include homonyms; a word in the query may refer to a specific entity while also functioning as a common word; or an entity mention in the query could refer to multiple distinct entities. 
[2] Generalization: the query focuses on specific information; however, a broader, closely related query might better capture the user's true information needs.
[3] Specify: the query has a clear focus but may encompass too broad a research scope. It is possible to further narrow down this scope by providing more specific information related to the query.\n"""

class SystemInstruction:
    def __init__(self,args):
        if args.stage == "preprocessing":
            instruction = "Given a query in an information-seeking system, determine if the query is ambiguous, i.e. further clarification could lead to better search results. " \
                          "Before generating a bool variable (true or false), provide a textual explanation of the reasoning behind your decision."
        elif args.stage == "reformulation":
            instruction = "Given a chat history, summarize the conversation by reformulating the initial query. The chat history includes the initial query and several "\
                          "clarification turns between the user and a virtual assistant. In each turn, the virtual assistant asks clarification questions "\
                          "to better understand the user's intention." 
        elif args.stage == "response":
            if args.user_simulation_mode == "select":
                instruction = "Imagine that you're a user seeking to find information using a chat assistant. Starting with an initial query, in each turn of the conversation, "\
                              "the chat assistant will provide you with multiple reformulated queries to better understand your intention. Given the initial query, {{CHAT_HISTORY}} "\
                              "and a paragraph describing the user's intention, choose the reformulated query that most accurately reflects the user's intention. If none of "\
                              "provided reformulated queries accurately reflects the user's intention, use the original query as the reformulated query."
            elif args.user_simulation_mode == "respond":
                instruction = "Imagine that you're a user seeking to find information using a chat assistant. Starting with an initial query, in each turn of the conversation, "\
                              "the chat assistant will ask a clarification question to better understand your intention. Given the initial query, {{CHAT_HISTORY}} "\
                              "and a paragraph describing the user's intention, respond to the clarification question based on the user's intention."
            elif args.user_simulation_mode == "select+respond":
                instruction = "Imagine that you're a user seeking to find information using a chat assistant. Starting with an initial query, in each turn of the conversation, "\
                              "the chat assistant will provide multiple clarification questions to better understand your intention. Given the initial query, {{CHAT_HISTORY}} "\
                              "and a paragraph describing the user's intention, choose the best clarification question and then respond to it."
                            #   "You should select the clarification question that best resolves the ambiguity of the query and then respond accordingly."
            if args.turn_id == 1:
                if args.user_simulation_mode == "select":
                    instruction = instruction.replace("{{CHAT_HISTORY}}","the list of reformulated queries,")
                elif args.user_simulation_mode == "respond":
                    instruction = instruction.replace("{{CHAT_HISTORY}}","the clarification question,")
                elif args.user_simulation_mode == "select+respond":
                    instruction = instruction.replace("{{CHAT_HISTORY}}","the list of clarification questions,")
            else:
                if args.user_simulation_mode == "select":
                    instruction = instruction.replace("{{CHAT_HISTORY}}","the list of reformulated queries,")
                elif args.user_simulation_mode == "respond":
                    instruction = instruction.replace("{{CHAT_HISTORY}}","clarification questions and their corresponding responses from previous turns, the clarification question in the current turn,")
                elif args.user_simulation_mode == "select+respond":
                    instruction = instruction.replace("{{CHAT_HISTORY}}","selected clarification questions and their corresponding responses from previous turns, the list of clarification questions in the current turn,")
        elif args.stage == "generation":
            if args.turn_id == 1:
                instruction = "Given a query in an information-seeking system, "
                if args.user_simulation_mode == "select":
                    instruction += "generate 5 reformulated queries that clarify the original query to gain a better understanding of the user's intention. Imagine that the user will select the reformulated query "\
                                "that most accurately describes their intention in each turn.\n"
                    generation_content = "reformulated queries"
                elif args.user_simulation_mode == "respond":
                    instruction += "generate a clarification question that you think is most appropriate to gain a better understanding of the user's intention.\n"
                    generation_content = "the clarification question"
                elif args.user_simulation_mode == "select+respond":
                    instruction += "generate 5 clarification questions to gain a better understanding of the user's intention. Imagine that the user will select the question that best meets their needs in each round and "\
                                "respond to it.\n"
                    generation_content = "clarification questions"
                else:
                    instruction += "generate 5 clarification questions to gain a better understanding of the user's intention."
                    generation_content = "clarification questions"

                if "AT" in args.prompt_type and "CoT" in args.prompt_type:
                    instruction += ambiguity_type_definitions + f"Before generating {generation_content}, provide a textual explanation of your reasoning about which types of ambiguity apply to the given query. "\
                                                                "Based on these ambiguity types, describe how you plan to clarify the original query.\n"
                elif "AT" in args.prompt_type:
                    instruction += ambiguity_type_definitions + "Consider the above ambiguity types when generating.\n"
                elif "CoT" in args.prompt_type:
                    instruction += f"Before generating {generation_content}, provide a textual explanation of your reasoning about why the original query is ambiguous and how you plan to clarify it."+"\n"
            else:
                instruction = "Given a "
                if args.user_simulation_mode == "select":
                    instruction += "query in an information-seeking system, generate 5 reformulated queries for clarification to gain a better understanding of the user's intention. Imagine that the user will select "\
                                   "the reformulated query that most accurately describes their intention in each turn.\n"
                    generation_content = "reformulated queries"
                elif args.user_simulation_mode == "respond":
                    instruction += "chat history that includes an initial query, clarification questions, and corresponding answers from previous turns, generate a clarification question that you think is "\
                                    "most appropriate to gain a better understanding of the user's intention.\n"
                    generation_content = "the clarification question"
                elif args.user_simulation_mode == "select+respond":
                    instruction += "chat history that includes an initial query, clarification questions, and corresponding answers from previous turns, generate 5 clarification questions to gain a better understanding "\
                                   "of the user's intention. Imagine that the user will select the question that best meets their needs in each round and respond to it.\n"
                    generation_content = "clarification questions"

                if "AT" in args.prompt_type and "CoT" in args.prompt_type:
                    instruction += ambiguity_type_definitions + f"Before generating {generation_content}, provide a textual explanation of your reasoning about which types of ambiguity apply to the given chat history. "\
                                                                 "Based on these ambiguity types, describe how you plan to clarify the user's search.\n"
                elif "AT" in args.prompt_type:
                    instruction += ambiguity_type_definitions + "Consider the above ambiguity types when generating.\n"
                elif "CoT" in args.prompt_type:
                    instruction += f"Before generating {generation_content}, provide a textual explanation of your reasoning about why the user's search is ambiguous and how you plan to clarify it."+"\n"
        

        self.instruction = instruction
