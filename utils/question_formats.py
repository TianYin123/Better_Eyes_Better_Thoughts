def medframeqa(question,choices,is_reasoning = False,lang = "en"):
    choices = [str(choice) for choice in choices]
    options = "\n".join(choices)

    if lang == "en":
        prompt = f"""{question}\n{options}"""
        if is_reasoning:
            prompt = prompt + "\n" + 'Please reason step by step, and put your final answer within \\boxed{}.'
        else:
            prompt = prompt + "\n" + "Answer with the option's letter from the given choices directly." 

    elif lang == "zh":
        prompt = f"""{question}\n{options}"""
        if is_reasoning:
            prompt = prompt + "\n" + '请直接使用给定选项中的选项字母来回答该问题,并将答案包裹在"\\boxed{}"里'
        else:
            prompt = prompt + "\n" +  "请直接使用给定选项中的选项字母来回答该问题。"
    return prompt
def get_multiple_choice_prompt(question,choices,is_reasoning = False,lang = "en"):
    choices = [str(choice) for choice in choices]
    options = "\n".join(choices)

    if lang == "en":
        prompt = f"""
Question: {question}
Options: 
{options}"""
        if is_reasoning:
            prompt = prompt + "\n" + 'Please reason step by step, and put your final answer within \\boxed{}.'
        else:
            prompt = prompt + "\n" + "Answer with the option's letter from the given choices directly." 

    elif lang == "zh":
        prompt = f"""
问题： {question}
选项： 
{options}"""
        if is_reasoning:
            prompt = prompt + "\n" + '请直接使用给定选项中的选项字母来回答该问题,并将答案包裹在"\\boxed{}"里'
        else:
            prompt = prompt + "\n" +  "请直接使用给定选项中的选项字母来回答该问题。"
    return prompt

def get_close_ended_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = question + "\n" + 'Please reason step by step, and put your final answer within \\boxed{}.'
        else:
            prompt = question + "\n" + "Answer the question using a single word or phrase."
    elif lang == "zh":
        if is_reasoning:
            prompt = question + "\n" + "请用一个单词或者短语回答该问题，并将答案放在一个'\\boxed{}'中。"
        else:
            prompt = question + "\n" + "请用一个单词或者短语回答该问题。"
    return prompt

def get_open_ended_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = question + "\n" + 'Please reason step by step, and put your final answer within \\boxed{}.'
        else:
            prompt = question + "\n" + "Please answer the question concisely."
    elif lang == "zh":
        if is_reasoning:
            prompt = question + "\n" + "请简要回答该问题，并将答案放在一个'\\boxed{}'中。"
        else:
            prompt = question + "\n" + "请简要回答该问题。"
    return prompt
def get_judgement_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = question + "\n" + 'Please reason step by step, and put your final answer within \\boxed{}.'
        else:
            prompt = question + "\n" + "Answer the question using a single word or phrase."
    elif lang == "zh":
        if is_reasoning:
            prompt = question + "\n" + "请输出'是'或'否'，并将答案放在一个'\\boxed{}'中。"
        else:
            prompt = question + "\n" + "请输出'是'或'否'(不要有任何其它输出)。"
    return prompt
def get_report_generation_prompt():
    prompt = "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."
    return prompt



