import json
import math
import sys
import copy
import re
import os
import json
import difflib
import asyncio
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

from tqdm.asyncio import tqdm_asyncio
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from mathruler.grader import extract_boxed_content
from concurrent.futures import as_completed

from collections import defaultdict, Counter
from openai import AzureOpenAI, OpenAI,AsyncAzureOpenAI,AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)


def tokenize(text):
    text = text.lower().replace(".", " .").split(" ")
    return text

def bleu(pred,target,n):
    weights=[1/n for _ in range(n)]
    tokenized_target = tokenize(target)
    tokenized_pred = tokenize(pred)
    return sentence_bleu([tokenized_target], tokenized_pred, weights=weights)

def rouge(pred,target):
    if not pred or not pred.strip():
        # 如果预测为空，直接返回 0 分结构（适配 rouge 库的返回格式）
        return [{
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }]

    rouge_scorer = Rouge()
    rouge_scores = rouge_scorer.get_scores(pred.lower(), target.lower())

    try:
        rouge_scores = rouge_scorer.get_scores(pred.lower(), target.lower())
    except ValueError:
        # 双重保险：如果转换后依然为空（例如只有标点符号被过滤后），捕获异常返回 0
        return [{
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }]
        
    return rouge_scores



def get_compare_messages(question,response,answer):
    prompt = f"""
Your task is to determine whether the user's answer is correct based on the provided questions and standard answers (for example, if the user expresses a similar meaning to the standard answer, or another interpretation of the standard answer, it is considered correct.)

The question is: {question}

The standard answer: {answer}

The user's answer: {response}

Please strictly follow the following format for output(0 represents correct, 1 represents incorrect):
<think>{{your concise think step}}</think>
<judge>{{0/1}}</judge>

for example:
<think>The standard answer is right, and the user's answer is right frontal lobe, they express the same meaning, so it is correct.</think>
<judge>0</judge>
    """
    messages = [{"role":"user","content":prompt}]
    return messages


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    most_similar_str = None
    most_similar_index = 0
    highest_similarity = 0
    
    for i, str in enumerate(str_list):
        similarity = str_similarity(str, target_str)
        
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    return most_similar_index

def judge_multi_choice(choices,answer,response,alphas = None):
    response = response.lower()
    if response.split("\n\n")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[0]
    elif response.split("\n\n")[-1].split(".")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[-1].split(".")[0]
    
    response = parse_response(response)
    alphas = [chr(ord('a') + i) for i in range(len(choices))]
    choices = [choice.lower() for choice in choices]
    flag = False
    response = response.strip().lower()
    response = response.replace("\n","")
    split_response = response.split(".")[0]
    split_response = split_response.split(":")[-1]
    answer = answer.strip().lower()
    
    if len(split_response) > 300:
        flag = False

    if split_response == answer:
        flag = True
    
    elif split_response in alphas:
        if choices[ord(split_response)-ord("a")]== answer:
            flag = True
    
    elif split_response in choices:
        if answer in alphas and split_response == choices[ord(answer)-ord("a")]:
            flag = True
    else:
        index = find_most_similar_index(choices,response)
        if alphas[index] == answer or choices[index] == answer:
            flag = True
    return flag


def parse_response(response):
    response = response.lower()
    if "boxed" in response:
        response = extract_boxed_content(response)
    elif "<answer>" in response:
        response = extract(response,"answer")
    answer_patterns = [
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is",
        "answer:",
        "答案:",
        "final answer",
        "final answer is"
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in response:
            response = response.split(answer_pattern)[-1]
    
    return response


def judge_close_end_vqa(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    response = response.replace("\n","").replace(".","")
    if response == answer:
        return True
    else:
        return False

def judge_judgement(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    response = response.replace("\n","").replace(".","")
    if ('yes' in response) ^ ('no' in response):
        if answer in response:
            return True
    return False


def judge_open_end_vqa(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    bleu1 = bleu(response,answer,1)
    bleu2 = bleu(response,answer,2)
    bleu3 = bleu(response,answer,3)
    bleu4 = bleu(response,answer,4)

    em = response == answer
    rouge_scores = rouge(response,answer)
    rouge_1 = rouge_scores[0]["rouge-1"]["f"]
    rouge_2 = rouge_scores[0]["rouge-2"]["f"]
    rouge_l = rouge_scores[0]["rouge-l"]["f"]

    precision,recall,f1 = calculate_f1(response,answer)


    return {
        "em" : em,
        "bleu1" : bleu1,
        "bleu2" : bleu2,
        "bleu3" : bleu3,
        "bleu4" : bleu4,
        "rouge1" : rouge_1,
        "rouge2" : rouge_2,
        "rougel" :  rouge_l,
        "precision": precision,
        "recall": recall,
        "f1" :f1         
    }


def calculate_f1(prediction, ground_truth):
    prediction_tokens = set(prediction.lower().split())
    ground_truth_tokens = set(ground_truth.lower().split())
    
    common = prediction_tokens & ground_truth_tokens
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0,0,0
    
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        return 0,0,0
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1,precision,recall





def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type,hard = True):
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            return target_str
        elif hard:
            return text
        else:
            return ""
    else:
        return ""

def save_json(filename, ds):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ds, f, indent=4, ensure_ascii=False)

class fake_response:
    def __init__(self,usage):
        self.usage = usage

def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

async def deal_tasks(tasks, max_concurrent_tasks=500):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results = []

    async def sem_task(task):
        async with semaphore:
            return await task  

    sem_tasks = [sem_task(task) for task in tasks]

    for coro in tqdm_asyncio.as_completed(sem_tasks, total=len(sem_tasks)):
        result = await coro
        results.append(result)

    return results

class openai_llm:
    def __init__(self, model=None, **kwargs):
        # self.model=kwargs.get("model", "gpt-4.1-2025-04-14")
        if model is None:
            model = os.environ.get("judge_model","gpt-4.1-2025-04-14")
        self.model = model
        print('using judge model:',self.model)
        # self.api_key = os.environ.get("openai_api_key")
        self.api_key = os.environ["api_key"]
        self.base_url = os.environ.get("base_url", "<OPENAI_BASE_URL>")
        print()
        if not self.api_key:
            raise ValueError("API key must be set.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def response(self, messages, **kwargs):
        attempt = 0
        while True: 
            try:
                attempt += 1
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    n=kwargs.get("n", 1),
                    temperature=0,
                    max_tokens=kwargs.get("max_tokens", 4000),
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API call failed on attempt {attempt}. Error: {e}. Retrying in 1 second...")
                time.sleep(1)
                
    def generate_outputs(self, messages_list, max_workers=64, **kwargs):
        results = [None] * len(messages_list)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.response, msg, **kwargs): i
                for i, msg in enumerate(messages_list)
            }
            
            for future in tqdm(as_completed(future_to_index), total=len(messages_list), desc="Generating Outputs"):
                index = future_to_index[future]
                try:
                    response = future.result()
                    results[index] = response
                except Exception as e:
                    print(f"A task failed unexpectedly after retries: {e}")

        return results


judger = openai_llm()