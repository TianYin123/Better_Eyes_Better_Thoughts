import torch
import os
import json
import gc

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from mathruler.grader import extract_boxed_content

from ..utils import save_json,extract,judger,get_compare_messages,judge_open_end_vqa,judge_judgement,judge_close_end_vqa
from ..base_dataset import BaseDataset

from ..question_formats import get_judgement_prompt,get_open_ended_prompt

class PATH_VQA(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "flaviagiammarino/path-vqa"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    

    def load_data(self):
        dataset_path = self.dataset_path
        dataset = load_dataset(dataset_path,split = "test")

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        question = sample["question"]
        image = sample["image"]
        answer = sample["answer"]

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        if answer in ["yes","no"]:
            prompt = get_judgement_prompt(question,is_reasoning)
        else:
            prompt = get_open_ended_prompt(question,is_reasoning)

        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages
        del sample["image"]
        return sample


    def cal_metrics(self, out_samples):
        judger_inputs = []

        metrics = {
            "total metrics": {"total": 0, "right": 0},
            "open": {
                "total": 0, "right": 0, "bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0,
                "rouge1": 0, "rouge2": 0, "rougel": 0, "precision": 0, "recall": 0,
                "f1": 0, "em": 0,
            },
            "close": {"total": 0, "right": 0}
        }

        for i, out_sample in tqdm(enumerate(out_samples), desc="Initial metric calculation"):
            response = out_sample["response"]
            if extract_boxed_content(response) != "None":
                response = extract_boxed_content(response)
            elif "<answer>" in response:
                response = extract(response, "answer")

            answer = out_sample["answer"]
            question = out_sample["question"]
            answer = answer.lower().strip()
            response_cleaned = response.lower().strip()

            metrics["total metrics"]["total"] += 1
            is_closed_question = answer in ["yes", "no"]

            if is_closed_question:
                metrics["close"]["total"] += 1
                correct = judge_judgement(answer, response_cleaned)
                out_samples[i]["correct_by_rule"] = correct
            else: # open-ended
                metrics["open"]["total"] += 1
                c_metrics = judge_open_end_vqa(answer, response_cleaned)
                out_samples[i]["metrics"] = c_metrics
                out_samples[i]["correct_by_rule"] = c_metrics["em"]
                for metric in c_metrics:
                    if metric in metrics["open"]:
                        metrics["open"][metric] += c_metrics[metric]

            if os.environ.get("use_llm_judge", "False") == "True":
                messages = get_compare_messages(question, response, answer)
                judger_inputs.append({
                    "index": i,
                    "messages": messages,
                    "is_closed": is_closed_question
                })

        if os.environ.get("use_llm_judge", "False") == "True" and judger_inputs:
            print(f"Using LLM Judger for {len(judger_inputs)} samples...")
            
            metrics["total metrics"]["right"] = 0
            metrics["open"]["right"] = 0
            metrics["close"]["right"] = 0
            
            llm = judger
            messages_list_for_judger = [item["messages"] for item in judger_inputs]
            results = llm.generate_outputs(messages_list_for_judger)
            
            for item, judger_result in zip(judger_inputs, results):
                sample_idx = item["index"]
                sample_is_closed = item["is_closed"]
                
                judgement = extract(judger_result, "judge")
                is_correct = True if judgement == "0" else False
                
                out_samples[sample_idx]["correct"] = is_correct
                out_samples[sample_idx]["judger_output"] = judger_result

                if is_correct:
                    metrics["total metrics"]["right"] += 1
                    if sample_is_closed:
                        metrics["close"]["right"] += 1
                    else:
                        metrics["open"]["right"] += 1
        else:
            print("Using rule-based judging...")
            for i in range(len(out_samples)):
                is_correct = out_samples[i].get("correct_by_rule", False)
                out_samples[i]["correct"] = is_correct
                if is_correct:
                    metrics["total metrics"]["right"] += 1
                    if out_samples[i]["answer"].lower() in ["yes", "no"]:
                        metrics["close"]["right"] += 1
                    else:
                        metrics["open"]["right"] += 1

        if metrics["total metrics"]["total"] > 0:
            metrics["total metrics"]["acc"] = metrics["total metrics"]["right"] / metrics["total metrics"]["total"]
        if metrics["open"]["total"] > 0:
            metrics["open"]["acc"] = metrics["open"]["right"] / metrics["open"]["total"]
            for metric in metrics["open"]:
                if metric not in ["right", "total", "acc"]:
                    metrics["open"][metric] /= metrics["open"]["total"]
        if metrics["close"]["total"] > 0:
            metrics["close"]["acc"] = metrics["close"]["right"] / metrics["close"]["total"]
            
        return metrics, out_samples