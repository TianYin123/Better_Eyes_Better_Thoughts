import base64
import json
import io
import re
import os
import time
import random
import csv
from PIL import Image
import argparse
import concurrent.futures
from datetime import datetime
from datasets import load_dataset, Dataset
from openai import OpenAI
from tqdm import tqdm

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "<OPENROUTER_API_KEY>")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "<OPENROUTER_BASE_URL>")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "<JUDGE_API_KEY>")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "<JUDGE_BASE_URL>")
DATASET_NAME = "flaviagiammarino/vqa-rad"
TEST_MODEL_NAME = "<TEST_MODEL_NAME>"
JUDGE_MODEL_NAME = "<JUDGE_MODEL_NAME>"

IS_REASONING = True

MEDUNIEVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MAX_WORKERS = 16
MAX_RETRIES = 10
RETRY_DELAY_BASE = 5


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_paths(test_model_name, dataset_name, is_reasoning):
    dir_mark = "cot" if is_reasoning else "direct"
    results_dir = os.path.join(MEDUNIEVAL_ROOT, "results", "exp7", test_model_name, dataset_name, dir_mark)
    results_file = os.path.join(results_dir, "results.json")
    metrics_file = os.path.join(results_dir, "metrics.json")
    return results_dir, results_file, metrics_file


def load_json_file(file_path, default_value):
    if not os.path.exists(file_path):
        return default_value
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value

# ================= Retry Decorator =================

def retry_request(func, max_retries, delay_base, *args, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1
            wait_time = delay_base * (2 ** (retries - 1)) + random.uniform(0, 1)
            print(f"\n[Warning] {func.__name__} failed: {e}. Retrying ({retries}/{max_retries}) in {wait_time:.2f}s...")
            time.sleep(wait_time)
    print(f"\n[Error] All {max_retries} retries failed for {func.__name__}.")
    return None

# ================= Prompt Construction Functions =================

def get_judgement_prompt(question, is_reasoning=False, lang="en"):
    if lang == "en":
        if is_reasoning:
            prompt = question + "\n" + 'Please reason step by step, and put your final answer within \\boxed{}.'
        else:
            prompt = question + "\n" + "Answer the question using a single word or phrase."
    return prompt

def get_open_ended_prompt(question, is_reasoning=False, lang="en"):
    if lang == "en":
        if is_reasoning:
            prompt = question + "\n" + 'Please reason step by step, and put your final answer within \\boxed{}.'
        else:
            prompt = question + "\n" + "Please answer the question concisely."
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

    return prompt

def get_close_ended_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = question + "\n" + 'Please reason step by step, and put your final answer within \\boxed{}.'
        else:
            prompt = question + "\n" + "Answer the question using a single word or phrase."
    return prompt


class DiffDataLoader:
    def __init__(self, is_reasoning):
        self.is_reasoning = is_reasoning

    def construct_1_messages(self, sample):
        question = sample["question"]
        image = sample["image"]
        answer = sample["answer"]

        answer = answer.lower()
        if answer in ["yes","no"]:
            prompt = get_judgement_prompt(question, self.is_reasoning)
        else:
            prompt = get_open_ended_prompt(question, self.is_reasoning)

        messages = {"prompt": prompt, "image": image}
        sample["messages"] = messages
        if "image" in sample:
            del sample["image"]
        return sample

    def construct_2_messages(self, sample, dataset_path):
        prompt = sample["prompt"]
        image = sample["image"] if os.path.exists(sample["image"]) else os.path.join(dataset_path, "images", sample["image"])

        messages = {"prompt": prompt, "image": image}
        sample["messages"] = messages
        if "image" in sample:
            del sample["image"]
        return sample

    def construct_3_messages(self, sample):
        image = sample["image_path"]
        # image = Image.open(sample["image_path"])
        choices = []
        question = sample["question"]
        answer = sample["gt_answer"]
        for option in ["A","B","C","D"]:
            if f"option_{option}" in sample:
                choice = sample[f"option_{option}"]
                if answer == choice:
                    sample["gt_answer"] = option
                choice = f"{option}. {choice}"
                choices.append(choice)
                
        prompt = get_multiple_choice_prompt(question, choices, self.is_reasoning)
            
        messages = {"prompt": prompt, "image": image}
        sample["messages"] = messages
        sample["choices"] = choices
        sample["answer"] = sample["gt_answer"]

        if "gt_answer" in sample:
            del sample["gt_answer"]
        return sample

    def load_diff_data(self, dataset_name):
        samples = []
        if dataset_name == 'VQA_RAD':
            data = load_dataset("flaviagiammarino/vqa-rad", split="test")
            for idx, sample in tqdm(enumerate(data), desc=f"Loading {dataset_name}"):
                sample = self.construct_1_messages(sample)
                samples.append(sample)
        
        elif dataset_name == 'PATH_VQA':
            data = load_dataset("flaviagiammarino/path-vqa", split="test")
            for idx, sample in tqdm(enumerate(data), desc=f"Loading {dataset_name}"):
                sample = self.construct_1_messages(sample)
                samples.append(sample)
        
        elif dataset_name == 'SLAKE':
            dataset_path = 'datas/SLAKE'
            test_json_path = os.path.join(dataset_path, "test.json")
            with open(test_json_path, "r", encoding='utf-8') as f:
                datas = json.load(f)
            for data in tqdm(datas, desc=f"Loading {dataset_name}"):
                img_path = data["img_name"]
                question = data["question"]
                answer = data["answer"]
                answer_type = data["answer_type"]
                lang = data["q_lang"]
                if lang == 'zh':
                    continue
                img_path = os.path.join(dataset_path, "imgs", img_path)
                # image = Image.open(img_path)
                if answer_type == "OPEN":
                    prompt = get_open_ended_prompt(question, self.is_reasoning, lang)
                else:
                    prompt = get_close_ended_prompt(question, self.is_reasoning, lang)
                messages = {"prompt": prompt, "image": img_path}
                samples.append({"messages": messages, "lang": lang, "answer_type": answer_type, "answer": answer, "question": question})
        
        elif dataset_name == 'PMC_VQA':
            dataset = []
            dataset_path = 'datas/PMC-VQA'
            csv_path = os.path.join(dataset_path, "test_2.csv")
            reader = csv.reader(open(csv_path, 'r', encoding='utf-8'))
            next(reader)
            for i, row in enumerate(reader):
                index, figure_path, caption, question, choiceA, choiceB, choiceC, choiceD, answer, split = row
                choices = [choiceA, choiceB, choiceC, choiceD]
                prompt = get_multiple_choice_prompt(question, choices, self.is_reasoning)
                image_path = os.path.join(dataset_path, "figures", figure_path)
                sample = {"prompt": prompt, "answer": answer, "image": image_path, "choices": choices, "question": question}
                dataset.append(sample)

            for idx, sample in tqdm(enumerate(dataset), desc=f"Loading {dataset_name}"):
                sample = self.construct_2_messages(sample, dataset_path)
                samples.append(sample)
        
        elif dataset_name == 'OmniMedVQA':
            dataset_path = 'datas/OmniMedVQA'
            datasets = []
            open_json_path = os.path.join(dataset_path, "QA_information", "Open-access")
            files = os.listdir(open_json_path)
            for file in tqdm(files, desc="Load Open-access data"):
                file_path = os.path.join(open_json_path, file)
                with open(file_path, "r", encoding='utf-8') as f:
                    datas = json.load(f)
                for data in datas:
                    data["image_path"] = os.path.join(dataset_path, data["image_path"])
                    datasets.append(data)
            for idx, sample in tqdm(enumerate(datasets), desc=f"Loading {dataset_name}"):
                sample = self.construct_3_messages(sample)
                samples.append(sample)

        return Dataset.from_list(samples)

def encode_image(image):
    pil_image = None
    if isinstance(image, dict):
        if 'bytes' in image:
            try:
                pil_image = Image.open(io.BytesIO(image['bytes']))
            except Exception as e:
                print(f"[Debug] Failed to load from bytes: {e}")
        
        if pil_image is None and 'path' in image:
            pil_image = Image.open(image['path'])
            
        if pil_image is None:
            for val in image.values():
                if isinstance(val, Image.Image):
                    pil_image = val
                    break

    elif isinstance(image, str):
        pil_image = Image.open(image)
    
    elif isinstance(image, Image.Image):
        pil_image = image

    if pil_image is None:
        raise ValueError(f"[Error] Cannot parse image input of type: {type(image)}")

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_model_response(client, model_name, image_b64, prompt_text):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=8192,
    )
    return response.choices[0].message.content

def get_judge_result(client, model_name, question, answer, response_text,dataset_name):
    if dataset_name in ["OmniMedVQA", "PMC_VQA"]:
        judge_prompt = f'''You are an objective evaluator. Your SOLE task is to compare the "Model Prediction" against the "Standard Answer" to determine if they match.
The question is: {question}
---
**Standard Answer**: {answer}
---
**Model Prediction**: 
{response_text}
---
**Evaluation Criteria (Read Carefully)**:
1. **Trust the Standard Answer**: Treat the Standard Answer as the ABSOLUTE TRUTH. Even if you think the Standard Answer is factually incorrect, mathematically wrong, or has unit errors based on the Question, you MUST judge solely on whether the Model Prediction matches the Standard Answer provided above.
2. **Do Not Re-solve**: Do not calculate the math yourself. Do not try to correct the Standard Answer.
3. **Numerical Matching**: 
   - If the Standard Answer is "1.2" and the Model Prediction is "0.0", this is a MISMATCH -> Output 1 (Incorrect).
   - Equivalent formats are allowed (e.g., "1.2" == "1.20", "1/2" == "0.5").
   - Unit conversions are allowed ONLY if the value becomes identical (e.g., "100 cm" == "1 m").
4. **Final Conclusion**: 
   - Focus on the final boxed value or the last sentence of the prediction.
   - If the prediction's final value does not match the Standard Answer, it is Incorrect.
**Output Format**:
<think>
1. Identify the final value in the Model Prediction.
2. Compare it strictly with the Standard Answer.
3. State whether they match.
</think>
<judge>{{0/1}}</judge>  (0 for Correct, 1 for Incorrect)'''
    else:
        judge_prompt = f"""
Your task is to determine whether the user's answer is correct based on the provided questions and standard answers (for example, if the user expresses a similar meaning to the standard answer, or another interpretation of the standard answer, it is considered correct.)

The question is: {question}

The standard answer: {answer}

The user's answer: {response_text}

Please strictly follow the following format for output(0 represents correct, 1 represents incorrect):
<think>{{your concise think step}}</think>
<judge>{{0/1}}</judge>

for example:
<think>The standard answer is right, and the user's answer is right frontal lobe, they express the same meaning, so it is correct.</think>
<judge>0</judge>
    """
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0
    )
    return completion.choices[0].message.content

def parse_judge_output(judge_output):
    if not judge_output or judge_output == "Error":
        return False
    try:
        match = re.search(r"<judge>\s*(\d+)\s*</judge>", judge_output)
        if match:
            return int(match.group(1)) == 0
        else:
            if "<judge>0</judge>" in judge_output: return True
            return False
    except:
        return False


def process_single_inference(item, cfg, test_client):
    image = item['messages']['image']

    question = item['question']
    standard_answer = str(item['answer'])

    prompt_text = item['messages']['prompt']
    image_b64 = encode_image(image)
    
    model_response = retry_request(
        get_model_response,
        cfg["max_retries"],
        cfg["retry_delay_base"],
        test_client,
        cfg["test_model_name"],
        image_b64,
        prompt_text,
    )
    
    if model_response is None:
        model_response = "[FAILED] Model Inference Failed"

    return {
        "image_id": item.get('qid', 'unknown'),
        "answer": standard_answer,
        "question": question,
        "prompt": prompt_text,
        "response": model_response
    }

def process_single_judge(item, cfg, judge_client):
    if item['response'] == "[FAILED] Model Inference Failed":
        judger_output = "<think>Inference failed previously.</think><judge>1</judge>"
    else:
        judger_output = retry_request(
            get_judge_result,
            cfg["max_retries"],
            cfg["retry_delay_base"],
            judge_client,
            cfg["judge_model_name"],
            item['question'],
            item['answer'],
            item['response'],
            cfg["dataset_name"]
        )
        if judger_output is None:
            judger_output = "<think>Judge API failed.</think><judge>1</judge>"
    
    is_correct = parse_judge_output(judger_output)
    
    return {
        "answer": item['answer'],
        "question": item['question'],
        "prompt": item['prompt'],
        "response": item['response'],
        "correct": is_correct,
        "judger_output": judger_output
    }


# ================= Main Process =================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM on the full medical VQA dataset with LLM-as-Judge.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--test-model-name", default=TEST_MODEL_NAME)
    parser.add_argument("--judge-model-name", default=JUDGE_MODEL_NAME)
    parser.add_argument("--is-reasoning", type=parse_bool, default=IS_REASONING)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--retry-delay-base", type=float, default=RETRY_DELAY_BASE)
    parser.add_argument("--openrouter-api-key", default=OPENROUTER_API_KEY)
    parser.add_argument("--openrouter-base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument("--deepseek-api-key", default=DEEPSEEK_API_KEY)
    parser.add_argument("--deepseek-base-url", default=DEEPSEEK_BASE_URL)
    return parser.parse_args()


def main():
    args = parse_args()
    _, results_file, metrics_file = build_paths(
        args.test_model_name, args.dataset_name, args.is_reasoning
    )

    cfg = {
        "dataset_name": args.dataset_name,
        "test_model_name": args.test_model_name,
        "judge_model_name": args.judge_model_name,
        "is_reasoning": args.is_reasoning,
        "max_workers": args.max_workers,
        "max_retries": args.max_retries,
        "retry_delay_base": args.retry_delay_base,
    }

    print("Loading the complete dataset...")
    data_loader = DiffDataLoader(cfg["is_reasoning"])
    dataset = data_loader.load_diff_data(args.dataset_name)
    total_samples = len(dataset)
    print(f"Complete dataset loaded: {total_samples} samples")

    test_client = OpenAI(api_key=args.openrouter_api_key, base_url=args.openrouter_base_url)
    judge_client = OpenAI(api_key=args.deepseek_api_key, base_url=args.deepseek_base_url)

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate_results = []

    print(f"\nStarting Step 1: Model Inference ({args.test_model_name}) - Threads: {args.max_workers}, Retries: {args.max_retries}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_single_inference, item, cfg, test_client) for item in dataset]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Inference"):
            try:
                result = future.result()
                intermediate_results.append(result)
            except Exception as e:
                print(f"Critical error in task: {e}")

    print(f"\nStarting Step 2: Judge Evaluation ({args.judge_model_name}) - Threads: {args.max_workers}, Retries: {args.max_retries}...")
    final_results = []
    correct_count = 0
    total_count = len(intermediate_results)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_single_judge, item, cfg, judge_client) for item in intermediate_results]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Judging"):
            try:
                result = future.result()
                final_results.append(result)
                if result["correct"]:
                    correct_count += 1
            except Exception as e:
                print(f"Critical error in task: {e}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    metrics = {
        "run_id": run_id,
        "dataset_info": {
            "mode": "full_dataset",
            "total_samples": total_samples
        },
        "metrics": {
            "total": total_count,
            "right": correct_count,
            "acc": accuracy
        }
    }

    print(f"Complete dataset evaluation finished, Accuracy: {accuracy:.4f}")
    print("\nSaving results...")

    existing_results = load_json_file(results_file, {})
    if not isinstance(existing_results, dict):
        existing_results = {}
    existing_results.setdefault("dataset", args.dataset_name)
    existing_results.setdefault("split", "test")
    existing_results.setdefault("history", [])
    existing_results["history"].append({
        "run_id": run_id,
        "total_samples": total_samples,
        "results": final_results
    })

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=4, ensure_ascii=False)

    existing_metrics = load_json_file(metrics_file, {})
    if not isinstance(existing_metrics, dict):
        existing_metrics = {}
    existing_metrics.setdefault("dataset", args.dataset_name)
    existing_metrics.setdefault("split", "test")
    existing_metrics.setdefault("history", [])
    existing_metrics["history"].append(metrics)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(existing_metrics, f, indent=4, ensure_ascii=False)

    print(f"Results saved to: {results_file}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()