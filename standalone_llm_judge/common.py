import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple


def load_json_list(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise ValueError(f'Expected a JSON list or object, got: {type(data)}')
    return data


def save_json(file_path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def extract_judge_bool(text: str) -> bool:
    match = re.search(r'<judge>\s*(\d)\s*</judge>', text or '')
    if match:
        return match.group(1) == '0'
    text = text or ''
    return '<judge>0' in text or '<judge> 0' in text


def remove_correct_field(records: List[Dict[str, Any]]) -> int:
    removed = 0
    for item in records:
        if 'correct' in item:
            removed += 1
        item.pop('correct', None)
        item.pop('judge_output', None)
    return removed


def normalize_prompt_text(text: str) -> str:
    text = text or ''
    return text.strip()


def get_question_text(item: Dict[str, Any]) -> str:
    if item.get('question'):
        return normalize_prompt_text(item['question'])
    if item.get('prompt'):
        return normalize_prompt_text(item['prompt'])
    if item.get('messages') and isinstance(item['messages'], dict):
        return normalize_prompt_text(item['messages'].get('prompt', ''))
    return ''


def get_choices_text(item: Dict[str, Any]) -> str:
    choices = item.get('choices', [])
    if isinstance(choices, list) and choices:
        return '\n'.join(str(c) for c in choices)
    return ''


def build_judge_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    question = get_question_text(item)
    choices_text = get_choices_text(item)
    if choices_text:
        question = f"{question}\n\nOptions:\n{choices_text}"

    standard_answer = str(item.get('answer', ''))
    user_response = str(item.get('response', ''))

    prompt = f"""
You are an objective evaluator for medical visual question answering.
Your SOLE task is to compare the model prediction against the provided standard answer and decide whether the prediction should be counted as correct.

Question:
{question}

Standard Answer:
{standard_answer}

Model Prediction:
{user_response}

Evaluation rules:
1. Treat the provided Standard Answer as the reference answer.
2. For multiple-choice questions, accept either the correct option letter or the correct option content.
3. Minor wording differences, paraphrases, abbreviations, and semantically equivalent medical expressions should be judged as correct.
4. If the prediction contains reasoning, focus on the final conclusion.
5. If the prediction is ambiguous, contradictory, or does not clearly match the Standard Answer, judge it as incorrect.
6. Do not re-answer the question. Only judge whether the prediction matches the Standard Answer.

Please strictly output in the following format only:
<think>{{brief reasoning}}</think>
<judge>{{0 or 1}}</judge>

Where:
- 0 means correct
- 1 means incorrect
""".strip()

    return [
        {'role': 'system', 'content': 'You are a precise and fair evaluator.'},
        {'role': 'user', 'content': prompt},
    ]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--input_file', type=str, default='results.json', help='Input results.json path.')
    parser.add_argument('--output_file', type=str, default=None, help='Output results.json path. Defaults to input_file.')
    parser.add_argument('--metrics_file', type=str, default='metrics_llm_judge.json', help='Metrics json path.')
    parser.add_argument('--judge_model', type=str, default='Qwen/Qwen2.5-72B-Instruct', help='Judge model name or local path.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='vLLM tensor parallel size.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='vLLM gpu memory utilization.')
    parser.add_argument('--max_tokens', type=int, default=512, help='Max new tokens for judge model.')
    parser.add_argument('--max_workers', type=int, default=16, help='CPU workers for prompt building.')
    parser.add_argument('--chunksize', type=int, default=100, help='Chunksize for process_map.')
    return parser


def aggregate_total(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    right = sum(1 for x in records if x.get('correct') is True)
    return {
        'total': total,
        'right': right,
        'acc': (right / total) if total else 0.0,
    }


def safe_ratio(a: int, b: int) -> float:
    return (a / b) if b else 0.0


def count_by_key(records: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    totals: Dict[str, int] = {}
    rights: Dict[str, int] = {}
    for item in records:
        value = str(item.get(key, 'UNKNOWN'))
        totals[value] = totals.get(value, 0) + 1
        if item.get('correct') is True:
            rights[value] = rights.get(value, 0) + 1
    return {
        k: {'total': totals[k], 'right': rights.get(k, 0), 'acc': safe_ratio(rights.get(k, 0), totals[k])}
        for k in sorted(totals.keys())
    }
