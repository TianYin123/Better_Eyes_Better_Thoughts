import json
import collections
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score


from tabulate import tabulate
import re
import warnings
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score

warnings.simplefilter('ignore')

QUESTION_TYPE_MAPPING = {
    1: "Plane",
    2: "Phase", 
    3: "Organ",
    4: "Abnormality",
    5: "Location"
}

THREERAD_TYPE_MAPPING = {
    "Medical_Computation": "Medical Computation",
    "Spatial_Relationship": "Spatial Relationship",
    "Abnormality_Detection": "Abnormality Detection",
    "Organ_Identification": "Organ Identification",
    "Image_Quality": "Image Quality"
}

def extract_choice_letter(text):
    text = str(text).lower().strip()
    match = re.match(r'^([a-d])[.\):]?\s*', text)
    if match:
        return match.group(1)
    if text in ['a', 'b', 'c', 'd']:
        return text
    return text

def _evaluate_core(out_samples, desc, category_key, category_name_mapping, calculate_overall=True):
    scores_by_type = collections.defaultdict(lambda: collections.defaultdict(list))
    closed_questions_count_by_type = collections.defaultdict(int)
    closed_questions_correct_by_type = collections.defaultdict(int)
    wrong_answers_by_type = collections.defaultdict(list)
    total_open_count_by_type = collections.defaultdict(int)

    rouge = Rouge()
    chencherry = SmoothingFunction()

    for pred_item in tqdm(out_samples, desc=desc):
        try:
            gt_value = pred_item['conversations'][1]['value']
            question = pred_item['conversations'][0]['value']
        except (KeyError, IndexError):
            print(f"Warning: Invalid sample format, skipping sample: {pred_item}")
            continue
        
        pred_value = pred_item.get('response', '')
        category = pred_item.get(category_key)
        openclose_type = pred_item.get('Question_Type', 'OPEN').upper()

        if openclose_type in ['OPEN']:
            if category is not None:
                total_open_count_by_type[category] += 1
            
            gt_lower = str(gt_value).strip().lower()
            pred_lower = str(pred_value).strip().lower()
            
            f1, precision, recall = calculate_f1score(pred_lower, gt_lower)
            
            try:
                rouge_scores = rouge.get_scores(pred_lower, gt_lower)[0] if pred_lower else {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}
                rouge_1, rouge_2, rouge_l = rouge_scores['rouge-1']['f'], rouge_scores['rouge-2']['f'], rouge_scores['rouge-l']['f']
            except (ValueError, KeyError):
                rouge_1, rouge_2, rouge_l = 0, 0, 0
            
            try:
                reference = [gt_lower.split()]
                candidate = pred_lower.split()
                bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
                bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
                bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
                bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
            except Exception:
                bleu_1, bleu_2, bleu_3, bleu_4 = 0, 0, 0, 0

            try:
                meteor = meteor_score([gt_lower.split()], pred_lower.split()) if pred_lower else 0
            except Exception:
                meteor = 0

            metrics_to_store = {
                'exact_match': calculate_exactmatch(pred_lower, gt_lower), 'f1': f1, 
                'precision': precision, 'recall': recall, 'rouge_1': rouge_1, 'rouge_2': rouge_2, 
                'rouge_l': rouge_l, 'bleu_1': bleu_1, 'bleu_2': bleu_2, 'bleu_3': bleu_3, 
                'bleu_4': bleu_4, 'meteor': meteor
            }
            
            if category is not None:
                for key, value in metrics_to_store.items():
                    scores_by_type[category][key].append(value)

        elif openclose_type in ['CLOSED', 'CLOSE']:
            if category is not None:
                closed_questions_count_by_type[category] += 1
            
            answer_letter = extract_choice_letter(gt_value)
            response_letter = extract_choice_letter(pred_value)
            is_correct = (answer_letter == response_letter)

            if is_correct and category is not None:
                closed_questions_correct_by_type[category] += 1
            
            if not is_correct and category is not None:
                wrong_answer_log = {
                    'question': question,
                    'correct_answer': gt_value,
                    'predicted_answer': pred_value,
                }
                if 'sub-type' in pred_item:
                    wrong_answer_log['sub_type'] = pred_item.get('sub-type')
                wrong_answers_by_type[category].append(wrong_answer_log)

    def _format_metrics_to_table(avg_metrics):
        table_data = [
            ['Exact Match Score', f"{avg_metrics.get('exact_match', 0) * 100:.4f}"],
            ['F1 Score', f"{avg_metrics.get('f1', 0) * 100:.4f}"],
            ['Precision', f"{avg_metrics.get('precision', 0) * 100:.4f}"],
            ['Recall', f"{avg_metrics.get('recall', 0) * 100:.4f}"],
            ['ROUGE-1', f"{avg_metrics.get('rouge_1', 0) * 100:.4f}"],
            ['ROUGE-2', f"{avg_metrics.get('rouge_2', 0) * 100:.4f}"],
            ['ROUGE-L', f"{avg_metrics.get('rouge_l', 0) * 100:.4f}"],
            ['BLEU-1', f"{avg_metrics.get('bleu_1', 0) * 100:.4f}"],
            ['BLEU-2', f"{avg_metrics.get('bleu_2', 0) * 100:.4f}"],
            ['BLEU-3', f"{avg_metrics.get('bleu_3', 0) * 100:.4f}"],
            ['BLEU-4', f"{avg_metrics.get('bleu_4', 0) * 100:.4f}"],
            ['METEOR', f"{avg_metrics.get('meteor', 0) * 100:.4f}"],
        ]
        return table_data

    def _calculate_and_format_metrics(scores_dict):
        avg_metrics = {key: (sum(lst) / len(lst)) if lst else 0 for key, lst in scores_dict.items()}
        table_data = _format_metrics_to_table(avg_metrics)
        return table_data, avg_metrics

    results_tables = []
    all_categories = sorted(list(set(scores_by_type.keys()) | set(closed_questions_count_by_type.keys())))
    category_open_metrics_dict = {}
    category_closed_accs = []
    metrics_result = {"by_category": {}}

    for cat_key in all_categories:
        category_name = category_name_mapping.get(cat_key, f"Unknown Type {cat_key}")
        
        open_table_data, open_metrics = _calculate_and_format_metrics(scores_by_type.get(cat_key, {}))
        category_open_metrics_dict[cat_key] = open_metrics
        
        closed_count = closed_questions_count_by_type.get(cat_key, 0)
        closed_correct = closed_questions_correct_by_type.get(cat_key, 0)
        closed_acc = (closed_correct / closed_count) if closed_count > 0 else 0
        if closed_count > 0:
             category_closed_accs.append(closed_acc)
        
        open_count = total_open_count_by_type.get(cat_key, 0)
        
        combined_table_data = open_table_data + [
            ['Closed Question Accuracy', f"{closed_acc * 100:.4f}" if closed_count > 0 else "N/A"],
            ['Open Questions', open_count],
            ['Closed Questions', closed_count],
            ['Total Samples', open_count + closed_count]
        ]
        
        results_tables.extend([
            f"\n{'='*60}", f"Category: {category_name}", '='*60,
            tabulate(combined_table_data, headers=['Metric', 'Performance (%)'], tablefmt='grid')
        ])
        
        metrics_result["by_category"][category_name] = {
            "open": {k: v * 100 for k, v in open_metrics.items()},
            "closed": {"accuracy": closed_acc * 100, "total": closed_count, "correct": closed_correct},
            "open_count": open_count, "closed_count": closed_count,
            "total_samples": open_count + closed_count
        }

    if calculate_overall and all_categories:
        overall_open_metrics = collections.defaultdict(list)
        for cat_metrics in category_open_metrics_dict.values():
            for key, value in cat_metrics.items():
                overall_open_metrics[key].append(value)
        
        overall_open_avg = {key: sum(values) / len(values) if values else 0 for key, values in overall_open_metrics.items()}
        overall_closed_acc_avg = sum(category_closed_accs) / len(category_closed_accs) if category_closed_accs else 0
        
        total_open_samples = sum(total_open_count_by_type.values())
        total_closed_samples = sum(closed_questions_count_by_type.values())
        total_closed_correct = sum(closed_questions_correct_by_type.values())
        
        overall_table_data = _format_metrics_to_table(overall_open_avg)
        
        overall_table_data.extend([
            ['Closed Question Accuracy (Category Avg)', f"{overall_closed_acc_avg * 100:.4f}"],
            ['Open Questions', total_open_samples],
            ['Closed Questions', total_closed_samples],
            ['Total Samples', total_open_samples + total_closed_samples],
            ['Total Categories', len(all_categories)]
        ])
        
        results_tables.extend([
            f"\n{'='*60}", "Overall Performance Summary (Category Average)", '='*60,
            tabulate(overall_table_data, headers=['Metric', 'Performance (%)'], tablefmt='grid')
        ])
        
        metrics_result["overall"] = {
            "open": {k: v * 100 for k, v in overall_open_avg.items()},
            "closed": {
                "accuracy_category_avg": overall_closed_acc_avg * 100,
                "total": total_closed_samples,
                "correct": total_closed_correct,
            },
            "total_samples": total_open_samples + total_closed_samples,
            "num_categories": len(all_categories)
        }
    
    return "\n".join(results_tables), metrics_result, wrong_answers_by_type

def evaluate_m3d(out_samples):
    return _evaluate_core(
        out_samples=out_samples,
        desc="Evaluating M3D predictions",
        category_key='type',
        category_name_mapping=QUESTION_TYPE_MAPPING,
        calculate_overall=True
    )

def evaluate_3drad(out_samples):
    return _evaluate_core(
        out_samples=out_samples,
        desc="Evaluating 3D-RAD predictions",
        category_key='type',
        category_name_mapping=THREERAD_TYPE_MAPPING,
        calculate_overall=False
    )
