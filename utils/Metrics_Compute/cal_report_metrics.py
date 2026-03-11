from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import BERTScorer
from RaTEScore import RaTEScore
from tqdm import tqdm
import json

import os
import nltk

import numpy as np


# caculate the metrics of ratescore, green score, cider, bleu, rouge, meteor
# other metrics please refer to https://github.com/rajpurkarlab/CXR-Report-Metric

#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['PYTHONIOENCODING'] = 'utf-8'

model_name = "eval_hulumed"
datasets = ["IU_XRAY","CheXpert_Plus","MIMIC_CXR"]
ratescore = RaTEScore(bert_model = "RaTE-NER-Deberta",eval_model='BioLORD-2023-C')
rouge_scorer = Rouge()
def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]

for dataset in datasets:
        results_path = f".{model_name}/{dataset}/results.json"
        output_metrics_path = f".{model_name}/{dataset}/metrics_2.json"
        if not os.path.exists(results_path):
                print(f"Results file {results_path} does not exist.")
                continue

        with open(results_path,"r", encoding='utf-8') as f:
                datas = json.load(f)

        reports = []
        reports2 = []
        preds = []

        total_bleu1 = 0
        total_bleu2 = 0
        total_bleu3 = 0
        total_bleu4 = 0

        total_rouge1 = 0
        total_rouge2 = 0
        total_rougel = 0

        total_precision = 0
        total_recall = 0
        total_f1 = 0

        total_meteor_scores = 0

        messages_list = []
        for sample in tqdm(datas):
                response = sample["response"]
                response = response.replace('Findings:','')
                response = response.replace('Impression:','')
                if response == "":
                        continue
                golden = sample['conversations'][1]['value']

                tokenized_response =prep_reports([response.lower()])[0]
                tokenized_golden = prep_reports([golden.lower()])[0] 

                reports.append(golden)
                reports2.append([golden.lower()])

                preds.append(response)

                bleu1 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1])
                bleu2 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.5,0.5])
                bleu3 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1/3,1/3,1/3])
                bleu4 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.25,0.25,0.25,0.25])
                total_bleu1 += bleu1
                total_bleu2 += bleu2
                total_bleu3 += bleu3
                total_bleu4 += bleu4

                rouge_scores = rouge_scorer.get_scores(response.lower()[:2048], golden.lower())
                total_rouge1 += rouge_scores[0]["rouge-1"]["f"]
                total_rouge2 += rouge_scores[0]["rouge-2"]["f"]
                total_rougel += rouge_scores[0]["rouge-l"]["f"]

                meteor_scores = single_meteor_score(hypothesis = tokenized_response,reference  = tokenized_golden)
                total_meteor_scores += meteor_scores



        print("begin to compute RaTE score...")
        rate_scores = ratescore.compute_score(preds, reports)
        rate_score = sum(rate_scores)/len(rate_scores)

        print("begin to compute Green score...")


        print(f"Meteor Score: {total_meteor_scores/len(datas)}")
        print(f"RaTE Score: {rate_score}")

        print("Metrics computed successfully!")
        print(f"Bleu-1: {total_bleu1/len(datas)}")
        print(f"Bleu-2: {total_bleu2/len(datas)}")
        print(f"Bleu-3: {total_bleu3/len(datas)}")
        print(f"Bleu-4: {total_bleu4/len(datas)}")
        print(f"Rouge-1: {total_rouge1/len(datas)}")
        print(f"Rouge-2: {total_rouge2/len(datas)}")
        print(f"Rouge-L: {total_rougel/len(datas)}")

        metrics = {
                "meteor": float(total_meteor_scores/len(datas)),
                "bleu1": float(total_bleu1/len(datas)),
                "bleu2": float(total_bleu2/len(datas)),
                "bleu3": float(total_bleu3/len(datas)),
                "bleu4": float(total_bleu4/len(datas)),
                "rouge1": float(total_rouge1/len(datas)),
                "rouge2": float(total_rouge2/len(datas)),
                "rougeL": float(total_rougel/len(datas)),
                "rate": float(rate_score),
        }


        with open(output_metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)