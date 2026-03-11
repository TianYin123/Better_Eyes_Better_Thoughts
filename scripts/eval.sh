#!/bin/bash

export HF_ENDPOINT="<HF_ENDPOINT>"

DATASETS_PATH="hf"
OUTPUT_PATH="./results/run/"
EVAL_DATASETS="VQA_RAD"

MODEL_NAME="Qwen3-VL"
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
USE_VLLM="True"
IFS=',' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=$TOTAL_GPUS

# Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1

# Eval LLM setting
MAX_NEW_TOKENS=16384
MAX_IMAGE_NUM=600
TEMPERATURE=0
TOP_P=0.95
REPETITION_PENALTY=1.0

# LLM judge setting
USE_LLM_JUDGE="True"

GPT_MODEL="<JUDGE_MODEL_NAME>"
JUDGE_MODEL_TYPE="<JUDGE_PROVIDER>"  
API_KEY="<API_KEY>"
BASE_URL="<BASE_URL>"


python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --use_vllm "$USE_VLLM" \
    --reasoning $REASONING \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$GPT_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES"
