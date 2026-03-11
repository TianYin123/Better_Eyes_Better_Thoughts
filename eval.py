import os
import torch
import sys
sys.setrecursionlimit(50000)
import random
import numpy as np
from tqdm import tqdm  
import json
from argparse import ArgumentParser
from LLMs import init_llm
import gc

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_eval_datasets(datasets_str):
    return datasets_str.split(',')

def main():
    parser = ArgumentParser()
    parser.add_argument('--eval_datasets', type=parse_eval_datasets, default=['MMMU'],
                    help='name of eval dataset')
    parser.add_argument('--datasets_path', type=str, default="benchmarks",
                    help='path of eval dataset')
    parser.add_argument('--output_path', type=str, default='eval_results/Qwen2-VL-7B-Instruct',
                        help='name of saved json')
    parser.add_argument('--model_name', type=str, default='Qwen2-VL-7B-Instruct',
                        help='name of model')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2-VL-7B-Instruct")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda_visible_devices', type=str, default=None)
    parser.add_argument('--tensor_parallel_size', type=str, default="1")
    parser.add_argument('--use_vllm', type=str, default="True")
    parser.add_argument('--reasoning', type=str, default="False")

    parser.add_argument('--num_chunks', type=str, default="1")
    parser.add_argument('--chunk_idx', type=str, default="0")

    parser.add_argument('--max_image_num', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.001)
    parser.add_argument('--repetition_penalty', type=float, default=1)

    parser.add_argument('--test_times', type=int, default=1)

    # parser.add_argument('--use_llm_judge', type=str, default="False")
    # parser.add_argument('--judge_gpt_model', type=str,default = "None")

    # parser.add_argument('--openai_api_key', type=str,default = "None")

    parser.add_argument('--use_llm_judge', type=str, default="False")
    parser.add_argument('--judge_model_type', type=str, default="openai")
    parser.add_argument('--judge_model', type=str,default = "None")
    parser.add_argument('--judge_gpt_model', type=str, default="None")
    parser.add_argument('--api_key', type=str,default = "None")
    parser.add_argument('--openai_api_key', type=str, default="None")
    parser.add_argument('--base_url', type=str, default= "None")
    parser.add_argument('--chat_template_path', type=str, default=None,
                        help='Path to custom chat template jinja file (for completion mode)')
    parser.add_argument('--completion_mode', action='store_true',
                        help='Enable completion mode (continue writing) instead of chat mode')
    parser.add_argument('--dataset', type=str, default= "None")
    parser.add_argument('--test_api_key', type=str, default=os.environ.get('TEST_API_KEY', 'None'))
    parser.add_argument('--test_base_url', type=str, default=os.environ.get('TEST_BASE_URL', 'None'))
    parser.add_argument('--test_max_workers', type=int, default=int(os.environ.get('TEST_MAX_WORKERS', '8')))
    parser.add_argument('--test_max_retries', type=int, default=int(os.environ.get('TEST_MAX_RETRIES', '8')))
    parser.add_argument('--test_retry_delay_base', type=float, default=float(os.environ.get('TEST_RETRY_DELAY_BASE', '2.0')))
    parser.add_argument('--test_image_detail', type=str, default=os.environ.get('TEST_IMAGE_DETAIL', 'auto'))
    parser.add_argument('--test_timeout', type=float, default=float(os.environ.get('TEST_TIMEOUT', '600')))
    

    args = parser.parse_args()

    judge_model_arg = args.judge_model if args.judge_model != "None" else args.judge_gpt_model
    api_key_arg = args.api_key if args.api_key != "None" else args.openai_api_key

    os.environ["VLLM_USE_V1"] = "0"

    # llm judge setting
    # if args.openai_api_key == "None" and args.use_llm_judge == "True":
    #     raise ValueError("If you want to use llm judge, please set the openai api key")
    

    # os.environ["judge_gpt_model"] = args.judge_gpt_model
    # os.environ["use_llm_judge"] = args.use_llm_judge
    # os.environ["openai_api_key"] = args.openai_api_key

    if api_key_arg == "None" and args.use_llm_judge == "True":
        raise ValueError("If you want to use llm judge, please set the openai api key")
    os.environ["use_llm_judge"] = args.use_llm_judge
    os.environ["judge_model_type"] = args.judge_model_type
    os.environ["judge_model"] = judge_model_arg
    os.environ["api_key"] = api_key_arg
    os.environ["base_url"] = args.base_url


    # eval modle setting
    os.environ["REASONING"] = args.reasoning
    os.environ["use_vllm"] = args.use_vllm

    os.environ["max_image_num"] = str(args.max_image_num)

    os.environ["DATASET"] = args.dataset
    os.environ["test_api_key"] = args.test_api_key
    os.environ["test_base_url"] = args.test_base_url
    os.environ["test_max_workers"] = str(args.test_max_workers)
    os.environ["test_max_retries"] = str(args.test_max_retries)
    os.environ["test_retry_delay_base"] = str(args.test_retry_delay_base)
    os.environ["test_image_detail"] = args.test_image_detail
    os.environ["test_timeout"] = str(args.test_timeout)


    # vllm and torch setting
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.use_vllm == "True":
        os.environ["tensor_parallel_size"] = args.tensor_parallel_size
        if int(args.tensor_parallel_size) > 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        torch.multiprocessing.set_start_method('spawn')
        os.environ["num_chunks"] = args.num_chunks
        os.environ["chunk_idx"] = args.chunk_idx


    os.makedirs(args.output_path, exist_ok=True)

    print('initializing LLM...')
    model = init_llm(args)
    
    total_results_path = os.path.join(args.output_path,'total_results.json')

    from benchmarks import prepare_benchmark

    for eval_dataset in tqdm(args.eval_datasets):
        set_seed(args.seed)
        print(f'evaluating on {eval_dataset}...')

        eval_dataset_path = os.path.join(args.datasets_path,eval_dataset) if args.datasets_path != "hf" else None
        eval_output_path = os.path.join(args.output_path,eval_dataset)
        os.makedirs(eval_output_path, exist_ok=True)
        benchmark = prepare_benchmark(model,eval_dataset,eval_dataset_path,eval_output_path)
        benchmark.load_data()
        final_results = benchmark.eval() if benchmark else {}
        print(f'final results on {eval_dataset}: {final_results}')
        # if final_results is not None:
        #     if os.path.exists(total_results_path):
        #         with open (total_results_path,"r",encoding="utf-8") as f:
        #             total_results = json.load(f)
        #     else:
        #         total_results = {}
        #     total_results[eval_dataset] = final_results

        #     with open(os.path.join(args.output_path,'total_results.json'),'w') as f:
        #         json.dump(total_results,f,indent=4)
        #gc.collect()

if __name__ == '__main__':
    main()
