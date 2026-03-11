import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import argparse
import os
import os.path as osp
import random
import traceback
from typing import Any, Dict, List, Union

import datetime
import json
import numpy as np
import torch
import torch.distributed as dist
from prettytable import PrettyTable
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import sys
sys.path.append(".")
from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3.model import load_pretrained_model
from videollama3.mm_utils import load_images, process_images, load_video, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollama3.model.processor import Videollama3Processor
# from evaluation.benchmarks import build_dataset
# from evaluation.register import INFERENCES
# from evaluation.utils import CUDADataLoader


from PIL import Image
import math
# os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
# os.environ.setdefault('MASTER_PORT', str(port))
# if rank == 0:
#     logger.info('init process group: '
#                 f'MASTER_ADDR={os.environ["MASTER_ADDR"]} '
#                 f'MASTER_PORT={os.environ["MASTER_PORT"]} '
#                 f'world_size={world_size}')

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"




detail_describe_instructions = [
    "Describe the following image in detail.",
    "Provide a detailed description of the given image.",
    "Give an elaborate explanation of the image you see.",
    "Share a comprehensive rundown of the presented image.",
    "Offer a thorough analysis of the image.",
    "Explain the various aspects of the image before you.",
    "Clarify the contents of the displayed image with great detail.",
    "Characterize the image using a well-detailed description.",
    "Break down the elements of the image in a detailed manner.",
    "Walk through the important details of the image.",
    "Portray the image with a rich, descriptive narrative.",
    "Narrate the contents of the image with precision.",
    "Analyze the image in a comprehensive and detailed manner.",
    "Illustrate the image through a descriptive explanation.",
    "Examine the image closely and share its details.",
    "Write an exhaustive depiction of the given image.",
]

concise_describe_instructions = [
    "Describe the following image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the following image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo below.",
    "Write a terse but informative summary of the following picture.",
    "Create a compact narrative representing the image presented.",
]

prompt_pool = detail_describe_instructions + concise_describe_instructions

prompt_pool = [ "Describe the following image in detail."]

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28
) -> tuple[int, int]:
    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
IMAGE_FACTOR = 28
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)





def eval_model(args):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name,device_map='cuda:0')
    model.config.use_token_compression=False
    processor = Videollama3Processor(image_processor, tokenizer)

    #tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    if args.return_gating_logit is not None:
        from moellava.utils import get_gating_logit_by_hook
        print(model)
        fea_hooks = get_gating_logit_by_hook(model)
        all_gating_logits = {}
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    cnt = -1
    for i, line in enumerate(tqdm(questions)):
        cnt += 1
        question_type = line.get('type', 'Default')
        openclose= line.get('Question_Type', 'open')
        # if question_type!=5:
        #     continue
       # idx = line["id"]
        # question = line['conversations'][0]
        # gt_ans = line["conversations"][1]
        try:
            question = line["conversations"][0] # ['value'].split('\n')[0]
            gt_ans = line["conversations"][1]['value'] # ['value']        
        except:
            question = line["conversatons"][0] # ['value'].split('\n')[0]
            gt_ans = line["conversatons"][1] # ['value']    

        qs = question['value']

        qs = qs.replace('<image>\n', '').strip()
        qs = qs.replace('<video>\n', '').strip()
        cur_prompt = qs

        image_file = line["video"][0]
        #image, timestamps = load_video(image_file, fps=1, max_frames=180)
        try:
            image, timestamps = load_video(image_file, fps=1, max_frames=1800)
        except Exception as e:
            print(f"Error: {e}, qs: {qs}")
            continue
        image_entries = [{"type": "video", "num_frames": len(image)}]
        modal='video'
        if(line['Question_Type']=='close' or line['Question_Type']=='CLOSED'):
            #qs=qs+"\n"+"Answer with the option's letter from the given choices directly."
            #image_entries = [{"type": "image"} for _ in range(len_img)]

            # conversation = [{
            #         "role": "user",
            #         "content": image_entries + [
            #             {
            #                 "type": "text",
            #                 "text": qs,
            #             }
            #         ]
            #     }]
            conversation = [{
                    "role": "user",
                    "content": image_entries + [
                        {
                            "type": "text",
                            "text": qs+"\nAnswer with the option's letter from the given choices directly.",
                        }
                    ]
                }]
            #modal='image'
            inputs = processor(
                images=[image] if modal != "text" else None,
                text=conversation,
                merge_size=2 if modal == "video" else 1,
                return_tensors="pt"
                )
            inputs = {k: v.cuda().to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        modals=[modal],
                        temperature=0,
                        max_new_tokens=8192,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        else:
            #qs = qs + '\n' + "Answer with a single word or short phrase"
            # image_entries = [{"type": "image"} for _ in range(len_img)]

            # conversation = [{
            #         "role": "user",
            #         "content": image_entries + [
            #             {
            #                 "type": "text",
            #                 "text": qs,
            #             }
            #         ]
            #     }]
            # modal='image'
            conversation = [{
                    "role": "user",
                    "content": image_entries + [
                        {
                            "type": "text",
                            "text": qs+"\nAnswer the question using a single word or phrase.",
                        }
                    ]
                }]
            inputs = processor(
                images=[image] if modal != "text" else None,
                text=conversation,
                merge_size=2 if modal == "video" else 1,
                return_tensors="pt"
                )
            inputs = {k: v.cuda().to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        modals=[modal],
                        temperature=0,
                        max_new_tokens=8192,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({#"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "question_type": question_type,
                                   "openclose":openclose, 
                                   "gt": gt_ans,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    if args.return_gating_logit is not None:
        torch.save(all_gating_logits, f'{args.return_gating_logit}.pt')
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--return_gating_logit", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
# import argparse
# import torch
# import os
# import json
# from tqdm import tqdm
# import shortuuid
# import sys
# import traceback
# import math

# # 设置镜像
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# # 导入videollama3的工具函数
# sys.path.append(".")
# from videollama3.mm_utils import (
#     load_video, 
#     get_model_name_from_path
# )

# from transformers import AutoModelForCausalLM, AutoProcessor

# def split_list(lst, n):
#     """Split a list into n (roughly) equal-sized chunks"""
#     chunk_size = math.ceil(len(lst) / n)
#     return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]

# def eval_model(args):
#     # 初始化模型
#     print(f"Loading model from {args.model_path}...")
    
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_path,
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16,
#         device_map=f"cuda:{args.device}",
#         attn_implementation="flash_attention_2",
#     )
    
#     processor = AutoProcessor.from_pretrained(
#         args.model_path,
#         trust_remote_code=True
#     )
    
#     model.eval()
    
#     # 加载问题
#     questions = json.load(open(os.path.expanduser(args.question_file), "r"))
#     questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
#     # 准备输出
#     answers_file = os.path.expanduser(args.answers_file)
#     os.makedirs(os.path.dirname(answers_file), exist_ok=True)
#     ans_file = open(answers_file, "w")

#     for i, line in enumerate(tqdm(questions, desc="Processing")):
#         question_type = line.get('type', 'Default')
#         openclose = line.get('Question_Type', 'open')
        
#         # 提取问题和答案
#         try:
#             question = line["conversations"][0]
#             gt_ans = line["conversations"][1]['value']
#         except:
#             question = line["conversatons"][0]
#             gt_ans = line["conversatons"][1]

#         qs = question['value']
#         qs = qs.replace('<image>\n', '').strip()
#         qs = qs.replace('<video>\n', '').strip()
        
#         cur_prompt = qs
        
#         # 获取视频路径
#         image_file = line["video"][0]
        
#         # 使用videollama3的load_video加载视频
#         try:
#             image, timestamps = load_video(image_file, fps=1, max_frames=180)
#             print(f"[{i+1}/{len(questions)}] Loaded {len(image)} frames")
#         except Exception as e:
#             print(f"Error loading video: {e}, qs: {qs}")
#             traceback.print_exc()
#             continue
        
#         # 构建image_entries（与原代码一致）
#         image_entries = [{"type": "video", "num_frames": len(image)}]
#         modal = 'video'
        
#         # 根据问题类型构建conversation
#         if openclose in ['close', 'CLOSED']:
#             conversation = [{
#                 "role": "user",
#                 "content": image_entries + [
#                     {
#                         "type": "text",
#                         "text": qs + "\nAnswer with the option's letter from the given choices directly.",
#                     }
#                 ]
#             }]
#         else:
#             conversation = [{
#                 "role": "user",
#                 "content": image_entries + [
#                     {
#                         "type": "text",
#                         "text": qs + "\nPlease answer the question concisely.",
#                     }
#                 ]
#             }]
        
#         try:
#             # 使用processor处理（与原代码完全一致）
#             inputs = processor(
#                 images=[image] if modal != "text" else None,
#                 text=conversation,
#                 merge_size=2 if modal == "video" else 1,
#                 return_tensors="pt"
#             )
            
#             # 移动到GPU
#             inputs = {k: v.to(f"cuda:{args.device}") if isinstance(v, torch.Tensor) else v 
#                      for k, v in inputs.items()}
            
#             if "pixel_values" in inputs:
#                 inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
#             # 生成（与原代码一致）
#             with torch.inference_mode():
#                 output_ids = model.generate(
#                     **inputs,
#                     do_sample=False,
#                     modals=[modal],
#                     temperature=0,
#                     max_new_tokens=512,
#                     use_cache=True,
#                     pad_token_id=processor.tokenizer.eos_token_id,
#                 )
            
#             # 解码
#             outputs = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
#         except Exception as e:
#             print(f"Error during inference: {e}")
#             traceback.print_exc()
#             outputs = "Error during inference"
        
#         print(outputs)
        
#         # 保存结果（与原代码一致）
#         ans_id = shortuuid.uuid()
#         ans_file.write(json.dumps({
#             "prompt": cur_prompt,
#             "text": outputs,
#             "question_type": question_type,
#             "openclose": openclose,
#             "gt": gt_ans,
#             "answer_id": ans_id,
#             "model_id": os.path.basename(args.model_path),
#             "metadata": {}
#         }) + "\n")
#         ans_file.flush()
    
#     ans_file.close()
#     print(f"\n✅ Results saved to {answers_file}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, default="ZJU-AI4H/Hulu-Med-7B")
#     parser.add_argument("--question-file", type=str, required=True)
#     parser.add_argument("--answers-file", type=str, required=True)
#     parser.add_argument("--num-chunks", type=int, default=1)
#     parser.add_argument("--chunk-idx", type=int, default=0)
#     parser.add_argument("--device", type=int, default=0)
#     parser.add_argument("--temperature", type=float, default=0)
#     parser.add_argument("--top_p", type=float, default=None)
#     parser.add_argument("--num_beams", type=int, default=1)
#     parser.add_argument("--max_new_tokens", type=int, default=512)
#     args = parser.parse_args()

#     eval_model(args)
