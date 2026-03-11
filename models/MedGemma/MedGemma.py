from transformers import AutoProcessor, AutoModelForImageTextToText
from vllm import LLM, SamplingParams
import os
import torch

from tqdm import tqdm

from PIL import Image
class MedGemma:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def process_messages(self,messages):
        current_messages = []
        imgs = []
        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                current_messages.append({"role":role,"content":[{"type":"text","text":content}]}) 

        else:
            prompt = messages["prompt"]
            if "system" in messages:
                system_prompt = messages["system"]
                current_messages.append({"role":"system","content":[{"type":"text","text":system_prompt}]})
            if "image" in messages:
                image = messages["image"]
                if isinstance(image,str):
                    image = Image.open(image)
                imgs.append(image)
                current_messages.append({"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]})
            elif "images" in messages:
                content = []
                for i,image in enumerate(messages["images"]):
                    content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image","image":image})
                    if isinstance(image,str):
                        image = Image.open(image)
                    imgs.append(image)
                content.append({"type":"text","text":messages["prompt"]})
                current_messages.append({"role":"user","content":content})
            else:
                current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]}) 
        
        
        inputs = self.processor.apply_chat_template(
            current_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.llm.device, dtype=torch.bfloat16)

        return inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        input_len = llm_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            do_sample = False if self.temperature == 0 else True
            generation = self.llm.generate(**llm_inputs,max_new_tokens=self.max_new_tokens,do_sample = do_sample, pad_token_id= self.processor.tokenizer.eos_token_id,top_k = None,top_p = None)
            generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in tqdm(messages_list):
            result = self.generate_output(messages)
            res.append(result)
        return res
