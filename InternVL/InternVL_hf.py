import torch
from transformers import AutoModel, AutoTokenizer

from .utils import load_image


class InternVL:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm =  AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="cuda",
                    attn_implementation="flash_attention_2"
                    )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.generation_config ={ 
            'max_new_tokens': args.max_new_tokens,
            'repetition_penalty': args.repetition_penalty,
            'temperature' : args.temperature,
            'top_p' : args.top_p
        }
    

    def process_messages(self,messages):
        prompt = ""
        if "system" in messages:
            prompt = messages["system"]

        if "image" in messages:
            text = messages["prompt"]
            prompt = prompt + "\n" + "<image>" + '\n' + text
            image = messages["image"]
            pixel_value = load_image(image).to(torch.bfloat16).to("cuda")
        elif "images" in messages:
            text = messages["prompt"]
            images = messages["images"]
            for i,image in enumerate(images):
                prompt = prompt + f"<image_{i+1}>: <image>" + "\n"
            prompt = prompt + '\n' + text
            pixel_value = [load_image(image).to(torch.bfloat16).to("cuda") for image in images]
            pixel_value = torch.cat(pixel_value,dim=0)
        else:
            text = messages["prompt"]
            prompt = prompt + "\n" + text
            pixel_value = None

        llm_inputs = {
            "prompt": prompt,
            "pixel_values": pixel_value
        }
        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        question = llm_inputs["prompt"]
        pixel_values = llm_inputs["pixel_values"]
        response, history = self.llm.chat(self.tokenizer, pixel_values, question, self.generation_config,
                               history=None, return_history=True)
        return response
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
