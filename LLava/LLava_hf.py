import torch
import os
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
class Llava:
    def __init__(self,model_path,args):
        super().__init__()
        self.processor = LlavaNextProcessor.from_pretrained(model_path)

        self.llm = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.llm.to("cuda")

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def process_messages(self,messages):
        prompt = ""
        images = None
        if "system" in messages:
            prompt = messages["system"]
        
        prompt = prompt + "\n" + messages["prompt"]
        if "image" in messages:
            image = messages["image"]
            if isinstance(image,str):
                image = Image.open(image).convert('RGB')
            else:
                image = image.convert('RGB')
            messages = [
                    {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}]
            images = image
        elif "images" in messages:
            images = messages["images"]
            content = []
            for i,image in enumerate(images):
                if isinstance(image,str):
                    image = Image.open(image).convert('RGB')
                else:
                    image = image.convert('RGB')
                content.append({"type": "text", "text": f"<image_{i+1}>: "})
                content.append({"type": "image"})
                images.append(image)
            content.append({"type": "text", "text": prompt})
            messages = [
                    {"role": "user", "content": content}]
        else:
            messages = [
                    {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]       
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        llm_inputs = self.processor(
                images,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.llm.device)
        return llm_inputs

    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        do_sample = False if self.temperature == 0 else True
        outputs = self.llm.generate(**llm_inputs,max_new_tokens=self.max_new_tokens, do_sample=do_sample,repetition_penalty=self.repetition_penalty,temperature=self.temperature,top_p = self.top_p)
        outputs = self.processor.decode(outputs[0])
        return outputs
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
