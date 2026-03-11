import torch
import os
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor


class LlamaVision:
    def __init__(self,model_path ,args):
        super().__init__()
        self.llm = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_new_tokens


    def process_messages(self,messages):
        prompt = messages["prompt"]
        images = messages["images"]
        video_inputs = []
        for image in images:
            if isinstance(image,str):
                image = Image.open(image)
            video_inputs.append(image)
            
        content = []
        for i,image in enumerate(images):      
            content.append({"type":"text","text":f"<image_{i+1}>: "})
            content.append({"type": "image"})
        content.append({"type":"text","text":prompt})
        messages = [{"role":"user","content":content}]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        llm_inputs = self.processor(
                video_inputs,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.llm.device)
        return llm_inputs

        return messages,video_inputs


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
