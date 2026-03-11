import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

class Janus:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="cuda").eval()
        self.processor : VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_new_tokens


    def process_messages(self,messages):
        new_messages = []
        prompt = ""
        if "system" in messages:
            prompt = messages["system"]
        prompt = prompt + messages["prompt"]
        if "image" in messages:
            new_messages = [
              {
                  "role": "<|User|>",
                  "content": f"<image_placeholder>\n{prompt}",
                  "images": [messages["image"]],
              },
              {"role": "<|Assistant|>", "content": ""},
          ]
        elif "images" in messages:
            images = messages["images"]
            content = ""
            for i,image in enumerate(images):
                content = content + f"\n<image_{i+1}: <image_placeholder>" + "\n"
            content = content + prompt
            new_messages = [
              {
                  "role": "<|User|>",
                  "content": content,
                  "images": messages["images"],
              },
              {"role": "<|Assistant|>", "content": ""},
          ]
        else:
            new_messages = [
              {
                  "role": "<|User|>",
                  "content": f"{prompt}"
              },
              {"role": "<|Assistant|>", "content": ""},
          ]
        conversation = new_messages
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.llm.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.llm.prepare_inputs_embeds(**prepare_inputs)

        return {"inputs_embeds": inputs_embeds, "prepare_inputs": prepare_inputs}


    def generate_output(self,messages):
        inputs = self.process_messages(messages)
        inputs_embeds = inputs["inputs_embeds"]
        prepare_inputs = inputs["prepare_inputs"]
        do_sample = False if self.temperature == 0 else True
        outputs = self.llm.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature = self.temperature,
            top_p = self.top_p,
            repetition_penalty = self.repetition_penalty,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in tqdm(messages_list):
            result = self.generate_output(messages)
            res.append(result)
        return res
