import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from tqdm import tqdm
import os
def load_images(image_path):
        images = []
        def safe_open(f):
            try:
                with Image.open(f).convert('RGB') as img:
                    return img
            except Exception:
                pass  

        if isinstance(image_path, str) and os.path.isfile(image_path):
            img = safe_open(image_path)
            if img is not None:
                images.append(img)

        elif isinstance(image_path, str) and os.path.isdir(image_path):
            for f in sorted(os.listdir(image_path)):
                full_path = os.path.join(image_path, f)
                if os.path.isfile(full_path):
                    img = safe_open(full_path)
                    if img is not None:
                        images.append(img)

        elif isinstance(image_path, list) and isinstance(image_path[0], str):
            for f in image_path:
                img = safe_open(f)
                if img is not None:
                    images.append(img)

        elif isinstance(image_path, list) and isinstance(image_path[0], Image.Image):
            images = [img.convert('RGB') for img in image_path]

        elif isinstance(image_path, Image.Image):
            images = [image_path.convert('RGB')]

        else:
            raise ValueError(f"Unsupported image path type: {type(image_path)}")

        return images
class CustomModel_A:

    def __init__(self, model_path, args):

        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.repetition_penalty = args.repetition_penalty
    def process_messages(self, messages):
       
        prompt = messages.get("prompt", "")
        
        conversation = [{"role": "user", "content": []}]
        
        loaded_images = None

        image_paths_or_pil = messages.get("images") or ([messages["image"]] if "image" in messages else [])
        if image_paths_or_pil:
            loaded_images = load_images(image_paths_or_pil)
            if len(loaded_images) > 5:
                conversation[0]["content"].append({"type": "video", "num_frames": len(loaded_images)})
            elif 0 < len(loaded_images) <= 5:
                for _ in loaded_images:
                    conversation[0]["content"].append({"type": "image"})
        conversation[0]["content"].append({"type": "text", "text": prompt})
        inputs = self.processor(
            images=[loaded_images] if loaded_images is not None else None,
            conversation=conversation,
            add_system_prompt=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            
        return inputs


    def generate_output(self, messages):

        llm_inputs = self.process_messages(messages)
        do_sample = True if self.temperature > 0 else False
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **llm_inputs,
                do_sample=False,
                temperature=self.temperature if do_sample else 0,
                #top_p=self.top_p,
                repetition_penalty = self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        return outputs
    
    def generate_outputs(self, messages_list):

        res = []
        for messages in tqdm(messages_list, desc="Generating Outputs"):
            result = self.generate_output(messages)
            print(result)
            res.append(result)
        return res