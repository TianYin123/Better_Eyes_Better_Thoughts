from transformers import OFATokenizer, OFAModel
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch


class BiomedGPT:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = OFAModel.from_pretrained(
        pretrained_model_name_or_path = model_path, ignore_mismatched_sizes=True
    ).to("cuda")

        # print(self.llm)

        self.tokenizer = OFATokenizer.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=False,
        )

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480

        self.patch_resize_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_new_tokens
        



    def process_messages(self,messages):

        prompt = ""

        if  "system" in messages:
            prompt = prompt + messages["system"] + "\n"
        
        imgs = []
        if "image" in messages:
            image = messages["image"]
            if isinstance(image,str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image,Image.Image):
                image = image.convert('RGB')
            image = self.patch_resize_transform(image).unsqueeze(0)
            imgs.append(image)
        elif "images" in messages:
            images = messages["images"]
            prompt = ""
            for i,image in enumerate(images):
                if isinstance(image,str):
                    if os.path.exists(image):
                        image = Image.open(image)
                elif isinstance(image,Image.Image):
                    image = image.convert("RGB")
                image = self.patch_resize_transform(image).unsqueeze(0)
                imgs.append(image)
            prompt += messages["prompt"]
        else:
            prompt += messages["prompt"]

        imgs = None if len(imgs) == 0 else imgs
        return prompt,imgs


    def generate_output(self,messages):
        prompt,imgs = self.process_messages(messages)
        inputs = self.tokenizer([prompt], return_tensors="pt").input_ids.to(self.llm.device)
        if imgs:
            imgs = torch.cat(imgs,dim=0).to(self.llm.device)
            output_ids = self.llm.generate(inputs, patch_images=imgs, num_beams=5, no_repeat_ngram_size=3, max_length=128)
        else:
            output_ids = self.llm.generate(inputs,  num_beams=5, no_repeat_ngram_size=3, max_length=128)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def generate_outputs(self,messages_list):
        outputs = []
        for messages in tqdm(messages_list,desc = "process messages"):
            output = self.generate_output(messages)
            outputs.append(output)
        return outputs
