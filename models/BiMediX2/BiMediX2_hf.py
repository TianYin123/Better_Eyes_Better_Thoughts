from io import BytesIO
import requests
from PIL import Image
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

class BiMediX2:
    def __init__(self,model_path='/path/to/local_model',args=None):
        super().__init__()
        model_name = "BiMediX2"
        device = "cuda"
        device_map = "auto"

        _x,_y,image_processor, _z = load_pretrained_model("/path/to/aux_model", None, "llavamistral", device_map=device_map)

        tokenizer, model, _i, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.max_length = max_length
        self.conv_template = "llava_llama_3"

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        '''
        self.temperature = 1
        self.top_p = 1
        self.repetition_penalty = 1.0
        self.max_new_tokens = 1024
        '''
    def process_messages(self,messages):
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.tokenizer = self.tokenizer
        conv.messages = []
        if  "system" in messages:
            conv.system = messages["system"]
        
        imgs = []
        if "image" in messages:
            image = messages["image"]
            if isinstance(image,str):
                image = Image.open(image).convert('RGB')
            else:
                image = image.convert('RGB')
            imgs.append(image)
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + messages["prompt"]
        elif "images" in messages:
            images = messages["images"]
            prompt = ""
            for i,image in enumerate(images):
                prompt += f"<image_{i+1}>: " + DEFAULT_IMAGE_TOKEN + '\n'
                if isinstance(image,str):
                    if os.path.exists(image):
                        image = Image.open(image)
                    else:
                        image = download(image)
                elif isinstance(image,Image.Image):
                    image = image.convert("RGB")
                imgs.append(image)
            prompt += messages["prompt"]
        else:
            prompt = messages["prompt"]

        conv.append_message(conv.roles[0],prompt)
        conv.append_message(conv.roles[1],None) 
        prompt = conv.get_prompt()

        imgs = None if len(imgs) == 0 else imgs
        return prompt,imgs


    def generate_output(self,messages):
        prompt,imgs = self.process_messages(messages)
        if imgs:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            imgs = process_images(imgs, self.image_processor, self.model.config)
            imgs = [_image.to(dtype=torch.float16, device="cuda") for _image in imgs]
        else:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            imgs = None

        with torch.inference_mode():
            do_sample = False if self.temperature == 0 else True
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample = do_sample,
                temperature = self.temperature,
                top_p = self.top_p,
                repetition_penalty = self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def generate_outputs(self,messages_list):
        outputs = []
        for messages in tqdm(messages_list):
            output = self.generate_output(messages)
            outputs.append(output)
        return outputs
        