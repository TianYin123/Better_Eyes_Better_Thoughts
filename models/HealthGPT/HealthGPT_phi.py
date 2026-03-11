import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import transformers
import tokenizers

from PIL import Image
from tqdm import tqdm
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


from .llava.constants import  IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from .llava.peft import LoraConfig, get_peft_model
from .llava import conversation as conversation_lib
from .llava.model import *
from .llava.mm_utils import tokenizer_image_token,process_images
from .llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM
from .utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square,com_vision_args


class HealthGPT:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_path,
        attn_implementation="flash_attention_2",
        torch_dtype= torch.float16,
        device_map="cuda"
    )
        print("load model done")
        lora_config = LoraConfig(
            r= 32,
            lora_alpha=64,
            target_modules=find_all_linear_names(self.llm),
            lora_dropout=0.0,
            bias='none',
            task_type="CAUSAL_LM",
            lora_nums=4,
        )
        self.llm = get_peft_model(self.llm, lora_config)
        print("load lora done")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=False,
        )
        print("load tokenizer done")

        num_new_tokens = add_special_tokens_and_resize_model(self.tokenizer, self.llm, 8192)
        print(f"Number of new tokens added for unified task: {num_new_tokens}")

        com_vision_args.model_name_or_path = model_path
        com_vision_args.vision_tower = '/path/to/vision_tower'
        com_vision_args.version = "phi4_instruct"

        self.llm.get_model().initialize_vision_modules(model_args=com_vision_args)
        self.llm.get_vision_tower().to(dtype=torch.float16)
        self.llm.get_model().mm_projector.to(dtype=torch.float16)
        print("load vision tower done")

        self.llm = load_weights(self.llm, "/path/to/model_weights.bin")
        print("load weights done")
        self.llm.eval()
        self.llm.to(dtype=torch.float16).cuda()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_new_tokens


    def process_messages(self,messages):
        conv = conversation_lib.conv_templates["phi4_instruct"].copy()
        conv.messages = []
        if  "system" in messages:
            conv.system = messages["system"]
        
        imgs = []
        if "image" in messages:
            image = messages["image"]
            if isinstance(image,str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image,Image.Image):
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
            # imgs = imgs[0]
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze_(0).cuda()
            imgs = [expand2square(img, tuple(int(x*255) for x in self.llm.get_vision_tower().image_processor.image_mean)) for img in imgs]
            imgs = self.llm.get_vision_tower().image_processor.preprocess(imgs, return_tensors='pt')['pixel_values'].to(dtype=torch.float16, device='cuda', non_blocking=True)
            # imgs = process_images(imgs,self.llm.get_vision_tower().image_processor,self.llm.config)
        else:
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze_(0).cuda()
            imgs = None

        with torch.inference_mode():
            do_sample = False if self.temperature == 0 else True
            output_ids = self.llm.base_model.model.generate(input_ids,images=imgs,do_sample=do_sample,num_beams=5,max_new_tokens=self.max_tokens,temperature = self.temperature,top_p = self.top_p,repetition_penalty = self.repetition_penalty,use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def generate_outputs(self,messages_list):
        outputs = []
        for messages in tqdm(messages_list):
            output = self.generate_output(messages)
            outputs.append(output)
        return outputs