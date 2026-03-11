from vllm import LLM, SamplingParams
import os
from PIL import Image

from .utils import DEFAULT_IMAGE_TOKEN,download
from .conversation import conv_templates

class LLavaMed:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model= model_path,
            tensor_parallel_size= int(os.environ.get("tensor_parallel_size",1)),
            enforce_eager=True,
            limit_mm_per_prompt = {"image": int(os.environ.get("max_image_num",1))},
            # limit_mm_per_prompt = {"image":10}
        )

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )

    def process_messages(self,messages):
        conv = conv_templates["mistral_instruct"].copy()
        conv.messages = []
        if "system" in messages:
            conv.system = messages["system"]
        else:
            conv.system = ""
        
        imgs = []
        if "image" in messages:
            text = messages["prompt"]
            inp = DEFAULT_IMAGE_TOKEN + '\n' + text
            conv.append_message(conv.roles[0],inp)
            image = messages["image"]
            if isinstance(image,str):
                if os.path.exists(image):
                    image = Image.open(image)
            imgs.append(image)
        elif "images" in messages:
            text = messages["prompt"]
            images = messages["images"]
            inp = ""
            for i,image in enumerate(images):
                inp = inp + f"<image_{i+1}>: " +DEFAULT_IMAGE_TOKEN + '\n' 
                if isinstance(image,str):
                    if os.path.exists(image):
                        image = Image.open(image)
                imgs.append(image)
            inp = inp + text
            conv.append_message(conv.roles[0],inp)
        else:
            text = messages["prompt"]
            inp = text
            conv.append_message(conv.roles[0],inp)
        
        conv.append_message(conv.roles[1],None) 
        prompt = conv.get_prompt()
        # from pdb import set_trace;set_trace()

        mm_data = {}
        if len(imgs) > 0:
            # image_inputs = process_images(imgs, self.image_processor, self.config)
            mm_data["image"] = imgs


        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_outputs(self,messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        # from pdb import set_trace;set_trace()
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res
