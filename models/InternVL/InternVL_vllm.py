from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import os
from PIL import Image

from .conversations import internvl_conv

class InternVL:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model= model_path,
            trust_remote_code=True,
            tensor_parallel_size= int(os.environ.get("tensor_parallel_size",4)),
            enforce_eager=True,
            gpu_memory_utilization = 0.7,
            limit_mm_per_prompt = {"image": int(os.environ.get("max_image_num",1))},

        )
        self.processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )
    

    def process_messages(self,messages):
        conv = internvl_conv.copy()
        conv.messages = []
        imgs = []
        
        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    conv.append_message(conv.roles[0],content)
                else:
                    conv.append_message(conv.roles[1],content)

        else:
            if "system" in messages:
                conv.system_message = messages["system"]
            
            
            if "image" in messages:
                text = messages["prompt"]
                inp = "<image>" + '\n' + text
                conv.append_message(conv.roles[0],inp)
                image = messages["image"]
                if isinstance(image,str):
                    if os.path.exists(image):
                        image = Image.open(image)
                elif isinstance(image,Image.Image):
                    image = image.convert("RGB")
                imgs.append(image)
            elif "images" in messages:
                text = messages["prompt"]
                images = messages["images"]
                inp = ""
                for i,image in enumerate(images):
                    inp = inp + f"<image_{i+1}>: " +"<image>" + '\n'
                    if isinstance(image,str):
                        if os.path.exists(image):
                            image = Image.open(image)
                    elif isinstance(image,Image.Image):
                        image = image.convert("RGB")
                    imgs.append(image)
                inp = inp + text
                conv.append_message(conv.roles[0],inp)
            else:
                text = messages["prompt"]
                inp = text
                conv.append_message(conv.roles[0],inp)
        
        conv.append_message(conv.roles[1],None) 
        prompt = conv.get_prompt()

        mm_data = {}
        if len(imgs) > 0:
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
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res

