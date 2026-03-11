from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import os
from PIL import Image

class LlamaVision:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model= model_path,
            enforce_eager=True,
            max_num_seqs = 1,
            tensor_parallel_size= int(os.environ.get("tensor_parallel_size",1)),
            gpu_memory_utilization = 0.8,
            limit_mm_per_prompt = {"image": args.max_image_num},
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )

    def process_messages(self,messages):
        current_messages = []
        image_inputs = None
        prompt = messages["prompt"]
        if "system" in messages:
            system_prompt = messages["system"]
            current_messages.append({"role":"system","content":system_prompt})
        if "image" in messages:
            image = messages["image"]
            current_messages.append({"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]})
            if isinstance(image,str):
                image = Image.open(image)
            elif isinstance(image,Image.Image):
                image = image.convert("RGB")
            image_inputs = [image]
        elif "images" in messages:
            images = messages["images"]
            image_inputs = []
            content = []
            for i,image in enumerate(images):
                content.append({"type":"text","text":f"<image_{i+1}>: "})
                content.append({"type":"image","image":image})
                if isinstance(image,str):
                    if os.path.exists(image):
                        image = Image.open(image)
                elif isinstance(image,Image.Image):
                    image = image.convert("RGB")
                image_inputs.append(image)
            content.append({"type":"text","text":prompt})
            current_messages.append({"role":"user","content":content})
        else:
            current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]})  
        messages = current_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt
        }
        if mm_data:
            llm_inputs["multi_modal_data"] = mm_data
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
