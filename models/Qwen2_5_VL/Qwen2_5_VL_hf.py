from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2_5_VLProcessor

class Qwen2_5_VL:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto",attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens


    def process_messages(self,messages):
        new_messages = []
        if "system" in messages:
            new_messages.append({"role":"system","content":messages["system"]}) 
        if "image" in messages:
            new_messages.append({"role":"user","content":[{"type":"image","image":messages["image"]},{"type":"text","text":messages["prompt"]}]})
        elif "images" in messages:
            content = []
            for i,image in enumerate(messages["images"]):
                content.append({"type":"text","text":f"<image_{i+1}>: "})
                content.append({"type":"image","image":image})
            content.append({"type":"text","text":messages["prompt"]})
            new_messages.append({"role":"user","content":content})
        else:
            new_messages.append({"role":"user","content":[{"type":"text","text":messages["prompt"]}]})
        messages = new_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
)
        inputs = inputs.to("cuda")

        return inputs


    def generate_output(self,messages):
        inputs = self.process_messages(messages)
        do_sample = False if self.temperature == 0 else True
        generated_ids = self.llm.generate(**inputs,temperature=self.temperature,top_p=self.top_p,repetition_penalty=self.repetition_penalty,max_new_tokens=self.max_new_tokens,do_sample = do_sample)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res

        