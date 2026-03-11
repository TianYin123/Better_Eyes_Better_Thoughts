from transformers import AutoProcessor,AutoTokenizer
from vllm import LLM, SamplingParams
import os


class Vllm_Text:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=int(os.environ.get("tensor_parallel_size", 1)),
            enforce_eager=True,
            trust_remote_code=True
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
        current_messages = []
        prompt = messages["prompt"]
        if "system" in messages:
            system_prompt = messages["system"]
            prompt = system_prompt + "\n" + prompt
        current_messages.append({"role":"user","content":prompt})   
        messages = current_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = False
        )
        return prompt


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
