import os

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


class Qwen3_VL:
    def __init__(self, model_path, args):
        super().__init__()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=int(os.environ.get("tensor_parallel_size", 8)),
            enforce_eager=True,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 50},
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[],
        )

    def process_messages(self, messages):
        current_messages = []

        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                current_messages.append(
                    {
                        "role": message["role"],
                        "content": [{"type": "text", "text": message["content"]}],
                    }
                )
        else:
            if "system" in messages:
                current_messages.append({"role": "system", "content": messages["system"]})

            if "image" in messages:
                current_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": messages["image"]},
                            {"type": "text", "text": messages["prompt"]},
                        ],
                    }
                )
            elif "images" in messages:
                content = []
                for i, image in enumerate(messages["images"]):
                    content.append({"type": "text", "text": f"<image_{i+1}>: "})
                    content.append({"type": "image", "image": image})
                content.append({"type": "text", "text": messages["prompt"]})
                current_messages.append({"role": "user", "content": content})
            else:
                current_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": messages["prompt"]}],
                    }
                )

        prompt = self.processor.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(current_messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {"prompt": prompt}
        if mm_data:
            llm_inputs["multi_modal_data"] = mm_data
        return llm_inputs

    def generate_output(self, messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_outputs(self, messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        return [output.outputs[0].text for output in outputs]
