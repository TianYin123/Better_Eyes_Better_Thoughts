from huggingface_hub import hf_hub_download
import torch
import os
from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from einops import repeat
from PIL import Image
import sys
import torch
from abc import ABC, abstractmethod


class AbstractProcessor(ABC):
    """
    Abstract class for processors to show what methods they need to implement.
    Processors handle text encoding and image preprocessing.
    """
    @abstractmethod
    def encode_text(self, prompt):
        pass

    @abstractmethod
    def preprocess_images(self, images: list):
        pass


class FlamingoProcessor(AbstractProcessor):
    """
    Processor class for Flamingo.
    """
    def __init__(self, tokenizer, vision_processor):
        """
        OF does not use same vision processor, image_processor only transforms single image
        """
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
    
    def encode_text(self, prompt):
        self.tokenizer.padding_side = "left" 
        # For generation padding tokens should be on the left
        return self.tokenizer([prompt],
            return_tensors="pt",
        )
    
    def preprocess_images(self, images: list):
        vision_x = [self.vision_processor(im).unsqueeze(0) for im in images]
        vision_x = torch.cat(vision_x, dim=0)
        return vision_x

def clean_generation(response):
    """
    for some reason, the open-flamingo based model slightly changes the input prompt (e.g. prepends <unk>, an adds some spaces)
    """
    return response.replace('<unk> ', '').strip()


#med_flamingo response is not stopping, we have to cap its response from the first next line split.
def get_before_first_next_line(tensor):
    indices_of_13 = (tensor == 13).nonzero(as_tuple=True)[0]
    if len(indices_of_13) > 0:
        first_13_index = indices_of_13[0]
        result = tensor[:first_13_index]
    else:
        result = tensor
    return result

class Med_Flamingo:
    def __init__(self,model_path,args):

        accelerator = Accelerator()
        self.device = accelerator.device

        # it is first creating a model architecture, then overwrite weights with Med-Flamingo weights
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="local-dir:/path/to/vision_encoder",
            clip_vision_encoder_pretrained=None,
            lang_encoder_path="/path/to/language_model",
            tokenizer_path="/path/to/language_model",
            cross_attn_every_n_layers=4
        )

        # self.checkpoint_path = hf_hub_download(model_path, "model.pt")
        # print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
        self.checkpoint_path = os.path.join(model_path,"model.pt")

        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device), strict=False)
        self.processor = FlamingoProcessor(self.tokenizer, self.image_processor)

        # self.temperature = args.temperature
        # self.top_p = args.top_p
        # self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = 100 # longer is useless for this model.

        # Prepare the model for evaluation
        self.model = accelerator.prepare(self.model)
        self.model.eval()

    def process_messages(self, messages):

        prompt = messages["prompt"]

        if "image" in messages:
            # images = [Image.open(messages["image"])]
            if isinstance(messages["image"],list):
                images = messages["image"]
            else:
                images = [messages["image"]]

            pixels = self.processor.preprocess_images(images)
            pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
            if "<image>" not in prompt:
                prompt = "<image>"*len(images) + prompt

        elif "images" in messages:
            # images = [Image.open(image) for image in messages["images"]]
            if isinstance(messages["images"],list):
                images = messages["images"]
            else:
                images = [messages["images"]]

            pixels = self.processor.preprocess_images(images)
            pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
            if "<image>" not in prompt:
                prompt = "<image>"*len(images) + prompt

        else:
            # this is a workaround for text-only benchmarks tested working
            pixels = torch.empty(1, 1, 1, 3, 336, 336)
        
        prompt = f"Question: {prompt} Answer:"
        print(prompt)
        tokenized_data = self.processor.encode_text(prompt)
        
        return pixels, tokenized_data

    def process_batch_messages(self, messages_list):

        prompts = [messages["prompt"] for messages in messages_list]

        # assumes the whole batch assumes the same format which otherwise the model doesn't supppor anyway
        if "image" in messages_list[0]:
            # images = [Image.open(messages["image"])]
            if isinstance(messages_list[0]["image"],list):
                batch_images = [messages["image"] for messages in messages_list]
            else:
                batch_images = [[messages["image"]] for messages in messages_list]

            batch_pixels = []
            batch_of_prompts = []
            for images_for_one_example in batch_images:
                # images_for_one_example = [Image.open(path) for path in image_path]
                processed_images = self.processor.preprocess_images(images_for_one_example)
                batch_pixels.append(processed_images)

            pixels = torch.stack(batch_pixels, dim=0) # Now 'b' will be the batch size
            pixels = repeat(pixels, 'b N c h w -> b N T c h w', T=1) # If T is always 1, no change needed

            for index, prompt in enumerate(prompts):
                if "<image>" not in prompt:
                    prompt = "<image>"*len(batch_images[index]) + prompt
                batch_of_prompts.append(prompt)
            

        elif "images" in messages_list[0]:
            # images = [Image.open(messages["image"])]
            if isinstance(messages_list[0]["images"],list):
                batch_images = [messages["images"] for messages in messages_list]
            else:
                batch_images = [[messages["images"]] for messages in messages_list]

            batch_pixels = []
            batch_of_prompts = []
            for images_for_one_example in batch_images:
                # images_for_one_example = [Image.open(path) for path in image_path]
                processed_images = self.processor.preprocess_images(images_for_one_example)
                batch_pixels.append(processed_images)

            # Stack them to create a batch dimension
            # This might require padding if the number of images per example varies
            # For now, assuming fixed number of images per example (e.g., 7 as in your example)
            pixels = torch.stack(batch_pixels, dim=0) # Now 'b' will be the batch size
            pixels = repeat(pixels, 'b N c h w -> b N T c h w', T=1) # If T is always 1, no change needed

            for index, prompt in enumerate(prompts):
                if "<image>" not in prompt:
                    prompt = "<image>"*len(batch_images[index]) + prompt
                batch_of_prompts.append(prompt)

        else:
            # this is a workaround for text-only benchmarks tested working
            pixels = torch.empty(len(prompts), 1, 1, 3, 336, 336)
            batch_of_prompts = prompts
        
        prompts = [f"Question: {prompt} Answer:" for prompt in prompts]
        # print(len(prompts))
        print(prompts)
        print(len(batch_of_prompts))
        print(pixels.shape)
        tokenized_data = self.processor.tokenizer(
            batch_of_prompts,
            padding='longest',  # Pad sequences to the length of the longest in the batch
            return_tensors='pt', # Return PyTorch tensors
            truncation=True     # Truncate if prompts are too long (optional, but often good practice)
        )

        return pixels, tokenized_data

    def generate_output(self, messages):
        pixels, tokenized_data = self.process_messages(messages)
        
        generated_text = self.model.generate(
            vision_x=pixels.to(self.device),
            lang_x=tokenized_data["input_ids"].to(self.device),
            attention_mask=tokenized_data["attention_mask"].to(self.device),
            max_new_tokens=self.max_new_tokens,
        )

        # it doesn't stop.
        response = self.processor.tokenizer.decode(get_before_first_next_line(generated_text[0][tokenized_data["input_ids"].shape[1]:]))
        response = clean_generation(response)
        print(response)
        print("")
        return response

    def generate_outputs(self, messages_list):
        results = []
        pixels, tokenized_data = self.process_batch_messages(messages_list)
        generated_text = self.model.generate(
            vision_x=pixels.to(self.device),
            lang_x=tokenized_data["input_ids"].to(self.device),
            attention_mask=tokenized_data["attention_mask"].to(self.device),
            max_new_tokens=self.max_new_tokens,
        )
        # results = [self.processor.tokenizer.decode(get_before_first_next_line(text[tokenized_data["input_ids"].shape[1]:])) for text in generated_text]
        results = [self.processor.tokenizer.decode(text[tokenized_data["input_ids"].shape[1]:]) for text in generated_text]
        print(results)
        return results
        
    # def generate_outputs(self, messages_list):
    #     results = []
    #     for messages in messages_list:
    #         result = self.generate_output(messages)
    #         results.append(result)
    #     return results
