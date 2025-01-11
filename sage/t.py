from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "Describe the image"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

raw_image = Image.open("kanye.jpg")
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
for i in inputs:
    print(i, inputs[i].shape)