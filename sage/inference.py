import torch
from model import *
from sys import argv
from tqdm import tqdm

assert len(argv) >= 3, "prompt and image path required"

pipe = Pipeline("cuda" if torch.cuda.is_available() else "cpu")
stop_token = pipe.model.config.text_config.eos_token_id
input_emb = pipe.first_emb(argv[1], argv[2])

generated_text = ""
with torch.inference_mode():
    for _ in tqdm(range(100)):
        predicted_token_index = pipe.generate(input_emb)
        predicted_token_index_val = predicted_token_index.item()
        if predicted_token_index_val == stop_token: break
        input_emb = torch.cat([input_emb, pipe.get_emb(predicted_token_index)], dim=1)
        new_token = pipe.tokenizer.decode(predicted_token_index_val)
        generated_text += new_token
print(generated_text)