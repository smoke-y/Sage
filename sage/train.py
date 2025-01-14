import torch
from model import *
from datasets import load_dataset

ds = load_dataset("flaviagiammarino/vqa-rad")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = Pipeline(device, train=True)
stop_token = pipe.model.config.text_config.eos_token_id
train = ds["train"].shuffle()
for i in train:
    input_emb = pipe.first_emb(i["question"], i["image"])
    lenQues = input_emb.shape[1]
    ans_id = torch.tensor(pipe.tokenizer.encode(i["answer"]) + [stop_token]).unsqueeze(0).to(device)
    for j in range(ans_id[:, 1]):
        predicted_token_index, loss = pipe.generate(input_emb, ans_id[:, j])
        input_emb = torch.cat([input_emb, pipe.get_emb(predicted_token_index)], dim=1)