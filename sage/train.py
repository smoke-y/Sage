import torch
from model import *
from datasets import load_dataset

ds = load_dataset("flaviagiammarino/vqa-rad")
pipe = Pipeline("cuda" if torch.cuda.is_available() else "cpu", train=True)
stop_token = pipe.model.config.text_config.eos_token_id
train = ds["train"]
for i in train: print(i)