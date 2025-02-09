import gc
import torch
from model import *
from datasets import load_dataset

ds = load_dataset("flaviagiammarino/vqa-rad")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = Pipeline(device, lora=True)
optim = pipe.get_optimizer()
stop_token = pipe.model.config.text_config.eos_token_id
train = ds["train"].shuffle()
x = 0
log = open("trace.log", "w+")
for epoch in range(10):
    for i in train:
        gc.collect()
        torch.cuda.empty_cache()
        optim.zero_grad()
        input_emb = pipe.first_emb(i["question"], i["image"])
        lenQues = input_emb.shape[1]
        labels = torch.tensor(pipe.tokenizer.encode(i["answer"]) + [stop_token]).unsqueeze(0).to(device)
        input_emb = torch.cat([input_emb, pipe.get_emb()(labels)], dim=1)
        logits = pipe.get_logits(input_emb)[:, lenQues: ,:]
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        log.write(f"{loss.detach().cpu().numpy()}\n")
        loss.backward()
        optim.step()
        if x == 7:
            x = 0
            pipe.model.saveLoRaWeights("lora.pth")
        x += 1
        del labels, input_emb
pipe.model.saveLoRaWeights("lora.pth")
log.close()