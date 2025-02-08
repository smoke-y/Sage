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
        optim.zero_grad()
        input_emb = pipe.first_emb(i["question"], i["image"])
        lenQues = input_emb.shape[1]
        ans_id = torch.tensor(pipe.tokenizer.encode(i["answer"]) + [stop_token]).unsqueeze(0).to(device)
        for j in range(ans_id.shape[1]):
            predicted_token_index, loss = pipe.generate(input_emb, ans_id[:, j])
            input_emb = torch.cat([input_emb, pipe.get_emb(ans_id[:, j]).unsqueeze(0)], dim=1)
            log.write(f"{loss.detach().cpu().numpy()}\n")
            loss.backward()
            optim.step()
        if x == 7:
            x = 0
            pipe.model.saveLoRaWeights("lora.pth")
        x += 1
pipe.model.saveLoRaWeights("lora.pth")
log.close()