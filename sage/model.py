import copy
import torch
import torch.nn as nn
import loralib as lora
from language_model.qwen import *
from vision_model.siglip import *
from transformers.utils import logging

logging.set_verbosity_error()

class SageConfig:
    def __init__(self, vision_config: SiglipVisionConfig, text_config: Qwen2Config) -> None:
        self.vision_config = vision_config
        self.text_config = text_config
        self.projector_hidden_act = "gelu"
        self.image_token_index = 151646

class SageMultiModalProjector(nn.Module):
    def __init__(self, config: SageConfig) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class Sage(nn.Module):
    def __init__(self, config: SageConfig):
        super().__init__()
        self.config = config
        self.language_model = Qwen2ForCausalLM(config.text_config)
        self.language_model.tie_weights()
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = SageMultiModalProjector(config)
    @staticmethod
    def from_pretrained(device: torch.device):
        from transformers import LlavaForConditionalGeneration
        pre_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")
        config = SageConfig(copy.deepcopy(pre_model.config.vision_config), copy.deepcopy(pre_model.config.text_config))
        model = Sage(config).to(device)
        for name, param in pre_model.named_parameters():
            if name in model.state_dict(): model.state_dict()[name].copy_(param)
        del pre_model
        return torch.compile(model)
    def forward(self, input_emb: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones(input_emb.shape[:2], dtype=torch.int, device=input_emb.device)
        logits = self.language_model.forward(
            attention_mask=attention_mask,
            inputs_embeds=input_emb
        )
        return logits
    
import numpy as np
from PIL import Image

class Pipeline:
    def __init__(self, device: torch.device, train: bool = False) -> None:
        print("Using", device)
        torch.set_float32_matmul_precision("high")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")
        self.model = Sage.from_pretrained(device)
        self.device = device
        if train: lora.mark_only_lora_as_trainable(self.model)
    def first_emb(self, text: str, img: Image) -> torch.Tensor:
        size = self.model.config.vision_config.image_size
        img = img.resize((size, size))
        img_array = np.array(img)
        img_tensor = torch.tensor(img_array).permute(2, 0, 1)  #[H, W, C] -> [C, H, W]
        img_tensor = (img_tensor.float() / 255.0).unsqueeze(0).to(self.device)

        prompt = f"<|im_start|>user\n<image> {text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        input_emb = self.model.language_model.get_input_embeddings()(input_ids)
        image = self.model.vision_tower(img_tensor)
        image_emb = self.model.multi_modal_projector(image)
        index_to_replace = (input_ids == self.model.config.image_token_index).nonzero(as_tuple=True)[0].item()
        return torch.cat([input_emb[:, :index_to_replace, :], image_emb, input_emb[:, index_to_replace+1: , :]], dim=1).to(self.device)
    def generate(self, input_emb: torch.Tensor, temp: float = 0.7, top_k: float = 50) -> torch.Tensor:
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            logits = self.model.forward(input_emb)
        top_k_logits, top_k_indices = torch.topk(logits[:, -1, :], top_k)
        probs = torch.softmax(top_k_logits / temp, dim=-1)
        pred = torch.multinomial(probs, num_samples=1)
        return top_k_indices.gather(-1, pred)
    def get_emb(self, index: torch.Tensor) -> torch.Tensor: return self.model.language_model.get_input_embeddings()(index)
    def save_lora(self, path: str) -> None: torch.save(lora.lora_state_dict(self.model), path)
    def load_lora(self, path: str) -> None: self.model.load_state_dict(torch.load(path), strict=False)