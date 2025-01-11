import torch
import torch.nn as nn
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
    def __init__(self, config: SageConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
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
        self.mod = None
    @staticmethod
    def from_pretrained():
        from transformers import AutoTokenizer, LlavaForConditionalGeneration

        modelId = "llava-hf/llava-interleave-qwen-0.5b-hf"
        tokenizer = AutoTokenizer.from_pretrained(modelId)
        pre_model = LlavaForConditionalGeneration.from_pretrained(modelId)
        config = SageConfig(pre_model.config.vision_config, pre_model.config.text_config)
        model = Sage(config)
        for name, param in pre_model.named_parameters():
            if name in model.state_dict(): model.state_dict()[name].copy_(param)
        return tokenizer, model
    def forward(self, input_ids, image):
        index_to_replace = (input_ids == self.config.image_token_index).nonzero(as_tuple=True)[0].item()
        input_emb = self.language_model.get_input_embeddings()(input_ids)
    ##
        image = self.vision_tower(image)
        image_emb = self.multi_modal_projector(image)
        input_emb = torch.cat([input_emb[:, :index_to_replace, :], image_emb, input_emb[:, index_to_replace+1: , :]], dim=1)
    ##
        attention_mask = torch.ones(input_emb.shape[:2], dtype=torch.int, device=input_emb.device)
        logits = self.language_model.forward(
            attention_mask=attention_mask,
            inputs_embeds=input_emb
        )
        return logits