import torch
from torch import nn
from siglip import *
from dataclasses import dataclass

@dataclass
class GemmaConfig:
    vocab_size: int
    max_position_embeddings: int = 8192
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int = 256
    num_key_value_head: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: bool = False

@dataclass
class PaliGemmaConfig:
    projection_dim: int = 2048
    hidden_size: int = 2048
    image_token_index: int
    pad_token_id: int
    vision_config: SiglipConfig
    text_config: GemmaConfig

def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class KVCache:
    def __init__(self) -> None:
        self.key_cache = []
        self.val_cache = []
    def num_items(self) -> int:
        if len(self.key_cache) == 0: return 0
        return self.key_cache[0].shape[-2]  #[batchSize, numHead, seqLen, headDim]
    def update(self, key: torch.Tensor, val: torch.Tensor, layerId: int):
        if len(self.key_cache) < layerId:
            self.key_cache.append(key)
            self.val_cache.append(val)
        else:
            # [batchSize, numHeads, seqLen, headDim]
            self.key_cache[layerId] = torch.cat([self.key_cache[layerId], key], dim=-2)
            self.val_cache[layerId] = torch.cat([self.val_cache[layerId], val], dim=-2)
        return self.key_cache[layerId], self.val_cache[layerId]

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_float = x_float * torch.sqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
        return (x_float * (1.0 + self.weight.float())).type_as(x)
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, device, config: GemmaConfig) -> None:
        super().__init__()
        inv_freq = 1/(config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float).float() / config.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.inv_freq.to(device)
    @torch.no_grad
    def forward(self, positionIds) -> torch.Tensor:
        freq_expaned = self.inv_freq[None, :, None].expand(positionIds.shape[0], -1, 1)
        pos_expanded = positionIds[:, None, :]
        #[Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
        freqs = (freq_expaned @ pos_expanded).transpose(1,2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.gelu(self.gate_proj(x)) * self.up_proj(x))
class GemmaAttention(nn.Module):
    def __init__(self, device, layerId, config: GemmaConfig) -> None:
        super().__init__()
        self.layerId = layerId
        self.num_key_value_head = config.num_key_value_head
        self.num_key_value_grp = config.num_attention_heads // self.num_key_value_head
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * config.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_head * config.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_head * config.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(device, config)
    def forward(self, x: torch.Tensor, attention_mask = None, position_ids = None, kv_cache: KVCache = None):
        B,T,C = x.shape()           #[batch, numPatch, embDim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_key_value_grp, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_key_value_grp, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb.forward(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if kv_cache is not None: k, v = kv_cache.update(k, v, self.layerId)
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v, attn_mask=attention_mask, is_causal=False, enable_gqa=True)
        return self.o_proj(y)
class GemmaDecoderLayer(nn.Module):
    def __init__(self, device, layerId, config: GemmaConfig) -> None:
        super().__init__()
        self.self_attn = GemmaAttention(device, layerId, config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
    def forward(self, x: torch.Tensor, attention_mask, position_ids, kv_cache) -> torch.Tensor:
        y = self.input_layernorm(x)
        y = self.self_attn(x, attention_mask, position_ids, kv_cache)
        x += y
        y = self.post_attention_layernorm(y)
        y = self.mlp(y)
        return x+y
class GemmaModel(nn.Module):
    def __init__(self, device, config: GemmaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = self.hidden_size ** 0.5
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([GemmaDecoderLayer(device) for _ in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
    def forward(self, x: torch.Tensor, attention_mask, position_ids, kv_cache) -> torch.Tensor:
        x = x * self.norm
        for decode_layer in self.layers: x = decode_layer(x, attention_mask, position_ids, kv_cache)
        return self.norm(x)
class GemmaForCausalLM(nn.Module):
    def __init__(self, device, config: GemmaConfig) -> None:
        super().__init__()
        self.model = GemmaModel(device, config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def forward(self, x: torch.Tensor, attention_mask, position_ids, kv_cache) -> torch.Tensor:
        output = self.model.forward(x, attention_mask, position_ids, kv_cache)
        return self.lm_head(output)
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.linear(x)
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.scale = config.hidden_size ** 0.5
        self.image_token_id = config.image_token_index
        self.pad_token_id = config.pad_token_id
        self.vision_tower = SiglipVisionTransformer(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)
    def tie_weights(self) -> None: self.language_model.lm_head.weight = self.language_model.model.embed_tokens.weight
    def _merge_input_ids_with_image_feature(self, image_feature: torch.Tensor, input_id: torch.Tensor, input_emb: torch.Tensor, attention_mask: torch.Tensor, kv_cache = None):
        _, _, emb_dim = image_feature.shape
        batch_size, seq_len = input_id.shape
        image_feature = image_feature / self.scale
        final_emb = torch.zeros(batch_size, seq_len, emb_dim, dtype=input_emb.dtype, device=input_emb.device)
        text_mask = (input_id != self.image_token_id) & (input_id != self.pad_token_id)
        image_mask = input_id == self.image_token_id
        pad_mask = input_id == self.pad_token_id
        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, emb_dim)
        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, emb_dim)
        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, emb_dim)
        final_emb = torch.where(text_mask, input_emb, final_emb)
        final_emb = final_emb.masked_scatter(image_mask, image_feature)
        final_emb = torch.where(pad_mask, torch.zeros_like(final_emb), final_emb)

        q_len = input_emb.shape[1]  #(batchSize, seqLen, embDim)
        device = input_emb.device
        #causal mask has to be of dim (batchSize, numHeads, seqLenQ, seqLenKV)
        if kv_cache is None:
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=input_emb.dtype, device=device)
            causal_mask = causal_mask.unsqueeze(1)
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=input_emb.dtype, device=device)
            causal_mask = causal_mask.unsqueeze(1)
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1: position_ids = position_ids.unsqueeze(0)
        return final_emb, causal_mask, position_ids
    def forward(self, input_ids, pixel_values, attention_mask, kv_cache):
        input_emb = self.language_model.model.embed_tokens(input_ids)
        image_feat = self.vision_tower.forward(pixel_values.to(input_emb.dtype))
        image_feat = self.multi_modal_projector.forward(image_feat)
        input_emb, attention_mask, position_id = self._merge_input_ids_with_image_feature(image_feat, input_ids, input_emb, attention_mask, kv_cache)
        return self.language_model.forward(input_emb, attention_mask, position_id, kv_cache)