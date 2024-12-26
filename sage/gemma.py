import torch
from torch import nn
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
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        