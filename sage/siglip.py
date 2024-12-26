import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class SiglipConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layer: int = 12
    num_attention_head: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

class SiglipEmbedding(nn.Module):
    def __init__(self, config: SiglipConfig) -> None:
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = config.hidden_size,
            kernel_size = config.patch_size,
            stride = config.patch_size,
        )
        numPos = (config.image_size // config.patch_size) ** 2
        self.positional_embedding = nn.Embedding(numPos, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(numPos).expand((1, -1)),
            persistent=False
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)  #[batch, embDim, numPatchY, numPatchX]
        x = x.flatten(2)              #[batch, embDim, numPatch]
        x = x.transpose(1, 2)         #[batch, numPatch, embDim]
        return x + self.positional_embedding(x.position_ids)
class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipConfig) -> None:
        super().__init__()
        assert config.hidden_size % config.num_attention_head == 0
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.nheads = config.num_attention_head
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size()           #[batch, numPatch, embDim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        k = k.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        v = v.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(y)
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(nn.functional.gelu(self.fc1(x)))
class SiglipLayer(nn.Module):
    def __init__(self, config: SiglipConfig) -> None:
        super().__init__()
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.self_attn(x)
        hidden = self.layer_norm1(hidden)
        x += hidden
        hidden = self.mlp(x)
        hidden = self.layer_norm2(hidden)
        return x + hidden
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipLayer(config) for _ in range(config.num_hidden_layer)]
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers: x = layer(x)
        return x
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipConfig) -> None:
        super().__init__()
        self.embeddings = SiglipEmbedding(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_layernorm(self.encoder(self.embeddings(x)))