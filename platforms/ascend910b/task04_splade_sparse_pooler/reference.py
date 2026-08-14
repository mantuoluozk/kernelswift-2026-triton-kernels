import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """SPLADE 稀疏池化参考实现。"""

    def __init__(self, hidden_size: int = 768, vocab_size: int = 30522, pooling: str = "max"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        x = self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))
        x = torch.log1p(F.relu(x))
        result = []
        offset = 0
        for length in seq_lens.tolist():
            chunk = x[offset : offset + length]
            if self.pooling == "max":
                result.append(chunk.max(dim=0).values)
            else:
                result.append(chunk.sum(dim=0))
            offset += length
        return result


def get_inputs():
    seq_lens = torch.tensor([20, 25, 18, 20], dtype=torch.int32, device="npu")
    hidden_states = torch.randn(83, 768, device="npu")
    return [hidden_states, seq_lens]


def get_init_inputs():
    return [768, 30522, "max"]
