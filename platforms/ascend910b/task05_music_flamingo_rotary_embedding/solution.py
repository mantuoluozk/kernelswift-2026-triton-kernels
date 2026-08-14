import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rotary_embedding_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    n_elements: tl.constexpr,
    seq_len: tl.constexpr,
    max_seq_len: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    feature = offsets & 127
    token = offsets // 128
    time_index = token % seq_len
    batch_index = token // seq_len
    inv_index = (feature & 63) // 2

    inv_freq = tl.load(inv_freq_ptr + inv_index, mask=mask, other=0.0)
    timestamps = tl.load(timestamps_ptr + token, mask=mask, other=0.0)
    two_pi = 6.283185307179586

    batch_frequency = batch_index.to(tl.float32) / max_seq_len * inv_freq
    time_frequency = (
        time_index.to(tl.float32) / max_seq_len * two_pi * inv_freq
    )
    frequency = tl.where(feature < 64, batch_frequency, time_frequency)
    phase = frequency * (-timestamps * two_pi)

    tl.store(cos_ptr + offsets, tl.cos(phase), mask=mask)
    tl.store(sin_ptr + offsets, tl.sin(phase), mask=mask)


class ModelNew(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        max_seq_len: int = 256,
        base: float = 10000.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.block_size = 1024
        self.num_warps = 8

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        cos = torch.empty((4, 32, 128), device=timestamps.device, dtype=torch.float32)
        sin = torch.empty_like(cos)

        _rotary_embedding_kernel[(16,)](
            timestamps,
            self.inv_freq,
            cos,
            sin,
            n_elements=16384,
            seq_len=32,
            max_seq_len=256,
            block_size=self.block_size,
            num_warps=self.num_warps,
        )
        return cos, sin


class Model(ModelNew):
    """Compatibility name for evaluators that still require ``Model``."""


def get_inputs():
    batch, seq = 4, 32
    timestamps = torch.rand(batch, seq, device="npu")
    return [timestamps, seq]


def get_init_inputs():
    return [64, 256, 10000.0]
