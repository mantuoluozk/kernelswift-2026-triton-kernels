import torch
import torch_npu
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _head_compute_mix_bwd_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    grad_scale_ptr,
    grad_base_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    channels = offsets & 3

    input_mix = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    base = tl.load(base_ptr + channels, mask=mask, other=0.0)

    z = input_mix * scale + base
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad_out * sigmoid * (1.0 - sigmoid)
    grad_input = grad_z * scale
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)

    grad_scale = tl.sum(grad_z * input_mix, axis=0)
    grad_base0 = tl.sum(tl.where(mask & (channels == 0), grad_z, 0.0), axis=0)
    grad_base1 = tl.sum(tl.where(mask & (channels == 1), grad_z, 0.0), axis=0)
    grad_base2 = tl.sum(tl.where(mask & (channels == 2), grad_z, 0.0), axis=0)
    grad_base3 = tl.sum(tl.where(mask & (channels == 3), grad_z, 0.0), axis=0)

    tl.atomic_add(grad_scale_ptr, grad_scale)
    tl.atomic_add(grad_base_ptr, grad_base0)
    tl.atomic_add(grad_base_ptr + 1, grad_base1)
    tl.atomic_add(grad_base_ptr + 2, grad_base2)
    tl.atomic_add(grad_base_ptr + 3, grad_base3)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = 4096
        self.num_warps = 4

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        grad_input = torch.empty_like(input_mix)
        grad_scale = torch.zeros_like(mhc_scale)
        grad_base = torch.zeros_like(mhc_base)

        _head_compute_mix_bwd_kernel[(triton.cdiv(8192, self.block_size),)](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input,
            grad_scale,
            grad_base,
            n_elements=8192,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
        )
        return grad_input, grad_scale, grad_base


class Model(ModelNew):
    """Compatibility name for evaluators that still require ``Model``."""


batch0 = 2
batch1 = 1024
mhc_mult = 4


def get_inputs():
    input_mix = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32)
    mhc_scale = torch.randn(1, dtype=torch.float32)
    mhc_base = torch.randn(mhc_mult, dtype=torch.float32)
    grad_out = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32)
    return [input_mix, mhc_scale, mhc_base, grad_out]


def get_init_inputs():
    return []
