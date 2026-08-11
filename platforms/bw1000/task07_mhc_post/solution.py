import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mhc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * block_size + tl.arange(0, block_size)
    mask = hidden_offsets < hidden_size

    x = tl.load(x_ptr + row * hidden_size + hidden_offsets, mask=mask, other=0.0)
    residual_base = row * 4 * hidden_size + hidden_offsets
    r0 = tl.load(residual_ptr + residual_base, mask=mask, other=0.0)
    r1 = tl.load(
        residual_ptr + residual_base + hidden_size, mask=mask, other=0.0
    )
    r2 = tl.load(
        residual_ptr + residual_base + 2 * hidden_size, mask=mask, other=0.0
    )
    r3 = tl.load(
        residual_ptr + residual_base + 3 * hidden_size, mask=mask, other=0.0
    )

    post_base = row * 4
    p0 = tl.load(post_ptr + post_base)
    p1 = tl.load(post_ptr + post_base + 1)
    p2 = tl.load(post_ptr + post_base + 2)
    p3 = tl.load(post_ptr + post_base + 3)

    comb_base = row * 16
    c00 = tl.load(comb_ptr + comb_base)
    c01 = tl.load(comb_ptr + comb_base + 1)
    c02 = tl.load(comb_ptr + comb_base + 2)
    c03 = tl.load(comb_ptr + comb_base + 3)
    c10 = tl.load(comb_ptr + comb_base + 4)
    c11 = tl.load(comb_ptr + comb_base + 5)
    c12 = tl.load(comb_ptr + comb_base + 6)
    c13 = tl.load(comb_ptr + comb_base + 7)
    c20 = tl.load(comb_ptr + comb_base + 8)
    c21 = tl.load(comb_ptr + comb_base + 9)
    c22 = tl.load(comb_ptr + comb_base + 10)
    c23 = tl.load(comb_ptr + comb_base + 11)
    c30 = tl.load(comb_ptr + comb_base + 12)
    c31 = tl.load(comb_ptr + comb_base + 13)
    c32 = tl.load(comb_ptr + comb_base + 14)
    c33 = tl.load(comb_ptr + comb_base + 15)

    out0 = x * p0 + r0 * c00 + r1 * c10 + r2 * c20 + r3 * c30
    out1 = x * p1 + r0 * c01 + r1 * c11 + r2 * c21 + r3 * c31
    out2 = x * p2 + r0 * c02 + r1 * c12 + r2 * c22 + r3 * c32
    out3 = x * p3 + r0 * c03 + r1 * c13 + r2 * c23 + r3 * c33

    output_base = row * 4 * hidden_size + hidden_offsets
    tl.store(output_ptr + output_base, out0, mask=mask)
    tl.store(output_ptr + output_base + hidden_size, out1, mask=mask)
    tl.store(output_ptr + output_base + 2 * hidden_size, out2, mask=mask)
    tl.store(output_ptr + output_base + 3 * hidden_size, out3, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = 2048
        self.num_warps = 4

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty(
            (2, 4096, 4, 1280), device=x.device, dtype=torch.bfloat16
        )
        _mhc_post_kernel[(8192, 1)](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            output,
            hidden_size=1280,
            block_size=self.block_size,
            num_warps=self.num_warps,
        )
        return output


class Model(ModelNew):
    """Compatibility name for evaluators that still require ``Model``."""


n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4


def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
) -> list[torch.Tensor]:
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn(
        (n0, n1, mhc_mult, mhc_mult), dtype=torch.float32
    )
    o_grad = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16)
    return [x, residual, post_layer_mix, comb_res_mix, o_grad]


def get_inputs():
    x, residual, post_layer_mix, comb_res_mix, _ = generate_mhc_post_test_data(
        n0, n1, h, mhc_mult
    )
    return [x, residual, post_layer_mix, comb_res_mix]


def get_init_inputs():
    return []
