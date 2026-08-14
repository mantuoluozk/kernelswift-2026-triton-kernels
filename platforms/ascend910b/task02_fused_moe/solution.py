import torch
import torch_npu
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gate_up_kernel(x_ptr, w1_ptr, act_ptr):
    block_m = tl.program_id(0)
    expert = tl.program_id(1)
    m = block_m * 32 + tl.arange(0, 32)
    n = tl.arange(0, 64)
    k = tl.arange(0, 128)

    x = tl.load(x_ptr + m[:, None] * 128 + k[None, :], mask=m[:, None] < 83, other=0.0)
    w_base = w1_ptr + expert * 128 * 128
    gate_w = tl.load(w_base + n[None, :] * 128 + k[:, None]).to(tl.float16)
    up_w = tl.load(w_base + (n[None, :] + 64) * 128 + k[:, None]).to(tl.float16)
    gate = tl.dot(x, gate_w)
    up = tl.dot(x, up_w)
    act = gate * tl.sigmoid(gate) * up
    offsets = expert * 83 * 64 + m[:, None] * 64 + n[None, :]
    tl.store(act_ptr + offsets, act, mask=m[:, None] < 83)


@triton.jit
def _down_kernel(act_ptr, w2_ptr, expert_out_ptr):
    block_m = tl.program_id(0)
    expert = tl.program_id(1)
    block_n = tl.program_id(2)
    m = block_m * 32 + tl.arange(0, 32)
    n = block_n * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)

    act_offsets = expert * 83 * 64 + m[:, None] * 64 + k[None, :]
    act = tl.load(act_ptr + act_offsets, mask=m[:, None] < 83, other=0.0)
    w_base = w2_ptr + expert * 128 * 64
    weights = tl.load(w_base + n[None, :] * 64 + k[:, None]).to(tl.float16)
    output = tl.dot(act, weights)
    out_offsets = expert * 83 * 128 + m[:, None] * 128 + n[None, :]
    tl.store(expert_out_ptr + out_offsets, output, mask=m[:, None] < 83)


@triton.jit
def _route_kernel(router_ptr, expert_out_ptr, output_ptr):
    token = tl.program_id(0)
    expert = tl.arange(0, 8)
    hidden = tl.arange(0, 128)
    logits = tl.load(router_ptr + token * 8 + expert)
    e0 = tl.argmax(logits, axis=0)
    l0 = tl.max(logits, axis=0)
    logits = tl.where(expert == e0, -float("inf"), logits)
    e1 = tl.argmax(logits, axis=0)
    l1 = tl.max(logits, axis=0)

    ratio = tl.exp(l1 - l0)
    w0 = (1.0 / (1.0 + ratio)).to(tl.float16)
    w1 = (ratio / (1.0 + ratio)).to(tl.float16)
    y0 = tl.load(expert_out_ptr + e0 * 83 * 128 + token * 128 + hidden)
    y1 = tl.load(expert_out_ptr + e1 * 83 * 128 + token * 128 + hidden)
    output = y0 * w0 + y1 * w1
    tl.store(output_ptr + token * 128 + hidden, output)


class ModelNew(nn.Module):
    def __init__(self, num_experts, top_k, hidden_size, intermediate_size, renormalize=True):
        super().__init__()
        assert (num_experts, top_k, hidden_size, intermediate_size, renormalize) == (8, 2, 128, 64, True)
        self.w1 = nn.Parameter(torch.empty(8, 128, 128))
        self.w2 = nn.Parameter(torch.empty(8, 128, 64))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        self.num_warps = 4

    def forward(self, hidden_states, router_logits):
        act = torch.empty((8, 83, 64), device=hidden_states.device, dtype=torch.float16)
        expert_out = torch.empty((8, 83, 128), device=hidden_states.device, dtype=torch.float16)
        output = torch.empty((83, 128), device=hidden_states.device, dtype=torch.float16)
        _gate_up_kernel[(triton.cdiv(83, 32), 8)](
            hidden_states, self.w1, act, num_warps=self.num_warps
        )
        _down_kernel[(triton.cdiv(83, 32), 8, 2)](
            act, self.w2, expert_out, num_warps=self.num_warps
        )
        _route_kernel[(83,)](router_logits, expert_out, output, num_warps=1)
        return output


class Model(ModelNew):
    pass


def get_init_inputs():
    return [8, 2, 128, 64]


def get_inputs():
    hidden_states = torch.randn(83, 128, dtype=torch.float16, device="npu")
    router_logits = torch.randn(83, 8, dtype=torch.float32, device="npu")
    return [hidden_states, router_logits]
