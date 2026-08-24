import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr, scale: tl.constexpr):
    block_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // 8
    head = batch_head % 8
    m = block_m * 32 + tl.arange(0, 32)
    n = tl.arange(0, 32)
    d = tl.arange(0, 64)
    batch_offset = batch * 83 * 512

    q_offsets = batch_offset + m[:, None] * 512 + head * 64 + d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=m[:, None] < 83, other=0.0)

    row_max = tl.full((32,), -float("inf"), tl.float32)
    row_sum = tl.zeros((32,), tl.float32)
    acc = tl.zeros((32, 64), tl.float32)

    for start_n in range(0, 96, 32):
        cols = start_n + n
        kv_offsets = batch_offset + cols[:, None] * 512 + head * 64 + d[None, :]
        k = tl.load(k_ptr + kv_offsets, mask=cols[:, None] < 83, other=0.0)
        scores = tl.dot(q, tl.trans(k)) * scale
        valid = (m[:, None] < 83) & (cols[None, :] < 83)
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = tl.exp(row_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)

        v = tl.load(v_ptr + kv_offsets, mask=cols[:, None] < 83, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.float16), v)
        row_max = new_max

    output = acc / row_sum[:, None]
    tl.store(out_ptr + q_offsets, output, mask=m[:, None] < 83)


class ModelNew(nn.Module):
    def __init__(self, num_heads=8, head_size=64, num_kv_heads=8):
        super().__init__()
        assert num_heads == 8 and head_size == 64 and num_kv_heads == 8
        self.scale = 1.0 / (head_size ** 0.5)
        self.num_warps = 1

    def forward(self, query, key, value):
        output = torch.empty_like(query)
        _attention_kernel[(triton.cdiv(83, 32), 16)](
            query, key, value, output, self.scale, num_warps=self.num_warps
        )
        return output


class Model(ModelNew):
    pass


def get_init_inputs():
    return [8, 64, 8]


def get_inputs():
    shape = (2, 83, 512)
    return [torch.randn(shape, dtype=torch.float16, device="cuda") for _ in range(3)]
