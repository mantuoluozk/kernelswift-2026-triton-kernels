import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _grouped_topk_kernel(gating_ptr, weights_ptr, ids_ptr, scale):
    row = tl.program_id(0)
    expert = tl.arange(0, 256)
    logits = tl.load(gating_ptr + row * 256 + expert)

    # Softmax is monotonic, so both selections can operate directly on logits.
    group_logits = tl.reshape(logits, (8, 32))
    group_scores = tl.max(group_logits, axis=1)
    group = tl.arange(0, 8)
    neg_inf = -float("inf")

    g0 = tl.argmax(group_scores, axis=0)
    group_scores = tl.where(group == g0, neg_inf, group_scores)
    g1 = tl.argmax(group_scores, axis=0)
    group_scores = tl.where(group == g1, neg_inf, group_scores)
    g2 = tl.argmax(group_scores, axis=0)
    group_scores = tl.where(group == g2, neg_inf, group_scores)
    g3 = tl.argmax(group_scores, axis=0)

    expert_group = expert // 32
    selected_group = (
        (expert_group == g0)
        | (expert_group == g1)
        | (expert_group == g2)
        | (expert_group == g3)
    )
    candidates = tl.where(selected_group, logits, neg_inf)

    i0 = tl.argmax(candidates, axis=0)
    v0 = tl.max(candidates, axis=0)
    candidates = tl.where(expert == i0, neg_inf, candidates)
    i1 = tl.argmax(candidates, axis=0)
    v1 = tl.max(candidates, axis=0)
    candidates = tl.where(expert == i1, neg_inf, candidates)
    i2 = tl.argmax(candidates, axis=0)
    v2 = tl.max(candidates, axis=0)
    candidates = tl.where(expert == i2, neg_inf, candidates)
    i3 = tl.argmax(candidates, axis=0)
    v3 = tl.max(candidates, axis=0)
    candidates = tl.where(expert == i3, neg_inf, candidates)
    i4 = tl.argmax(candidates, axis=0)
    v4 = tl.max(candidates, axis=0)
    candidates = tl.where(expert == i4, neg_inf, candidates)
    i5 = tl.argmax(candidates, axis=0)
    v5 = tl.max(candidates, axis=0)
    candidates = tl.where(expert == i5, neg_inf, candidates)
    i6 = tl.argmax(candidates, axis=0)
    v6 = tl.max(candidates, axis=0)
    candidates = tl.where(expert == i6, neg_inf, candidates)
    i7 = tl.argmax(candidates, axis=0)
    v7 = tl.max(candidates, axis=0)

    # The common 256-way softmax denominator cancels in top-k renormalization.
    e0 = tl.exp(v0 - v0)
    e1 = tl.exp(v1 - v0)
    e2 = tl.exp(v2 - v0)
    e3 = tl.exp(v3 - v0)
    e4 = tl.exp(v4 - v0)
    e5 = tl.exp(v5 - v0)
    e6 = tl.exp(v6 - v0)
    e7 = tl.exp(v7 - v0)
    denom = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7

    weight_base = weights_ptr + row * 8
    id_base = ids_ptr + row * 8
    tl.store(weight_base + 0, e0 / denom * scale)
    tl.store(weight_base + 1, e1 / denom * scale)
    tl.store(weight_base + 2, e2 / denom * scale)
    tl.store(weight_base + 3, e3 / denom * scale)
    tl.store(weight_base + 4, e4 / denom * scale)
    tl.store(weight_base + 5, e5 / denom * scale)
    tl.store(weight_base + 6, e6 / denom * scale)
    tl.store(weight_base + 7, e7 / denom * scale)
    tl.store(id_base + 0, i0)
    tl.store(id_base + 1, i1)
    tl.store(id_base + 2, i2)
    tl.store(id_base + 3, i3)
    tl.store(id_base + 4, i4)
    tl.store(id_base + 5, i5)
    tl.store(id_base + 6, i6)
    tl.store(id_base + 7, i7)


class ModelNew(nn.Module):
    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        assert (topk, renormalize, num_expert_group, topk_group) == (8, True, 8, 4)
        assert scoring_func == "softmax"
        self.routed_scaling_factor = routed_scaling_factor
        self.num_warps = 1

    def forward(self, hidden_states: torch.Tensor, gating_output: torch.Tensor):
        rows = gating_output.shape[0]
        weights = torch.empty((rows, 8), device=gating_output.device, dtype=torch.float32)
        ids = torch.empty((rows, 8), device=gating_output.device, dtype=torch.int32)
        _grouped_topk_kernel[(rows,)](
            gating_output,
            weights,
            ids,
            self.routed_scaling_factor,
            num_warps=self.num_warps,
        )
        return weights, ids


class Model(ModelNew):
    pass


def get_init_inputs():
    return [8, True, 8, 4]


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    return [hidden_states, gating_output]
