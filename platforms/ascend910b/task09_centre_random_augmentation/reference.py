import math
from typing import Optional

import torch
import torch.nn as nn


def random_rotation_matrices(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    u1 = torch.rand(n, device=device, dtype=dtype)
    u2 = torch.rand(n, device=device, dtype=dtype)
    u3 = torch.rand(n, device=device, dtype=dtype)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(n, 3, 3)


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    n_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    device = x_input_coords.device
    dtype = x_input_coords.dtype
    if mask is None:
        center = x_input_coords.mean(dim=-2, keepdim=True)
    else:
        m = mask.to(dtype=dtype).unsqueeze(-1)
        center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + eps)
    x = x_input_coords - center
    x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()
    if centre_only:
        return x

    rotation = random_rotation_matrices(n_sample, device=device, dtype=dtype)
    translation = s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)
    x = rot_vec_mul(rotation[:, None, :, :].expand(-1, x.shape[1], -1, -1), x) + translation[:, None, :]
    if mask is not None:
        x = x * mask.to(dtype=dtype)[None, :, None]
    return x


class Model(nn.Module):
    def __init__(self, n_sample: int = 1, s_trans: float = 1.0, centre_only: bool = False):
        super().__init__()
        self.n_sample = n_sample
        self.s_trans = s_trans
        self.centre_only = centre_only

    def forward(self, x_input_coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return centre_random_augmentation(
            x_input_coords=x_input_coords,
            n_sample=self.n_sample,
            s_trans=self.s_trans,
            centre_only=self.centre_only,
            mask=mask,
        )


def get_inputs():
    torch.manual_seed(42)
    x_input_coords = torch.randn(256, 3, device="npu")
    mask = torch.ones(256, device="npu", dtype=torch.float32)
    return [x_input_coords, mask]


def get_init_inputs():
    return [4, 1.0, False]
