import torch
import torch_npu  # noqa: F401 - registers the torch.npu backend
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    n_elements = 4096
    x = torch.randn(n_elements, device="npu", dtype=torch.float32)
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    _add_kernel[(triton.cdiv(n_elements, 256),)](
        x, y, out, n_elements, BLOCK=256
    )
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), (x + y).cpu())
    print("PASS Triton-Ascend vector add")


if __name__ == "__main__":
    main()
