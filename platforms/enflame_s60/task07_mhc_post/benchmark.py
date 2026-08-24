import argparse
import statistics
import time

import torch
import torch_gcu  # noqa: F401 - registers GCU compatibility hooks

from reference import Model, get_init_inputs, get_inputs
from solution import ModelNew


def move_to_device(values, device):
    return tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in values)


def measure(model, inputs, warmup, repeat):
    for _ in range(warmup):
        model(*inputs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        model(*inputs)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--num-warps", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("accelerator is unavailable")

    device = torch.device("cuda:0")
    reference = Model(*get_init_inputs()).to(device).eval()
    optimized = ModelNew(*get_init_inputs()).to(device).eval()
    optimized.block_size = args.block_size
    optimized.num_warps = args.num_warps
    inputs = move_to_device(get_inputs(), device)

    with torch.no_grad():
        expected = reference(*inputs)
        actual = optimized(*inputs)
    max_abs_diff = (expected.float() - actual.float()).abs().max().item()
    print(f"max_abs_diff={max_abs_diff:.8e}")
    torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)

    with torch.no_grad():
        reference_ms = measure(reference, inputs, args.warmup, args.repeat)
        optimized_ms = measure(optimized, inputs, args.warmup, args.repeat)

    print("PASS accuracy")
    print(f"reference_median_ms={reference_ms:.6f}")
    print(f"optimized_median_ms={optimized_ms:.6f}")
    print(f"speedup={reference_ms / optimized_ms:.3f}x")


if __name__ == "__main__":
    main()
