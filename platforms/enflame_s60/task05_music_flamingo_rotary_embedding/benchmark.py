import argparse
import statistics
import time

import torch
import torch_gcu  # noqa: F401 - registers GCU compatibility hooks

from reference import Model, get_init_inputs, get_inputs
from solution import ModelNew


def measure(model, inputs, warmup, repeat):
    for _ in range(warmup):
        model(*inputs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        model(*inputs)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--num-warps", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("accelerator is unavailable")

    device = torch.device("cuda:0")
    reference = Model(*get_init_inputs()).to(device).eval()
    optimized = ModelNew(*get_init_inputs()).to(device).eval()
    optimized.load_state_dict(reference.state_dict())
    optimized.block_size = args.block_size
    optimized.num_warps = args.num_warps
    inputs = get_inputs()

    with torch.no_grad():
        expected = reference(*inputs)
        actual = optimized(*inputs)

    for index, (lhs, rhs) in enumerate(zip(expected, actual)):
        max_abs_diff = (lhs.float() - rhs.float()).abs().max().item()
        print(f"output[{index}] max_abs_diff={max_abs_diff:.8e}")
        torch.testing.assert_close(rhs, lhs, atol=args.atol, rtol=args.rtol)

    with torch.no_grad():
        reference_us = measure(reference, inputs, args.warmup, args.repeat)
        optimized_us = measure(optimized, inputs, args.warmup, args.repeat)

    print("PASS accuracy")
    print(f"reference_median_us={reference_us:.3f}")
    print(f"optimized_median_us={optimized_us:.3f}")
    print(f"speedup={reference_us / optimized_us:.3f}x")


if __name__ == "__main__":
    main()
