import argparse
import statistics
import time

import torch

import reference as reference_module
from solution import ModelNew


def reset_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def measure(model, inputs, warmup, repeat):
    for _ in range(warmup):
        reset_seed()
        model(*inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        reset_seed()
        start = time.perf_counter()
        model(*inputs)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--num-warps", type=int, default=1)
    args = parser.parse_args()

    reference = reference_module.Model(*reference_module.get_init_inputs()).cuda().eval()
    optimized = ModelNew(*reference_module.get_init_inputs()).cuda().eval()
    optimized.num_warps = args.num_warps
    inputs = reference_module.get_inputs()

    with torch.no_grad():
        reset_seed()
        expected = reference(*inputs)
        reset_seed()
        actual = optimized(*inputs)
    max_abs_diff = (expected.float() - actual.float()).abs().max().item()
    print(f"max_abs_diff={max_abs_diff:.8e}")
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    with torch.no_grad():
        reference_us = measure(reference, inputs, args.warmup, args.repeat)
        optimized_us = measure(optimized, inputs, args.warmup, args.repeat)
    print("PASS accuracy")
    print(f"reference_median_us={reference_us:.3f}")
    print(f"optimized_median_us={optimized_us:.3f}")
    print(f"speedup={reference_us / optimized_us:.3f}x")


if __name__ == "__main__":
    main()
