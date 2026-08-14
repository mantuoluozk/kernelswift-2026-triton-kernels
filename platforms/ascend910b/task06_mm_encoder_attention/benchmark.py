import argparse
import statistics
import time

import torch
import torch_npu

from reference import Model as ReferenceModel, get_init_inputs, get_inputs
from solution import ModelNew


def elapsed_ms(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warps", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=300)
    args = parser.parse_args()

    inputs = get_inputs()
    reference = ReferenceModel(*get_init_inputs()).npu()
    optimized = ModelNew(*get_init_inputs()).npu()
    optimized.num_warps = args.num_warps
    expected = reference(*inputs)
    actual = optimized(*inputs)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    ref_ms = elapsed_ms(lambda: reference(*inputs), args.warmup, args.repeat)
    opt_ms = elapsed_ms(lambda: optimized(*inputs), args.warmup, args.repeat)
    print(f"correctness: PASS (num_warps={args.num_warps})")
    print(f"reference: {ref_ms:.6f} ms")
    print(f"optimized: {opt_ms:.6f} ms")
    print(f"speedup:   {ref_ms / opt_ms:.3f}x")


if __name__ == "__main__":
    main()
