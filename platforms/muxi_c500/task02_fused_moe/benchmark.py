import argparse
import statistics
import time

import torch

from reference import Model as ReferenceModel, get_init_inputs, get_inputs
from solution import ModelNew


def elapsed_ms(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warps", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()
    torch.manual_seed(42)
    reference = ReferenceModel(*get_init_inputs()).cuda().eval()
    torch.manual_seed(42)
    optimized = ModelNew(*get_init_inputs()).cuda().eval()
    optimized.load_state_dict(reference.state_dict())
    optimized.num_warps = args.num_warps
    inputs = get_inputs()
    expected = reference(*inputs)
    actual = optimized(*inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    ref_ms = elapsed_ms(lambda: reference(*inputs), args.warmup, args.repeat)
    opt_ms = elapsed_ms(lambda: optimized(*inputs), args.warmup, args.repeat)
    print(f"correctness: PASS (num_warps={args.num_warps})")
    print(f"reference: {ref_ms:.6f} ms")
    print(f"optimized: {opt_ms:.6f} ms")
    print(f"speedup:   {ref_ms / opt_ms:.3f}x")


if __name__ == "__main__":
    main()
