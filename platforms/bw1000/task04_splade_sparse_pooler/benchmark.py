import argparse
import statistics
import time

import torch

import reference as reference_module
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
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--block-n", type=int, default=512)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--materialized", action="store_true")
    parser.add_argument("--direct-block-n", type=int, default=64)
    parser.add_argument("--direct-block-k", type=int, default=64)
    parser.add_argument("--direct-num-warps", type=int, default=4)
    parser.add_argument("--direct-all-segments", action="store_true")
    args = parser.parse_args()

    reference = reference_module.Model(*reference_module.get_init_inputs()).cuda().eval()
    optimized = ModelNew(*reference_module.get_init_inputs()).cuda().eval()
    optimized.load_state_dict(reference.state_dict())
    optimized.pool_block_n = args.block_n
    optimized.pool_num_warps = args.num_warps
    optimized.direct_decoder_pool = not args.materialized
    optimized.direct_block_n = args.direct_block_n
    optimized.direct_block_k = args.direct_block_k
    optimized.direct_num_warps = args.direct_num_warps
    optimized.direct_all_segments = args.direct_all_segments
    inputs = reference_module.get_inputs()

    with torch.no_grad():
        expected = reference(*inputs)
        actual = optimized(*inputs)
    for index, (lhs, rhs) in enumerate(zip(expected, actual)):
        max_abs_diff = (lhs.float() - rhs.float()).abs().max().item()
        print(f"output[{index}] max_abs_diff={max_abs_diff:.8e}")
        torch.testing.assert_close(rhs, lhs, atol=1e-2, rtol=1e-2)

    with torch.no_grad():
        reference_us = measure(reference, inputs, args.warmup, args.repeat)
        optimized_us = measure(optimized, inputs, args.warmup, args.repeat)
    print("PASS accuracy")
    print(f"reference_median_us={reference_us:.3f}")
    print(f"optimized_median_us={optimized_us:.3f}")
    print(f"speedup={reference_us / optimized_us:.3f}x")


if __name__ == "__main__":
    main()
