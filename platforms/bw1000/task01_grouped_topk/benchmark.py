import argparse
import statistics
import time

import torch

from reference import Model as ReferenceModel, get_init_inputs, get_inputs
from solution import ModelNew


def elapsed_ms(fn, warmup: int, repeat: int) -> float:
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
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=300)
    args = parser.parse_args()

    inputs = [x.cuda() for x in get_inputs()]
    reference = ReferenceModel(*get_init_inputs()).cuda()
    optimized = ModelNew(*get_init_inputs()).cuda()
    optimized.num_warps = args.num_warps

    expected_weights, expected_ids = reference(*inputs)
    actual_weights, actual_ids = optimized(*inputs)
    torch.cuda.synchronize()
    if not torch.equal(actual_ids, expected_ids):
        print("reference ids[0]:", expected_ids[0].cpu().tolist())
        print("optimized ids[0]:", actual_ids[0].cpu().tolist())
        for row in range(5):
            descending = actual_ids[row].cpu().tolist()
            permutation = [descending.index(x) for x in expected_ids[row].cpu().tolist()]
            print(f"output rank permutation[{row}]:", permutation)
        set_mismatch = sum(
            set(expected_ids[i].cpu().tolist()) != set(actual_ids[i].cpu().tolist())
            for i in range(expected_ids.shape[0])
        )
        print("top-k set mismatch rows:", set_mismatch)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-2, atol=1e-2)
    if not torch.equal(actual_ids, expected_ids):
        mismatch = int((actual_ids != expected_ids).sum().item())
        raise AssertionError(f"top-k id mismatch count: {mismatch}")

    ref_ms = elapsed_ms(lambda: reference(*inputs), args.warmup, args.repeat)
    opt_ms = elapsed_ms(lambda: optimized(*inputs), args.warmup, args.repeat)
    print(f"correctness: PASS (num_warps={args.num_warps})")
    print(f"reference: {ref_ms:.6f} ms")
    print(f"optimized: {opt_ms:.6f} ms")
    print(f"speedup:   {ref_ms / opt_ms:.3f}x")


if __name__ == "__main__":
    main()
