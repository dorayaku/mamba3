"""KulaMamba benchmark."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from mamba3 import KulaMamba, KulaMambaBlock, KulaMambaLM


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_layer_forward(d_model, seq_len, batch, device, dtype, n_warmup=3, n_iter=10):
    layer = KulaMamba(
        d_model=d_model, d_state=64, expand=2, headdim=64,
        mimo_rank=2, chunk_size=64,
    ).to(device, dtype)
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)

    for _ in range(n_warmup):
        _ = layer(x)
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = layer(x)
    _sync()
    dt = (time.perf_counter() - t0) / n_iter

    return batch * seq_len / dt, dt * 1000


def benchmark_step_latency(d_model, batch, device, dtype, n_warmup=10, n_iter=100):
    layer = KulaMamba(
        d_model=d_model, d_state=64, expand=2, headdim=64,
        mimo_rank=2, chunk_size=64,
    ).to(device, dtype).eval()
    state = layer.allocate_inference_state(batch, dtype=dtype, device=device)
    x = torch.randn(batch, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(n_warmup):
            _, _ = layer.step(x, state)
        _sync()

        state = layer.allocate_inference_state(batch, dtype=dtype, device=device)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _, state = layer.step(x, state)
        _sync()
        dt = (time.perf_counter() - t0) / n_iter

    return dt * 1000


def benchmark_prefill_vs_sequential(d_model, seq_len, batch, device, dtype, n_warmup=3, n_iter=10):
    layer = KulaMamba(
        d_model=d_model, d_state=64, expand=2, headdim=64,
        mimo_rank=2, chunk_size=64,
    ).to(device, dtype).eval()
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(n_warmup):
            _, _ = layer.prefill(x)
        _sync()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _, _ = layer.prefill(x)
        _sync()
        dt_par = (time.perf_counter() - t0) / n_iter

        state = layer.allocate_inference_state(batch, dtype=dtype, device=device)
        for _ in range(n_warmup):
            state = layer.allocate_inference_state(batch, dtype=dtype, device=device)
            for t in range(seq_len):
                _, state = layer.step(x[:, t:t+1], state)
        _sync()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            state = layer.allocate_inference_state(batch, dtype=dtype, device=device)
            for t in range(seq_len):
                _, state = layer.step(x[:, t:t+1], state)
        _sync()
        dt_seq = (time.perf_counter() - t0) / n_iter

    return dt_par * 1000, dt_seq * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default='bf16', choices=['fp32', 'fp16', 'bf16'])
    args = parser.parse_args()

    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = args.device

    print(f"Device: {device}, Dtype: {args.dtype}\n")

    print("=== Training Throughput ===")
    print(f"{'d_model':>8} {'seq_len':>8} {'batch':>6} {'tok/s':>12} {'ms/batch':>10}")
    for d_model in [256, 512, 768, 1024]:
        for seq_len in [256, 1024]:
            batch = 4
            try:
                tps, ms = benchmark_layer_forward(d_model, seq_len, batch, device, dtype)
                print(f"{d_model:>8} {seq_len:>8} {batch:>6} {tps:>12,.0f} {ms:>10.2f}")
            except Exception as e:
                print(f"{d_model:>8} {seq_len:>8} {batch:>6} {'ERROR':>12} {str(e)[:30]}")
    print()

    print("=== Step Latency ===")
    print(f"{'d_model':>8} {'batch':>6} {'ms/step':>10}")
    for d_model in [256, 512, 768, 1024]:
        for batch in [1, 8]:
            try:
                ms = benchmark_step_latency(d_model, batch, device, dtype)
                print(f"{d_model:>8} {batch:>6} {ms:>10.3f}")
            except Exception as e:
                print(f"{d_model:>8} {batch:>6} {'ERROR':>10} {str(e)[:30]}")
    print()

    print("=== Prefill: Parallel vs Sequential ===")
    print(f"{'d_model':>8} {'seq_len':>8} {'parallel_ms':>12} {'sequential_ms':>14} {'speedup':>8}")
    for d_model in [256, 512]:
        for seq_len in [64, 256, 512]:
            try:
                par_ms, seq_ms = benchmark_prefill_vs_sequential(d_model, seq_len, 1, device, dtype)
                speedup = seq_ms / par_ms if par_ms > 0 else float('inf')
                print(f"{d_model:>8} {seq_len:>8} {par_ms:>12.2f} {seq_ms:>14.2f} {speedup:>7.1f}x")
            except Exception as e:
                print(f"{d_model:>8} {seq_len:>8} {'ERROR':>12} {str(e)[:30]}")
    print()

    if device == 'cuda':
        print("=== GPU Memory (d=768, T=1024, B=4) ===")
        torch.cuda.reset_peak_memory_stats()
        layer = KulaMamba(d_model=768, d_state=64, expand=2, headdim=64, mimo_rank=2).to(device, dtype)
        x = torch.randn(4, 1024, 768, device=device, dtype=dtype)
        _ = layer(x)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        params = sum(p.numel() * p.element_size() for p in layer.parameters()) / 1024**2
        print(f"  Parameters:  {params:.1f} MB")
        print(f"  Peak memory: {peak:.1f} MB")


if __name__ == '__main__':
    main()
