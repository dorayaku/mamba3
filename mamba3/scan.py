"""Inter-chunk SSD scan.

h[0] = h_init, h[k+1] = decay[k] * h[k] + h_chunk[k]

CUDA kernel JIT-compiled at import (not in forward) for torch.compile.
"""

from __future__ import annotations

from pathlib import Path

import torch


def _try_load_cuda_ext():
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        csrc = Path(__file__).parent / "csrc"
        return load(
            name="mamba3_scan",
            sources=[str(csrc / "scan.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        return None


_cuda_ext = _try_load_cuda_ext()


def _scan_forward_torch(decay, h_chunk, h_init=None):
    B, K, H = decay.shape
    PN = h_chunk.shape[-1]
    d = decay.float()
    hf = h_chunk.float()
    h = h_init.float() if h_init is not None else hf.new_zeros(B, H, PN)
    out = [h]
    for k in range(K - 1):
        h = h * d[:, k, :, None] + hf[:, k]
        out.append(h)
    return torch.stack(out, dim=1)


def _scan_backward_torch(decay, h_states, grad_h, has_h_init=False):
    B, K, H, PN = h_states.shape
    d = decay.float()
    hs = h_states.float()
    dh = grad_h.float().clone()
    d_decay = torch.zeros_like(decay)
    d_chunk = torch.zeros_like(h_states)

    for k in range(K - 2, -1, -1):
        g = dh[:, k + 1]
        d_chunk[:, k] = g
        d_decay[:, k] = (g * hs[:, k]).sum(dim=-1)
        dh[:, k] = dh[:, k] + d[:, k, :, None] * g

    d_h_init = dh[:, 0] if has_h_init else None
    return d_decay, d_chunk, d_h_init


class SSDScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, chunk_decay, h_chunk, h_init=None):
        use_cuda = _cuda_ext is not None and chunk_decay.is_cuda

        if use_cuda:
            orig_shape = h_chunk.shape
            flat = h_chunk.float().contiguous().reshape(*orig_shape[:3], -1)
            h_init_flat = None
            if h_init is not None:
                h_init_flat = (
                    h_init.float().contiguous()
                    .reshape(h_init.shape[0], h_init.shape[1], -1)
                )
            h_states = _cuda_ext.scan_forward(
                chunk_decay.float().contiguous(),
                flat,
                h_init_flat,
            )
            h_states = h_states.reshape(orig_shape)
        else:
            orig_shape = h_chunk.shape
            flat = h_chunk.reshape(*orig_shape[:3], -1)
            h_init_flat = (
                h_init.reshape(h_init.shape[0], h_init.shape[1], -1)
                if h_init is not None else None
            )
            h_states = _scan_forward_torch(chunk_decay, flat, h_init_flat)
            h_states = h_states.reshape(orig_shape)

        ctx.save_for_backward(chunk_decay, h_states)
        ctx.has_h_init = h_init is not None
        return h_states.to(h_chunk.dtype)

    @staticmethod
    def backward(ctx, grad_h):
        chunk_decay, h_states = ctx.saved_tensors
        use_cuda = _cuda_ext is not None and grad_h.is_cuda

        if use_cuda:
            orig_shape = h_states.shape
            hs_flat = h_states.float().contiguous().reshape(*orig_shape[:3], -1)
            gh_flat = grad_h.float().contiguous().reshape(*orig_shape[:3], -1)
            d_decay, d_chunk_flat, d_h_init_flat = _cuda_ext.scan_backward(
                chunk_decay.float().contiguous(),
                hs_flat,
                gh_flat,
                ctx.has_h_init,
            )
            d_chunk = d_chunk_flat.reshape(orig_shape)
            if d_h_init_flat is not None:
                d_h_init = d_h_init_flat.reshape(
                    orig_shape[0], orig_shape[2], *orig_shape[3:]
                )
            else:
                d_h_init = None
        else:
            orig_shape = h_states.shape
            hs_flat = h_states.reshape(*orig_shape[:3], -1)
            gh_flat = grad_h.reshape(*orig_shape[:3], -1)
            d_decay, d_chunk_flat, d_h_init_flat = _scan_backward_torch(
                chunk_decay, hs_flat, gh_flat, ctx.has_h_init,
            )
            d_chunk = d_chunk_flat.reshape(orig_shape)
            if d_h_init_flat is not None:
                d_h_init = d_h_init_flat.reshape(
                    orig_shape[0], orig_shape[2], *orig_shape[3:]
                )
            else:
                d_h_init = None

        return (
            d_decay.to(chunk_decay.dtype),
            d_chunk.to(grad_h.dtype),
            d_h_init.to(grad_h.dtype) if d_h_init is not None else None,
        )


def ssd_scan(chunk_decay, h_chunk, h_init=None):
    return SSDScanFn.apply(chunk_decay, h_chunk, h_init)
